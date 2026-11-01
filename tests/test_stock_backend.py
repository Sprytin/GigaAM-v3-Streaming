"""The real backend path on a tiny torch RNN-T built like stock GigaAM.

Runs on CPU in a second: ``StockGigaAM`` (dtype and device cast,
``model.forward`` returning ``[B, D, T]``), the central-frame map and
``RNNTGreedyState`` (``head.decoder.predict``, ``head.joint.joint``, decoder
state carried across windows). The tiny encoder is frame-local, so streaming
must reproduce an offline greedy decode of the whole clip token for token.
"""

from __future__ import annotations

import numpy as np
import pytest

from gigaam_v3_streaming import Final, RecognizerConfig, StreamingRecognizer, StreamingSession
from gigaam_v3_streaming.recognizer import _default_stable

torch = pytest.importorskip("torch")
nn = torch.nn

RATE = 16_000
HOP = 640
PIECES = ["▁а", "▁бе", "ве", "▁го", "да", "▁е"]
BLANK = len(PIECES)


class Tokenizer:
    charwise = False

    def id_to_str(self, token: int) -> str:
        return PIECES[token]


class Preprocessor:
    hop_length = 160
    win_length = 400


class PreEncode:
    _sampling_num = 2  # x4 subsampling: 160-sample hop -> 640-sample frames


class Encoder(nn.Module):
    """One frame per 640 samples; each piece is a loudness band, silence is blank."""

    def __init__(self) -> None:
        super().__init__()
        self.pre_encode = PreEncode()
        self.register_buffer("centers", torch.linspace(0.08, 0.58, len(PIECES)))

    def forward(self, wav, length):
        frames = -(-wav.shape[-1] // HOP)
        wav = nn.functional.pad(wav, (0, frames * HOP - wav.shape[-1]))
        rms = wav.view(1, frames, HOP).pow(2).mean(-1).sqrt().unsqueeze(-1)  # [1, T, 1]
        pieces = 1 - 20 * (rms - self.centers).abs()
        blank = torch.where(rms < 0.03, 5.0, 0.0).to(rms.dtype)
        return torch.cat([pieces, blank], dim=-1).transpose(1, 2), length // HOP


class Decoder(nn.Module):
    """Predictor whose output forbids repeating the last emitted piece."""

    blank_id = BLANK

    def predict(self, label, state, batch_size=1):
        g = torch.zeros(batch_size, 1, BLANK + 1, dtype=torch.float64)
        if label is None:
            return g, torch.zeros(1)  # like an LSTM, a real (initial) state
        return g - 10 * nn.functional.one_hot(label, BLANK + 1).to(g.dtype), state + 1


class Joint(nn.Module):
    def joint(self, f, g):
        return f.unsqueeze(2) + g.unsqueeze(1)  # [B, T, U, V]


class Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = Decoder()
        self.joint = Joint()


class Decoding:
    tokenizer = Tokenizer()


class TinyGigaAM(nn.Module):
    """The attributes and calls the stock path uses, at toy size."""

    def __init__(self) -> None:
        super().__init__()
        self.preprocessor = Preprocessor()
        self.encoder = Encoder()
        self.head = Head()
        self.decoding = Decoding()
        self._device = torch.device("cpu")
        self._dtype = torch.float64  # the window arrives as float32: exercises the cast
        self.double()
        self.calls: list[int] = []

    def forward(self, wav, length):
        assert wav.dtype == self._dtype and int(length[0]) == wav.shape[-1]
        if wav.shape[-1] < Preprocessor.win_length:  # the stock STFT frontend fails too
            raise RuntimeError("window shorter than the frontend's win_length")
        self.calls.append(wav.shape[-1])
        return self.encoder(wav, length)


def offline_greedy(model, audio, max_symbols=10):
    """GigaAM's offline greedy RNN-T decode of the whole clip: (token, frame)."""
    wav = torch.from_numpy(audio).to(torch.float64).unsqueeze(0)
    with torch.inference_mode():
        encoded, _ = model.encoder(wav, torch.tensor([wav.shape[-1]]))
        label, state, out = None, None, []
        for index in range(encoded.shape[-1]):
            f = encoded[:, :, index].view(1, 1, -1)
            for _ in range(max_symbols):
                g, hidden = model.head.decoder.predict(label, state)
                token = int(model.head.joint.joint(f, g)[0, 0, 0].argmax())
                if token == BLANK:
                    break
                out.append((token, index))
                label, state = torch.tensor([[token]]), hidden
    return out


def clip(seconds=7.0, seed=3):
    """Noise bursts of different loudness between silences."""
    rng = np.random.default_rng(seed)
    audio = np.zeros(int(seconds * RATE) + 123, dtype=np.float32)  # not a whole frame
    at = 2_000
    while at < audio.size - 4_000:
        length = int(rng.integers(2_000, 9_000))
        audio[at : at + length] = rng.normal(0, rng.uniform(0.05, 0.6), length)
        at += length + int(rng.integers(1_000, 12_000))
    return audio


@pytest.fixture(scope="module")
def model():
    return TinyGigaAM()


@pytest.fixture(scope="module")
def audio():
    return clip()


@pytest.fixture(scope="module")
def reference(model, audio):
    tokens = offline_greedy(model, audio)
    assert 10 < len(tokens) < 200  # the toy model emits something worth comparing
    return tokens


@pytest.mark.parametrize("chunk", [1_280, 4_000, 17_011])
def test_session_reproduces_offline_greedy_decode(model, audio, reference, chunk):
    session = StreamingSession(model, _default_stable())
    events = []
    for at in range(0, audio.size, chunk):
        events.extend(session.push(audio[at : at + chunk]))
    events.extend(session.finish())
    streamed = [
        (token, sample)
        for event in events
        for token, sample in zip(event.token_ids, event.token_samples, strict=True)
    ]
    assert streamed == [(token, frame * HOP) for token, frame in reference]
    assert events[-1].is_final and events[-1].end_sample == audio.size


def test_recognizer_finals_carry_the_offline_text(model, audio, reference):
    recognizer = StreamingRecognizer(model)
    events = []
    for at in range(0, audio.size, 2_560):
        events.extend(recognizer.feed(audio[at : at + 2_560]))
    events.extend(recognizer.finish())
    finals = [event for event in events if isinstance(event, Final)]
    expected = "".join(PIECES[token] for token, _ in reference).replace("▁", " ").split()
    assert " ".join(final.text for final in finals).split() == expected
    words = [word for final in finals for word in final.words]
    assert all(0 <= word.start < word.end <= audio.size / RATE for word in words)
    assert all(round(word.start * RATE) % HOP == 0 for word in words)


def test_a_stream_shorter_than_one_frame_decodes_cleanly(model):
    # Windows always span the emit step and right context (finish pads the
    # tail), so the encoder never sees less than the frontend's win_length.
    model.calls.clear()
    recognizer = StreamingRecognizer(model, RecognizerConfig(preview=None))
    tiny = np.full(100, 0.3, dtype=np.float32)
    assert recognizer.feed(tiny) == []
    finals = [event for event in recognizer.finish() if isinstance(event, Final)]
    assert [final.text for final in finals] == [PIECES[offline_greedy(model, tiny)[0][0]][1:]]
    assert model.calls and min(model.calls) >= Preprocessor.win_length
