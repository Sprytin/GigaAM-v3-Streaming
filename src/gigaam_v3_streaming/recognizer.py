"""Live recognizer: partial hypotheses, endpointed finals and word times.

Two bounded lanes run over the same sample clock:

* the *stable* lane (default 240 ms steps, 8 s left / 800 ms right context)
  produces text that never changes;
* the *preview* lane (160 ms steps, 3.2 s left / 320 ms right context) runs
  ahead of it and only fills the unsettled tail of the partial hypothesis.

An utterance ends when the stable lane has settled ``endpoint_ms`` of audio
after its last token, so no extra voice-activity model is needed. Ending the
stream pads silence so the last words get the same lookahead as the rest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from .config import StreamingConfig
from .session import Backend, StreamingSession, coerce_pcm

# A window edge can make the model emit punctuation as its own word piece
# ("TV ."); it belongs to the word before it.
_PUNCTUATION = re.compile(r"^[.,!?;:…»)\]]+$")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+(?=[.,!?;:…»)\]])")


@dataclass(frozen=True, slots=True)
class Word:
    """One recognized word; times are RNN-T emission times in seconds."""

    text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class Partial:
    """The current utterance hypothesis. It may change until its final.

    ``stable_text`` is the settled prefix of ``text``; the rest comes from the
    preview lane and may still be revised.
    """

    text: str
    stable_text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class Final:
    """One finished utterance. It never changes and supersedes its partials."""

    text: str
    start: float
    end: float
    words: tuple[Word, ...]


Event = Union[Partial, Final]


def _default_stable() -> StreamingConfig:
    return StreamingConfig(emit_ms=240, left_context_ms=8_000, right_context_ms=800)


def _default_preview() -> StreamingConfig:
    return StreamingConfig(emit_ms=160, left_context_ms=3_200, right_context_ms=320)


MODES = ("gpu", "cpu", "cpu-preview")
"""The supported modes; README "Choose your mode" has their measured trade-offs."""


@dataclass(frozen=True, slots=True)
class RecognizerConfig:
    """Lane schedules and utterance segmentation.

    ``preview=None`` turns the preview lane off (partials then carry stable
    text only, for about half the compute). ``endpoint_ms=None`` turns
    endpointing off: one final per stream, at
    :meth:`StreamingRecognizer.finish`, unless ``max_utterance_ms`` cuts
    earlier.
    """

    stable: StreamingConfig = field(default_factory=_default_stable)
    preview: Optional[StreamingConfig] = field(default_factory=_default_preview)
    endpoint_ms: Optional[int] = 600
    max_utterance_ms: int = 30_000

    def __post_init__(self) -> None:
        if self.preview is not None and self.preview.sample_rate != self.stable.sample_rate:
            raise ValueError("stable and preview sample rates must match")
        if self.endpoint_ms is not None and self.endpoint_ms <= 0:
            raise ValueError("endpoint_ms must be positive or None")
        if self.max_utterance_ms <= 0:
            raise ValueError("max_utterance_ms must be positive")

    @classmethod
    def for_mode(cls, mode: str, **overrides: Any) -> RecognizerConfig:
        """The lane schedule a mode was measured with; ``overrides`` replace single fields.

        * ``gpu``: the defaults, stable 240/8000/800 ms (step / left / right
          context) and preview 160/3200/320 ms.
        * ``cpu``: stable 960/4000/800 ms, no preview. A four times longer step
          and half the left context cut the encoder work about 6x; finals keep
          the GPU accuracy, partials only show settled text.
        * ``cpu-preview``: the ``cpu`` stable lane plus a light preview lane,
          480/2400/320 ms, for earlier partials at about twice the compute.
        """
        if mode == "gpu":
            schedule: dict[str, Any] = {}
        elif mode in ("cpu", "cpu-preview"):
            schedule = {
                "stable": StreamingConfig(emit_ms=960, left_context_ms=4_000, right_context_ms=800),
                "preview": None
                if mode == "cpu"
                else StreamingConfig(emit_ms=480, left_context_ms=2_400, right_context_ms=320),
            }
        else:
            raise ValueError(f"unknown mode {mode!r}; choose one of {', '.join(MODES)}")
        return cls(**(schedule | overrides))


class StreamingRecognizer:
    """Feed audio, get :class:`Partial` and :class:`Final` events.

    One recognizer is one audio stream; the model may be shared by many
    recognizers used from one thread at a time. Finals, their text and their
    times do not depend on how the audio is chunked; partials are emitted
    whenever the hypothesis changes, so their count does.

    ``backend`` runs the model (see :class:`~gigaam_v3_streaming.session.Backend`);
    the default runs a stock GigaAM checkpoint.
    """

    def __init__(
        self,
        model: Any,
        config: Optional[RecognizerConfig] = None,
        backend: Optional[Backend] = None,
    ) -> None:
        self.config = config or RecognizerConfig()
        self.model = model
        self._stable = StreamingSession(model, self.config.stable, backend)
        self._preview = (
            StreamingSession(model, self.config.preview, backend) if self.config.preview else None
        )
        self._rate = self.config.stable.sample_rate
        self._hop = self._stable.alignment.encoder_hop_samples
        ms = self._rate // 1_000
        self._endpoint = None if self.config.endpoint_ms is None else self.config.endpoint_ms * ms
        self._max_utterance = self.config.max_utterance_ms * ms
        self._tokenizer = _tokenizer(model)
        self._utterance: list[tuple[int, int]] = []  # settled (token, sample)
        self._tail: list[tuple[int, int]] = []  # preview (token, sample) past the settled clock
        self._shown = ""
        self._stream_end = 0
        self._done = False

    @classmethod
    def from_pretrained(
        cls,
        name: str = "v3_e2e_rnnt",
        device: Optional[str] = None,
        config: Optional[RecognizerConfig] = None,
        mode: str = "auto",
        **load_kwargs: Any,
    ) -> StreamingRecognizer:
        """Load a stock GigaAM RNN-T checkpoint in one of the :data:`MODES` and open one stream.

        ``mode="auto"`` is ``gpu`` when CUDA is available, else ``cpu``. Keyword
        arguments go to :func:`gigaam_v3_streaming.load_engine` (``threads``,
        ``cache_dir``, ``download_root`` ...). To serve several streams, load
        the engine once and call :meth:`Engine.recognizer` per stream.
        """
        from .engine import load_engine

        return load_engine(mode, name, device=device, **load_kwargs).recognizer(config)

    @property
    def audio_seconds(self) -> float:
        return self._stream_end / self._rate

    @property
    def finished(self) -> bool:
        return self._done

    def feed(self, audio: Any, sample_offset: Optional[int] = None) -> list[Event]:
        """Append mono 16 kHz audio and return the events it produced.

        ``audio`` may be float samples, an int16 array or s16le bytes.
        ``sample_offset`` (absolute) is checked against the stream so gaps and
        overlaps fail loudly instead of shifting every later timestamp.
        """
        if self._done:
            raise RuntimeError("cannot feed after finish() or cancel()")
        pcm = coerce_pcm(audio)
        offset = self._stream_end if sample_offset is None else int(sample_offset)
        if offset != self._stream_end:
            relation = "gap" if offset > self._stream_end else "overlap"
            raise ValueError(f"{relation}: expected sample_offset {self._stream_end}, got {offset}")
        events: list[Event] = []
        at = 0
        while at < pcm.size:
            # Split on the encoder hop (every emit step is a whole number of
            # hops), so at most one stable window settles per piece:
            # segmentation then depends on the sample clock only.
            boundary = (self._stream_end // self._hop + 1) * self._hop
            take = min(int(pcm.size) - at, boundary - self._stream_end)
            piece, piece_offset = pcm[at : at + take], self._stream_end
            at += take
            self._stream_end += take
            if self._preview is not None:
                for event in self._preview.push(piece, piece_offset):
                    self._tail.extend(zip(event.token_ids, event.token_samples, strict=True))
            for event in self._stable.push(piece, piece_offset):
                events.extend(self._settle(zip(event.token_ids, event.token_samples, strict=True)))
            events.extend(self._after_step(self._stable.processed_samples))
        return events

    def finish(self) -> list[Event]:
        """End the stream: decode the tail with full lookahead, close it."""
        if self._done:
            return []
        if self._preview is not None:
            self._preview.cancel()
        self._tail = []
        end = self._stream_end
        events: list[Event] = []
        for event in self._stable.finish():
            tokens = zip(event.token_ids, event.token_samples, strict=True)
            events.extend(self._settle(tokens))
        events.extend(self._close(end))
        self._done = True
        return events

    def cancel(self) -> None:
        """Drop the stream without decoding or emitting anything more."""
        self._stable.cancel()
        if self._preview is not None:
            self._preview.cancel()
        self._utterance, self._tail = [], []
        self._done = True

    # Segmentation -------------------------------------------------------

    def _settle(self, tokens: Any) -> list[Event]:
        events: list[Event] = []
        for token, sample in tokens:
            # The model can emit the rest of a word or its punctuation well
            # after the word began, so an utterance only closes before a token
            # that starts a new word; a late piece joins the word it belongs to.
            if self._utterance and not self._continues(token):
                last = self._utterance[-1][1] + self._hop
                long = sample - self._utterance[0][1] >= self._max_utterance
                if (self._endpoint is not None and sample - last >= self._endpoint) or long:
                    events.extend(self._close(last))
            elif not self._utterance and self._is_punctuation(token):
                continue  # punctuation that arrives after its utterance closed
            self._utterance.append((token, sample))
        return events

    def _after_step(self, settled: int) -> list[Event]:
        events: list[Event] = []
        if self._utterance and self._endpoint is not None:
            last = self._utterance[-1][1] + self._hop
            if settled - last >= self._endpoint and not self._late_piece_ahead():
                events.extend(self._close(last))
        # The stable lane may emit a word up to one step later than the
        # preview lane did, so preview tokens that fall just behind the settled
        # clock stay until a stable token covers them; dropping them at the
        # clock makes the partial lose and regain words.
        covered = self._utterance[-1][1] if self._utterance else -1
        horizon = settled - self.config.stable.emit_samples
        self._tail = [item for item in self._tail if item[1] > covered and item[1] >= horizon]
        tokens = self._utterance + self._tail
        text = self._render([token for token, _ in tokens])
        if text and text != self._shown:
            self._shown = text
            events.append(
                Partial(
                    text=text,
                    stable_text=self._render([token for token, _ in self._utterance]),
                    start=tokens[0][1] / self._rate,
                    end=(tokens[-1][1] + self._hop) / self._rate,
                )
            )
        return events

    def _close(self, end: int) -> list[Event]:
        if not self._utterance:
            return []
        words = self._words(self._utterance, end)
        self._utterance = []
        self._shown = ""
        if not words:
            return []
        return [
            Final(
                text=" ".join(word.text for word in words),
                start=words[0].start,
                end=words[-1].end,
                words=tuple(words),
            )
        ]

    def _words(self, tokens: list[tuple[int, int]], limit: int) -> list[Word]:
        groups: list[list[tuple[int, int]]] = []
        for token, sample in tokens:
            if not groups or self._starts_word(token):
                groups.append([])
            groups[-1].append((token, sample))
        words: list[Word] = []
        for group in groups:
            text = self._render([token for token, _ in group])
            if not text:
                continue
            end = min(group[-1][1] + self._hop, max(limit, group[-1][1])) / self._rate
            if words and _PUNCTUATION.match(text):
                words[-1] = Word(words[-1].text + text, words[-1].start, end)
            else:
                words.append(Word(text, group[0][1] / self._rate, end))
        return words

    def _late_piece_ahead(self) -> bool:
        """The preview lane already holds the next token, and it continues the utterance."""
        covered = self._utterance[-1][1]
        ahead = [token for token, sample in self._tail if sample > covered]
        return bool(ahead) and self._continues(ahead[0])

    def _continues(self, token: int) -> bool:
        """A piece inside a word, or punctuation ("▁." is a piece of its own)."""
        return not self._starts_word(token) or self._is_punctuation(token)

    def _is_punctuation(self, token: int) -> bool:
        return bool(_PUNCTUATION.match(self._render([token])))

    def _starts_word(self, token: int) -> bool:
        piece = self._tokenizer.id_to_str(token)
        if getattr(self._tokenizer, "charwise", False):
            return piece == " "
        return piece.startswith("▁")

    def _render(self, token_ids: list[int]) -> str:
        pieces = "".join(self._tokenizer.id_to_str(token) for token in token_ids)
        text = " ".join(pieces.replace("▁", " ").split())
        return _SPACE_BEFORE_PUNCTUATION.sub("", text)


def _tokenizer(model: Any) -> Any:
    tokenizer = getattr(getattr(model, "decoding", None), "tokenizer", None)
    tokenizer = tokenizer or getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("model exposes no tokenizer (expected model.decoding.tokenizer)")
    return tokenizer


def load_model(name: str = "v3_e2e_rnnt", device: Optional[str] = None, **kwargs: Any) -> Any:
    """Load a GigaAM checkpoint and check that it has an RNN-T head.

    Checkpoints with pinned hashes (:data:`gigaam_v3_streaming.download.PINS`)
    are fetched with :func:`gigaam_v3_streaming.download.fetch` first, so a cut
    download never leaves a broken file behind.
    """
    import gigaam

    from .download import PINS, fetch

    if name in PINS:
        fetch(name, kwargs.get("download_root"))
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = gigaam.load_model(name, device=device, **kwargs)
    head = getattr(model, "head", None)
    if not (hasattr(head, "joint") and hasattr(head, "decoder")):
        raise ValueError(f"{name} is not an RNN-T model; use an *_rnnt checkpoint")
    return model
