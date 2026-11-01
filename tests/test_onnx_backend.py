"""The CPU modes' encoder in ONNX Runtime, its optimized-graph cache and the lean model.

A tiny encoder built like the one in ``test_stock_backend.py`` (one frame per
640 samples, pieces by loudness) is exported to ONNX, so streaming through
ONNX Runtime must reproduce an offline decode of the same graph token for
token, with and without the cached optimized graph.
"""

from __future__ import annotations

import warnings

import pytest

from gigaam_v3_streaming import Final, RecognizerConfig, StreamingRecognizer
from gigaam_v3_streaming.config import StreamingConfig

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

from test_stock_backend import HOP, PIECES, RATE, Decoding, Head, clip  # noqa: E402

from gigaam_v3_streaming.onnx_backend import (  # noqa: E402
    WEIGHTS,
    OnnxEncoder,
    open_session,
    optimized_model_path,
)

nn = torch.nn
FEATURE_HOP = 160


class Preprocessor(nn.Module):
    """Mean square of each 160-sample hop: features [1, 1, T]."""

    hop_length = FEATURE_HOP
    sample_rate = RATE

    def forward(self, wav, length):
        frames = -(-wav.shape[-1] // FEATURE_HOP)
        wav = nn.functional.pad(wav, (0, frames * FEATURE_HOP - wav.shape[-1]))
        return wav.view(1, 1, frames, FEATURE_HOP).pow(2).mean(-1), (length + 159) // 160


class TinyEncoder(nn.Module):
    """x4 subsampling of the features, then pieces by loudness as in ``test_stock_backend``."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.08, 0.58, len(PIECES)))
        # A weight big enough (> 1 KB) to be stored beside the cached graph, as a real one is.
        self.register_buffer("spread", torch.full((64, len(PIECES)), 1 / 64))

    def forward(self, features, lengths):
        frames = (features.shape[-1] + 3) // 4
        features = nn.functional.pad(features, (0, frames * 4 - features.shape[-1]))
        rms = features.view(1, frames, 4).mean(-1).sqrt().unsqueeze(-1)  # [1, T, 1]
        pieces = 1 - 20 * (rms.expand(1, frames, 64) @ self.spread - self.centers).abs()
        blank = (rms < 0.03).to(rms.dtype) * 5.0
        return torch.cat([pieces, blank], dim=-1).transpose(1, 2), (lengths + 3) // 4


class TinyCpuModel:
    """What the CPU modes keep of a model: frontend, head, tokenizer, frame geometry."""

    subsampling_factor = 4

    def __init__(self) -> None:
        self.preprocessor = Preprocessor()
        self.head = Head()
        self.decoding = Decoding()


@pytest.fixture(scope="module")
def encoder_onnx(tmp_path_factory):
    path = tmp_path_factory.mktemp("onnx") / "tiny_encoder.onnx"
    features = torch.rand(1, 1, 400)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            TinyEncoder(),
            (features, torch.tensor([400])),
            str(path),
            input_names=["features", "lengths"],
            output_names=["encoded", "encoded_lengths"],
            dynamic_axes={"features": {2: "time"}, "encoded": {2: "frames"}},
            opset_version=17,
            dynamo=False,  # the TorchScript exporter, as GigaAM's to_onnx uses
        )
    return path


@pytest.fixture(scope="module")
def audio():
    return clip()


def offline_tokens(backend, audio):
    decoder = backend.new_decoder(10)
    return [(token, frame * HOP) for token, frame in decoder.decode(backend.encode(audio))]


def streamed_tokens(model, backend, audio, chunk=3_000):
    recognizer = StreamingRecognizer(model, RecognizerConfig.for_mode("cpu"), backend)
    events = []
    for at in range(0, audio.size, chunk):
        events.extend(recognizer.feed(audio[at : at + chunk]))
    events.extend(recognizer.finish())
    return [event for event in events if isinstance(event, Final)]


def test_onnx_streaming_reproduces_the_offline_decode(encoder_onnx, audio):
    model = TinyCpuModel()
    backend = OnnxEncoder(model, encoder_onnx, threads=1)
    assert backend.cached_graph is None
    reference = offline_tokens(backend, audio)
    assert 10 < len(reference) < 200
    finals = streamed_tokens(model, backend, audio)
    expected = "".join(PIECES[token] for token, _ in reference).replace("▁", " ").split()
    assert " ".join(final.text for final in finals).split() == expected


def test_graph_cache_is_built_once_and_gives_the_same_output(encoder_onnx, audio, tmp_path):
    model = TinyCpuModel()
    first = OnnxEncoder(model, encoder_onnx, threads=1, cache_dir=tmp_path)
    assert first.cached_graph == optimized_model_path(encoder_onnx, 1, tmp_path)
    assert first.cached_graph.is_file()
    stamp = first.cached_graph.stat().st_mtime_ns
    second = OnnxEncoder(model, encoder_onnx, threads=1, cache_dir=tmp_path)
    assert second.cached_graph == first.cached_graph
    assert second.cached_graph.stat().st_mtime_ns == stamp  # loaded, not rebuilt
    window = audio[: 5 * RATE]
    assert torch.equal(first.encode(window), second.encode(window))
    assert torch.equal(second.encode(window), OnnxEncoder(model, encoder_onnx, 1).encode(window))
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []
    # The weights live beside the graph, so ONNX Runtime maps them instead of copying.
    assert (first.cached_graph.parent / WEIGHTS).stat().st_size > 1024


def test_graph_cache_key_covers_runtime_threads_and_cpu(encoder_onnx, tmp_path, monkeypatch):
    from gigaam_v3_streaming import onnx_backend

    one = optimized_model_path(encoder_onnx, 1, tmp_path)
    assert f"ort{ort.__version__}" in one.parent.name
    assert one.parent.name.startswith("tiny_encoder-")
    assert optimized_model_path(encoder_onnx, 3, tmp_path) != one
    monkeypatch.setattr(onnx_backend, "cpu_isa", lambda: "aarch64-00000000")
    assert optimized_model_path(encoder_onnx, 1, tmp_path) != one
    monkeypatch.setattr(ort, "__version__", "0.0.0")
    assert "ort0.0.0" in optimized_model_path(encoder_onnx, 1, tmp_path).parent.name


@pytest.mark.parametrize("damage", ["graph", "weights"])
def test_a_broken_cache_entry_is_rebuilt(encoder_onnx, audio, tmp_path, damage):
    path = OnnxEncoder(TinyCpuModel(), encoder_onnx, threads=1, cache_dir=tmp_path).cached_graph
    size = path.stat().st_size
    if damage == "graph":
        path.write_bytes(b"half a file")
    else:
        weights = path.parent / WEIGHTS
        weights.write_bytes(weights.read_bytes()[:100])  # what a full disk leaves
    with pytest.warns(UserWarning, match="rebuilding"):
        backend = OnnxEncoder(TinyCpuModel(), encoder_onnx, threads=1, cache_dir=tmp_path)
    assert path.stat().st_size == size
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []
    reference = OnnxEncoder(TinyCpuModel(), encoder_onnx, threads=1)
    assert torch.equal(backend.encode(audio[:RATE]), reference.encode(audio[:RATE]))


def test_an_unwritable_cache_only_costs_the_speed_up(encoder_onnx, tmp_path):
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    with pytest.warns(UserWarning, match="cache disabled"):
        session, graph = open_session(encoder_onnx, 1, blocked / "cache")
    assert graph is None and session.get_inputs()[0].name == "features"


# --- the lean CPU model from a checkpoint -----------------------------------------------------


def test_cpu_model_reads_frontend_and_head_but_not_the_encoder(tmp_path):
    pytest.importorskip("gigaam")
    from omegaconf import OmegaConf

    from gigaam_v3_streaming.cpu import CpuModel

    vocabulary = list("абв ")
    cfg = OmegaConf.create(
        {
            "preprocessor": {
                "_target_": "gigaam.preprocess.FeatureExtractor",
                "sample_rate": 16_000,
                "features": 8,
                "win_length": 320,
                "hop_length": 160,
                "n_fft": 320,
                "center": False,
            },
            "encoder": {"subsampling_factor": 4, "d_model": 6},
            "head": {
                "_target_": "gigaam.decoder.RNNTHead",
                "decoder": {"pred_hidden": 5, "pred_rnn_layers": 1, "num_classes": 5},
                "joint": {"enc_hidden": 6, "pred_hidden": 5, "joint_hidden": 7, "num_classes": 5},
            },
            "decoding": {
                "_target_": "gigaam.decoding.RNNTGreedyDecoding",
                "vocabulary": vocabulary,
            },
        }
    )
    import hydra

    head = hydra.utils.instantiate(cfg.head)
    preprocessor = hydra.utils.instantiate(cfg.preprocessor)
    state = {f"head.{key}": value for key, value in head.state_dict().items()}
    state |= {f"preprocessor.{key}": value for key, value in preprocessor.state_dict().items()}
    state["encoder.layers.0.weight"] = torch.full((4, 4), float("nan"))  # must never be read
    torch.save({"cfg": cfg, "state_dict": state}, tmp_path / "toy.ckpt")

    model = CpuModel(tmp_path / "toy.ckpt", None)
    assert model.subsampling_factor == 4 and not hasattr(model, "encoder")
    for key, value in head.state_dict().items():
        assert torch.equal(model.head.state_dict()[key], value)
    alignment = StreamingConfig().validate(model)
    assert alignment.encoder_hop_samples == HOP
    wav = torch.from_numpy(clip()[:RATE]).unsqueeze(0)
    features, _ = model.preprocessor(wav, torch.tensor([wav.shape[-1]]))
    assert features.shape[1] == 8 and torch.isfinite(features).all()
