"""CPU backend: GigaAM's encoder exported to ONNX (INT8) under ONNX Runtime.

The log-mel frontend and the RNN-T predictor and joint stay the stock PyTorch
modules of the checkpoint; only the encoder, which is almost all of the
compute, runs in ONNX Runtime. :mod:`gigaam_v3_streaming.cpu` builds the INT8
encoder from the checkpoint and wires everything up.

ONNX Runtime spends most of a cold start optimizing the graph. With a
``cache_dir`` the optimized graph is saved once and later sessions load it
instead, memory-mapping its weights (:func:`open_session`). The optimized
graph is specific to the ONNX Runtime version, the thread count and the CPU's
instruction set (ORT picks kernels and layouts for them), so all three are part
of the cache key, as in gigastt's session cache
(https://github.com/ekhodzitsky/gigastt, MIT; the idea, no code).
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .download import file_sha256
from .rnnt import RNNTGreedyState
from .session import FrameDecoder

# CPU features that change which kernels ONNX Runtime picks for an INT8 or fp32 graph.
_ISA_FLAGS = {
    "sse4_1", "sse4_2", "avx", "avx2", "fma", "f16c", "avx512f", "avx512bw", "avx512vl",
    "avx512dq", "avx512_vnni", "avx_vnni", "avx512_bf16", "amx_tile", "amx_int8", "amx_bf16",
    "asimd", "asimddp", "i8mm", "sve", "sve2", "bf16", "fphp", "asimdhp",
}  # fmt: skip


def cpu_isa() -> str:
    """``<machine>-<hash of the relevant CPU feature flags>``, e.g. ``x86_64-1f0c2a9e``."""
    flags: set[str] = set()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                key, _, value = line.partition(":")
                if key.strip() in ("flags", "Features"):
                    flags |= set(value.split()) & _ISA_FLAGS
    except OSError:  # not Linux: the processor string is the best stable stand-in
        flags = {platform.processor()}
    digest = hashlib.sha256(" ".join(sorted(flags)).encode()).hexdigest()[:8]
    return f"{platform.machine() or 'cpu'}-{digest}"


GRAPH = "encoder.onnx"  # the optimized graph inside a cache entry
WEIGHTS = "encoder.data"  # its weights, which ONNX Runtime memory-maps


def optimized_model_path(encoder_path: os.PathLike | str, threads: int, cache_dir: Path) -> Path:
    """Where the optimized graph of ``encoder_path`` for this runtime, threads and CPU lives."""
    import onnxruntime as ort

    encoder_path = Path(encoder_path)
    key = "-".join(
        (
            encoder_path.stem,
            file_sha256(encoder_path)[:16],
            f"ort{ort.__version__}",
            f"t{threads}",
            cpu_isa(),
        )
    )
    return Path(cache_dir) / key / GRAPH


def _options(threads: int, optimize: bool = True) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    return options


def _discard(entry: Path) -> None:
    """Move a cache entry out of the way first, so no reader sees it half removed."""
    try:
        aside = Path(tempfile.mkdtemp(dir=entry.parent, prefix=f".{entry.name}.old-"))
        os.replace(entry, aside / entry.name)
        shutil.rmtree(aside, ignore_errors=True)
    except OSError:
        shutil.rmtree(entry, ignore_errors=True)


def open_session(
    encoder_path: os.PathLike | str, threads: int, cache_dir: Optional[os.PathLike | str] = None
) -> tuple[Any, Optional[Path]]:
    """An ONNX Runtime CPU session for the encoder; returns it and the cached graph used or made.

    Without ``cache_dir`` the graph is optimized on every start. With it, the
    optimized graph is saved once as ONNX with its weights in a separate file,
    and later sessions load it with optimization off: ONNX Runtime then
    memory-maps the weights instead of copying them, so a start is faster and
    the weights are shared by every process on the machine through the page
    cache. A cache entry is built in a temporary directory and renamed into
    place, so a reader never sees half of one; one that does not load is
    rebuilt; a cache that cannot be written only costs the speed-up.
    """
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    if cache_dir is None:
        session = ort.InferenceSession(
            str(encoder_path), sess_options=_options(threads), providers=providers
        )
        return session, None
    cached = optimized_model_path(encoder_path, threads, Path(cache_dir))
    entry = cached.parent
    if cached.is_file():
        try:
            return ort.InferenceSession(
                str(cached), sess_options=_options(threads, optimize=False), providers=providers
            ), cached
        except Exception as error:  # noqa: BLE001 - any unreadable cache is rebuilt
            warnings.warn(
                f"rebuilding unreadable ONNX Runtime cache {entry}: {error}", stacklevel=2
            )
    options = _options(threads)
    work = None
    try:
        entry.parent.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(dir=entry.parent, prefix=f".{entry.name}."))
        options.optimized_model_filepath = str(work / GRAPH)
        options.add_session_config_entry(
            "session.optimized_model_external_initializers_file_name", WEIGHTS
        )
        options.add_session_config_entry(
            "session.optimized_model_external_initializers_min_size_in_bytes", "1024"
        )
        # ORT warns that the saved graph suits only this CPU; the cache key says so already.
        options.log_severity_level = 3
    except OSError as error:
        warnings.warn(
            f"ONNX Runtime cache disabled, {entry.parent} not writable: {error}", stacklevel=2
        )
    try:
        session = ort.InferenceSession(str(encoder_path), sess_options=options, providers=providers)
        if work is None:
            return session, None
        if entry.exists():
            _discard(entry)
        try:
            os.rename(work, entry)
        except OSError:
            if not cached.is_file():  # not just another process that got there first
                warnings.warn(f"ONNX Runtime cache not saved in {entry}", stacklevel=2)
                return session, None
        else:
            work = None
        return session, cached
    finally:
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)


class OnnxEncoder:
    """:class:`~gigaam_v3_streaming.session.Backend` running the encoder in ONNX Runtime on CPU.

    ``model`` supplies the stock frontend (``model.preprocessor``) and RNN-T
    head (``model.head``); its own encoder is not used. ``threads`` is ONNX
    Runtime's intra-op thread count. ``cache_dir`` keeps the optimized graph
    between processes (see :func:`open_session`).
    """

    def __init__(
        self,
        model: Any,
        encoder_path: os.PathLike | str,
        threads: int = 3,
        cache_dir: Optional[os.PathLike | str] = None,
    ) -> None:
        self.model = model
        self.encoder_path = Path(encoder_path)
        self.session, self.cached_graph = open_session(encoder_path, threads, cache_dir)
        self._inputs = [node.name for node in self.session.get_inputs()]
        self._output = self.session.get_outputs()[0].name

    def features(self, window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch

        wav = torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32)).unsqueeze(0)
        length = torch.tensor([window.size], dtype=torch.long)
        with torch.inference_mode():
            features, lengths = self.model.preprocessor(wav, length)
        return features.numpy(), lengths.numpy().astype(np.int64)

    def encode(self, window: np.ndarray) -> Any:
        import torch

        features, lengths = self.features(window)
        (encoded,) = self.session.run(
            [self._output], dict(zip(self._inputs, (features, lengths), strict=True))
        )
        # GigaAM's encoder returns channels first: [1, D, T].
        return torch.from_numpy(encoded)

    def new_decoder(self, max_symbols_per_step: int) -> FrameDecoder:
        return RNNTGreedyState(self.model.head, max_symbols_per_step)
