"""CPU modes: GigaAM's encoder in ONNX Runtime, INT8, built once from the stock checkpoint.

Nothing but the stock checkpoint is downloaded. On first use :func:`prepare`
exports the encoder with GigaAM's own ``to_onnx``, quantizes its MatMul weights
to INT8 with ONNX Runtime's dynamic quantization and stores the result in the
cache, keyed by the checkpoint's hash and the quantization. The export needs
the whole PyTorch model, so it runs in a child process: the memory it takes
is given back when it exits. Afterwards a CPU start reads only the frontend,
the RNN-T head and the tokenizer from the checkpoint (memory-mapped) and the
INT8 encoder, with ONNX Runtime's optimized graph cached as well
(:func:`gigaam_v3_streaming.onnx_backend.open_session`).

Prepare ahead of time, for example while building an image::

    python -m gigaam_v3_streaming.cpu --model v3_e2e_rnnt --threads 3
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Optional

from .download import PINS, fetch, file_sha256

QUANTIZATIONS = ("int8", "int8-per-channel")
DEFAULT_QUANTIZATION = "int8"


def default_cache_dir() -> Path:
    """``$GIGAAM_V3_STREAMING_CACHE``, else ``gigaam-v3-streaming`` in the user cache directory."""
    explicit = os.environ.get("GIGAAM_V3_STREAMING_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser() / "gigaam-v3-streaming"


def default_threads() -> int:
    return max(1, min(3, os.cpu_count() or 1))


def checkpoint_files(
    name: str, download_root: Optional[os.PathLike | str]
) -> tuple[Path, Optional[Path]]:
    """The verified checkpoint and tokenizer (``None`` when the model has none) of ``name``."""
    if name not in PINS:
        raise ValueError(f"CPU modes support {', '.join(PINS)}; got {name!r}")
    paths = fetch(name, download_root)
    return paths[0], (paths[1] if len(paths) > 1 else None)


def encoder_path(name: str, checkpoint: Path, quantization: str, cache_dir: Path) -> Path:
    if quantization not in QUANTIZATIONS:
        raise ValueError(f"unknown quantization {quantization!r}; choose one of {QUANTIZATIONS}")
    return cache_dir / f"{name}-{file_sha256(checkpoint)[:16]}-encoder-{quantization}.onnx"


def prepare(
    name: str = "v3_e2e_rnnt",
    *,
    cache_dir: Optional[os.PathLike | str] = None,
    download_root: Optional[os.PathLike | str] = None,
    quantization: Optional[str] = None,
    in_process: bool = False,
) -> Path:
    """Make the INT8 encoder of ``name`` present in the cache; returns its path.

    Builds it if missing: export and quantization in a temporary directory
    beside the cache, then an atomic rename, so concurrent or interrupted
    builds never leave a partial file under the final name. The build runs in
    a child process unless ``in_process``.
    """
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    quantization = quantization or DEFAULT_QUANTIZATION
    checkpoint, _ = checkpoint_files(name, download_root)
    target = encoder_path(name, checkpoint, quantization, cache)
    if target.is_file():
        return target
    cache.mkdir(parents=True, exist_ok=True)
    if not in_process:
        command = [sys.executable, "-m", "gigaam_v3_streaming.cpu", "--model", name]
        command += ["--cache-dir", str(cache), "--quantization", quantization, "--no-warm"]
        if download_root is not None:
            command += ["--download-root", str(download_root)]
        subprocess.run(command, check=True)
        if not target.is_file():
            raise RuntimeError(f"building the CPU encoder did not produce {target}")
        return target
    _build(name, checkpoint.parent, quantization, target)
    return target


def _build(name: str, download_root: Path, quantization: str, target: Path) -> None:
    """Export, then quantize, each in its own process so their peaks never add up."""
    import multiprocessing

    from onnxruntime.quantization import QuantType, quantize_dynamic

    _drop_stale_builds(target.parent)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".build-") as work:
        export = multiprocessing.get_context("spawn").Process(
            target=_export, args=(name, str(download_root), work)
        )
        export.start()
        export.join()
        if export.exitcode != 0:
            raise RuntimeError(f"exporting {name} to ONNX failed (exit code {export.exitcode})")
        quantized = Path(work) / target.name
        quantize_dynamic(
            Path(work) / f"{name}_encoder.onnx",
            quantized,
            op_types_to_quantize=["MatMul"],
            per_channel=quantization == "int8-per-channel",
            weight_type=QuantType.QInt8,
        )
        os.replace(quantized, target)


def _export(name: str, download_root: str, work: str) -> None:
    import gigaam

    model = gigaam.load_model(name, fp16_encoder=False, device="cpu", download_root=download_root)
    model.to_onnx(work)


def _drop_stale_builds(cache: Path, older_than_s: float = 6 * 3600) -> None:
    """Remove build directories a killed build left behind (a live build is far younger)."""
    import time

    for path in cache.glob(".build-*"):
        try:
            if time.time() - path.stat().st_mtime > older_than_s:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


class CpuModel:
    """The parts of a stock GigaAM RNN-T model the CPU modes run in PyTorch.

    Frontend (``preprocessor``), RNN-T ``head`` and ``decoding`` (tokenizer),
    built from the checkpoint's own config and weights; the encoder is never
    built, its weights are never read (the checkpoint is memory-mapped).
    ``subsampling_factor`` gives the encoder's frame geometry.
    """

    def __init__(self, checkpoint: Path, tokenizer: Optional[Path]) -> None:
        import hydra
        import torch

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            loaded = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        cfg = loaded["cfg"]
        if tokenizer is not None:
            cfg.decoding.model_path = str(tokenizer)
        self.cfg = cfg
        self.preprocessor = hydra.utils.instantiate(cfg.preprocessor)
        self.head = hydra.utils.instantiate(cfg.head)
        self.decoding = hydra.utils.instantiate(cfg.decoding)
        self.subsampling_factor = int(cfg.encoder.subsampling_factor)
        for prefix, module in (("preprocessor.", self.preprocessor), ("head.", self.head)):
            state = {
                key[len(prefix) :]: value.clone()
                for key, value in loaded["state_dict"].items()
                if key.startswith(prefix)
            }
            module.load_state_dict(state, strict=True)
            module.eval()
        del loaded


def load_cpu(
    name: str = "v3_e2e_rnnt",
    *,
    threads: Optional[int] = None,
    cache_dir: Optional[os.PathLike | str] = None,
    download_root: Optional[os.PathLike | str] = None,
    quantization: Optional[str] = None,
    torch_threads: Optional[int] = 1,
) -> tuple[CpuModel, Any]:
    """The lean model and its :class:`~gigaam_v3_streaming.onnx_backend.OnnxEncoder` backend."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "CPU modes need ONNX Runtime: pip install 'gigaam-v3-streaming[cpu]'"
        ) from error
    from .onnx_backend import OnnxEncoder

    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    encoder = prepare(name, cache_dir=cache, download_root=download_root, quantization=quantization)
    checkpoint, tokenizer = checkpoint_files(name, download_root)
    if torch_threads is not None:
        import torch

        torch.set_num_threads(torch_threads)
    model = CpuModel(checkpoint, tokenizer)
    backend = OnnxEncoder(model, encoder, threads or default_threads(), cache)
    return model, backend


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the INT8 encoder for the CPU modes and warm ONNX Runtime's cache."
    )
    parser.add_argument("--model", default="v3_e2e_rnnt", choices=sorted(PINS))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--download-root", type=Path, default=None)
    parser.add_argument("--quantization", choices=QUANTIZATIONS, default=DEFAULT_QUANTIZATION)
    parser.add_argument("--threads", type=int, default=None, help="warm the cache for this many")
    parser.add_argument("--no-warm", action="store_true", help="only build the INT8 encoder")
    parser.add_argument("--rebuild", action="store_true", help="drop this model's cached files")
    args = parser.parse_args()

    cache = args.cache_dir or default_cache_dir()
    if args.rebuild and cache.is_dir():
        for path in cache.glob(f"{args.model}-*"):
            path.unlink() if path.is_file() else shutil.rmtree(path)
    encoder = prepare(
        args.model,
        cache_dir=cache,
        download_root=args.download_root,
        quantization=args.quantization,
        in_process=True,
    )
    print(encoder)
    if not args.no_warm:
        from .onnx_backend import open_session

        _, graph = open_session(encoder, args.threads or default_threads(), cache)
        print(graph)


if __name__ == "__main__":
    main()
