"""A loaded model in one of the supported modes, shared by any number of streams."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from .recognizer import RecognizerConfig, StreamingRecognizer, load_model
from .session import Backend


@dataclass(frozen=True)
class Engine:
    """What :func:`load_engine` returns: the model, how it runs, the mode's schedule.

    Recognizers made from one engine share its model and never change it;
    call into them from one thread at a time.
    """

    mode: str
    model: Any
    backend: Optional[Backend]
    config: RecognizerConfig

    def recognizer(self, config: Optional[RecognizerConfig] = None) -> StreamingRecognizer:
        """A new stream; ``config`` defaults to the mode's schedule."""
        return StreamingRecognizer(self.model, config or self.config, self.backend)


def load_engine(
    mode: str = "auto",
    name: str = "v3_e2e_rnnt",
    *,
    device: Optional[str] = None,
    threads: Optional[int] = None,
    cache_dir: Optional[os.PathLike | str] = None,
    download_root: Optional[os.PathLike | str] = None,
    quantization: Optional[str] = None,
    torch_threads: Optional[int] = 1,
    **load_kwargs: Any,
) -> Engine:
    """Load ``name`` in one of :data:`~gigaam_v3_streaming.MODES` (or ``"auto"``).

    * ``gpu``: stock PyTorch on CUDA, encoder in fp16 (``device`` picks the
      card; ``load_kwargs`` go to ``gigaam.load_model``).
    * ``cpu`` / ``cpu-preview``: the encoder in ONNX Runtime, INT8, on
      ``threads`` CPU threads (default ``min(3, cores)``); built from the
      checkpoint on first use and cached in ``cache_dir`` (see
      :func:`gigaam_v3_streaming.cpu.prepare`). ``torch_threads`` sets
      PyTorch's process-wide thread count for the small frontend and RNN-T
      head (``None`` leaves it alone); the published numbers use 1.
    * ``auto``: ``gpu`` when CUDA is available, else ``cpu``.
    """
    if mode == "auto":
        import torch

        mode = "gpu" if torch.cuda.is_available() else "cpu"
    config = RecognizerConfig.for_mode(mode)  # also rejects unknown modes
    if mode == "gpu":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "mode 'gpu' needs CUDA; on this machine use mode='cpu' or 'cpu-preview'"
            )
        model = load_model(name, device or "cuda", download_root=download_root, **load_kwargs)
        return Engine(mode, model, None, config)
    if load_kwargs:
        raise TypeError(f"unexpected arguments for mode {mode!r}: {', '.join(load_kwargs)}")
    from .cpu import load_cpu

    model, backend = load_cpu(
        name,
        threads=threads,
        cache_dir=cache_dir,
        download_root=download_root,
        quantization=quantization,
        torch_threads=torch_threads,
    )
    return Engine(mode, model, backend, config)
