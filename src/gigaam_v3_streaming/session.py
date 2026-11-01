"""Fragmentation-invariant, bounded fixed-window streaming session."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional, Protocol

import numpy as np

from .config import FrontendAlignment, StreamingConfig
from .ring import PCMRing
from .rnnt import RNNTGreedyState


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """An append-only recognition update for one logical encoder interval.

    ``token_samples`` holds, for each token, the absolute sample index at which
    its emitting encoder frame starts (RNN-T emission time, 40 ms grid for the
    stock GigaAM frontend).
    """

    start_sample: int
    end_sample: int
    token_ids: tuple[int, ...] = ()
    is_final: bool = False
    token_samples: tuple[int, ...] = ()


@dataclass(slots=True)
class StreamStats:
    input_samples: int = 0
    logical_windows: int = 0
    emitted_frames: int = 0
    emitted_tokens: int = 0
    events: int = 0
    push_calls: int = 0
    peak_ring_samples: int = 0
    ring_capacity_samples: int = 0
    finished: bool = False


class FrameDecoder(Protocol):
    """Greedy decoder state carried from one window to the next."""

    def decode(self, encoded: Any) -> list[tuple[int, int]]:
        """Return ``(token_id, frame_index)`` pairs for ``[1, D, T]`` features."""


class Backend(Protocol):
    """How a session runs a model: one encoder call per window, one decoder per stream."""

    def encode(self, window: np.ndarray) -> Any:
        """Encode mono float32 PCM to ``[1, D, T]`` features, one frame per encoder hop."""

    def new_decoder(self, max_symbols_per_step: int) -> FrameDecoder:
        """Fresh decoder state for one stream."""


class StockGigaAM:
    """:class:`Backend` for a stock GigaAM RNN-T checkpoint.

    The encoder runs through ``model.forward(wav, length)`` on the model's
    device and dtype; the predictor and joint of ``model.head`` decode greedily
    with state carried across windows (:class:`RNNTGreedyState`).
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def encode(self, window: np.ndarray) -> Any:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - production dependency guard
            raise RuntimeError("stock GigaAM streaming requires the torch extra") from exc

        device = getattr(self.model, "_device", None)
        dtype = getattr(self.model, "_dtype", torch.float32)
        wav = torch.from_numpy(window).unsqueeze(0).to(device=device, dtype=dtype)
        length = torch.tensor([window.size], dtype=torch.long, device=device)
        with torch.inference_mode():
            result = self.model.forward(wav, length)
        return result[0] if isinstance(result, tuple) else result

    def new_decoder(self, max_symbols_per_step: int) -> FrameDecoder:
        return RNNTGreedyState(self.model.head, max_symbols_per_step)


class StreamingSession:
    """Run stock GigaAM E2E RNN-T on deterministic overlapping windows.

    The encoder is re-run on a bounded ``left | emit | right`` PCM window.  A
    separate absolute sample clock determines the windows, rather than packet
    boundaries passed to :meth:`push`; therefore identical PCM produces the
    same calls to the model for every fragmentation plan.

    ``model`` supplies the frontend geometry checked by
    :meth:`StreamingConfig.validate`; ``backend`` runs it and defaults to
    :class:`StockGigaAM`.
    """

    def __init__(
        self, model: Any, config: StreamingConfig, backend: Optional[Backend] = None
    ) -> None:
        self.model = model
        self.config = config
        self.alignment: FrontendAlignment = config.validate(model)
        self._backend = backend if backend is not None else StockGigaAM(model)
        self._decoder = self._backend.new_decoder(config.max_symbols_per_step)
        self._ring = PCMRing(config.ring_capacity_samples)
        self._window = np.zeros(config.window_samples, dtype=np.float32)
        self._next_emit_start = 0
        self._stream_end = 0
        self._finished = False
        self._stats = StreamStats(ring_capacity_samples=self._ring.capacity)

    @property
    def stats(self) -> StreamStats:
        """A snapshot; callers cannot mutate the session's counters."""
        return replace(self._stats)

    @property
    def processed_samples(self) -> int:
        """Samples whose decoding is settled: the start of the next window."""
        return min(self._next_emit_start, self._stream_end)

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def ring_occupancy_samples(self) -> int:
        """Exposed for operational monitoring and deterministic tests."""
        return self._ring.size

    def push(
        self,
        samples: Any,
        sample_offset: Optional[int] = None,
    ) -> list[StreamEvent]:
        """Append contiguous PCM and return all newly available updates.

        ``samples`` may be normalized numeric mono PCM or little-endian signed
        16-bit PCM bytes.  ``sample_offset`` is an absolute sample index; gaps
        and overlaps are rejected rather than silently changing the timeline.
        """
        if self._finished:
            raise RuntimeError("cannot push after finish()")
        pcm = coerce_pcm(samples)
        expected = self._stream_end
        offset = expected if sample_offset is None else int(sample_offset)
        if offset != expected:
            relation = "gap" if offset > expected else "overlap"
            raise ValueError(f"{relation}: expected sample_offset {expected}, got {offset}")

        self._stats.push_calls += 1
        self._stats.input_samples += int(pcm.size)
        return self._append(pcm)

    def _append(self, pcm: np.ndarray) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        source_index = 0
        while source_index < pcm.size:
            events.extend(self._drain_ready())
            if not self._ring.free:
                # With capacity exactly one logical window, a full ring must
                # always contain a ready block.  Anything else is an invariant
                # violation rather than a reason to grow memory.
                raise RuntimeError("full PCM ring without a processable window")
            take = min(self._ring.free, int(pcm.size - source_index))
            piece = pcm[source_index : source_index + take]
            self._ring.append(piece, self._stream_end)
            source_index += take
            self._stream_end += take
            self._stats.peak_ring_samples = max(self._stats.peak_ring_samples, self._ring.size)
            events.extend(self._drain_ready())
        return events

    def finish(self) -> list[StreamEvent]:
        """Decode the tail with full lookahead, emit a final event, release all PCM.

        Silence is padded to the next emit boundary plus the right context, so
        the last words are decoded like every other word. The padding belongs
        to the decoder only: tokens and times past the stream end are dropped.
        """
        if self._finished:
            return []
        end = self._stream_end
        events: list[StreamEvent] = []
        if end:
            emit = self.config.emit_samples
            padded = -(-end // emit) * emit + self.config.right_context_samples
            produced = self._append(np.zeros(padded - end, dtype=np.float32))
            for event in produced:
                kept = [
                    (token, sample)
                    for token, sample in zip(event.token_ids, event.token_samples, strict=True)
                    if sample < end
                ]
                if kept:
                    events.append(
                        replace(
                            event,
                            end_sample=min(event.end_sample, end),
                            token_ids=tuple(token for token, _ in kept),
                            token_samples=tuple(sample for _, sample in kept),
                        )
                    )
            self._stats.events -= len(produced) - len(events)
        self._stream_end = end
        self._ring.clear()
        self._finished = True
        self._stats.finished = True
        events.append(StreamEvent(start_sample=end, end_sample=end, is_final=True))
        self._stats.events += 1
        return events

    def cancel(self) -> None:
        """Release buffered state without decoding or emitting a final event."""
        if self._finished:
            return
        self._ring.clear()
        self._finished = True
        self._stats.finished = True

    def _drain_ready(self) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        required_end = (
            self._next_emit_start + self.config.emit_samples + self.config.right_context_samples
        )
        while self._ring.end >= required_end:
            events.extend(self._process_one())
            required_end = (
                self._next_emit_start + self.config.emit_samples + self.config.right_context_samples
            )
        return events

    def _process_one(self) -> list[StreamEvent]:
        start = self._next_emit_start
        logical_window_start = start - self.config.left_context_samples
        available_start = max(0, logical_window_start)
        available_end = min(
            start + self.config.emit_samples + self.config.right_context_samples,
            self._stream_end,
        )
        self._window.fill(0.0)
        if available_end > available_start:
            self._ring.copy_into(
                available_start,
                available_end,
                self._window,
                dst_start=available_start - logical_window_start,
            )

        # Do not make the model attend to seconds of synthetic silence at a
        # session boundary.  Interior updates remain fixed-size; the first and
        # final windows are trimmed to the real stream boundary, matching the
        # stock frontend's ``center=False`` utterance treatment more closely.
        active_start = available_start - logical_window_start
        active_end = active_start + (available_end - available_start)
        active_window = self._window[active_start:active_end]
        hop = self.alignment.encoder_hop_samples
        frame_count = self.config.emit_samples // hop
        encoded = self._backend.encode(active_window)
        central = self._central_frames(encoded, (start - available_start) // hop, frame_count)
        decoded = self._decoder.decode(central)
        token_ids = tuple(token for token, _ in decoded)
        token_samples = tuple(start + frame * hop for _, frame in decoded)

        actual_end = min(start + self.config.emit_samples, self._stream_end)
        self._next_emit_start += self.config.emit_samples
        self._ring.discard_before(max(0, self._next_emit_start - self.config.left_context_samples))
        self._stats.logical_windows += 1
        self._stats.emitted_frames += frame_count
        self._stats.emitted_tokens += len(token_ids)

        if not token_ids:
            return []
        event = StreamEvent(
            start_sample=start,
            end_sample=actual_end,
            token_ids=token_ids,
            token_samples=token_samples,
        )
        self._stats.events += 1
        return [event]

    def _central_frames(self, encoded: Any, central_frame_start: int, frame_count: int) -> Any:
        first = central_frame_start
        last = first + frame_count
        if encoded.ndim != 3 or encoded.shape[0] != 1 or encoded.shape[-1] < last:
            raise ValueError(f"encoder returned {tuple(encoded.shape)}; need [1, D, >={last}]")
        return encoded[:, :, first:last]


def coerce_pcm(samples: Any) -> np.ndarray:
    """Return mono float32 PCM from float samples, int16 samples or s16le bytes."""
    if isinstance(samples, (bytes, bytearray, memoryview)):
        if len(memoryview(samples).cast("B")) % 2:
            raise ValueError("s16le PCM byte length must be even")
        raw = np.frombuffer(samples, dtype="<i2")
        return raw.astype(np.float32) / 32768.0
    if hasattr(samples, "detach"):
        samples = samples.detach().cpu().numpy()
    pcm = np.asarray(samples)
    if pcm.ndim != 1:
        raise ValueError("samples must be mono and one-dimensional")
    if np.issubdtype(pcm.dtype, np.integer):
        if pcm.dtype == np.int16:
            return pcm.astype(np.float32) / 32768.0
        raise ValueError("integer samples must be int16 PCM; use float samples otherwise")
    return np.ascontiguousarray(pcm, dtype=np.float32)
