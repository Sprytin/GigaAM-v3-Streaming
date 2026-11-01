"""A fixed-size, absolute-index PCM ring buffer."""

from __future__ import annotations

import numpy as np


class PCMRing:
    """Preallocated float32 ring with a contiguous absolute sample interval.

    The caller controls the absolute interval and may discard only its prefix.
    This keeps input buffering bounded without silently reallocating as a stream
    gets longer.
    """

    def __init__(self, capacity_samples: int) -> None:
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be positive")
        self._data = np.empty(capacity_samples, dtype=np.float32)
        self._capacity = capacity_samples
        self._start = 0
        self._size = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def start(self) -> int:
        return self._start

    @property
    def end(self) -> int:
        return self._start + self._size

    @property
    def size(self) -> int:
        return self._size

    @property
    def free(self) -> int:
        return self._capacity - self._size

    def append(self, samples: np.ndarray, offset: int) -> None:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("PCM ring accepts one-dimensional samples")
        if offset != self.end:
            raise ValueError(f"non-contiguous append: expected {self.end}, got {offset}")
        if samples.size > self.free:
            raise BufferError("PCM ring capacity exceeded")
        if not samples.size:
            return
        write = self.end % self._capacity
        first = min(samples.size, self._capacity - write)
        self._data[write : write + first] = samples[:first]
        if first < samples.size:
            self._data[: samples.size - first] = samples[first:]
        self._size += int(samples.size)

    def discard_before(self, offset: int) -> None:
        if offset <= self._start:
            return
        if offset >= self.end:
            self._start = self.end
            self._size = 0
            return
        discarded = offset - self._start
        self._start = offset
        self._size -= discarded

    def copy_into(self, start: int, end: int, destination: np.ndarray, dst_start: int = 0) -> None:
        """Copy ``[start, end)`` into a caller-owned array without allocation."""
        if start < self.start or end > self.end or end < start:
            raise ValueError(
                f"requested [{start}, {end}) outside retained [{self.start}, {self.end})"
            )
        n = end - start
        if destination.ndim != 1 or dst_start < 0 or dst_start + n > destination.size:
            raise ValueError("destination is too small")
        if not n:
            return
        read = start % self._capacity
        first = min(n, self._capacity - read)
        destination[dst_start : dst_start + first] = self._data[read : read + first]
        if first < n:
            destination[dst_start + first : dst_start + n] = self._data[: n - first]

    def clear(self) -> None:
        self._start = self.end
        self._size = 0
