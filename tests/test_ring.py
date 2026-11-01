import numpy as np
import pytest

from gigaam_v3_streaming.ring import PCMRing


def test_ring_wrap_copy_and_prefix_discard() -> None:
    ring = PCMRing(5)
    ring.append(np.array([0, 1, 2], dtype=np.float32), 0)
    ring.discard_before(2)
    ring.append(np.array([3, 4, 5], dtype=np.float32), 3)
    actual = np.empty(4, dtype=np.float32)
    ring.copy_into(2, 6, actual)
    assert actual.tolist() == [2, 3, 4, 5]
    assert (ring.start, ring.end, ring.size, ring.free) == (2, 6, 4, 1)


def test_ring_rejects_noncontiguous_append_and_overread() -> None:
    ring = PCMRing(4)
    ring.append(np.array([1], dtype=np.float32), 0)
    with pytest.raises(ValueError, match="non-contiguous"):
        ring.append(np.array([2], dtype=np.float32), 2)
    with pytest.raises(ValueError, match="outside retained"):
        ring.copy_into(0, 2, np.empty(2, dtype=np.float32))
