"""The modes' schedule presets and the hash-pinned, atomic checkpoint download."""

from __future__ import annotations

import hashlib
import io

import pytest

from gigaam_v3_streaming import MODES, RecognizerConfig, download, load_engine
from gigaam_v3_streaming.config import StreamingConfig


def test_mode_presets():
    assert MODES == ("gpu", "cpu", "cpu-preview")
    assert RecognizerConfig.for_mode("gpu") == RecognizerConfig()
    cpu = RecognizerConfig.for_mode("cpu")
    assert cpu.stable == StreamingConfig(emit_ms=960, left_context_ms=4_000, right_context_ms=800)
    assert cpu.preview is None and cpu.endpoint_ms == 600
    preview = RecognizerConfig.for_mode("cpu-preview")
    assert preview.stable == cpu.stable
    assert preview.preview == StreamingConfig(
        emit_ms=480, left_context_ms=2_400, right_context_ms=320
    )
    assert RecognizerConfig.for_mode("cpu", endpoint_ms=400).endpoint_ms == 400


def test_unknown_mode_names_the_modes():
    with pytest.raises(ValueError, match="gpu, cpu, cpu-preview"):
        RecognizerConfig.for_mode("tpu")
    with pytest.raises(ValueError, match="gpu, cpu, cpu-preview"):
        load_engine("fast")


# --- pinned downloads -------------------------------------------------------------------------

BLOB = bytes(range(256)) * 4_000  # ~1 MB, more than one read block


class Source(io.BytesIO):
    def __init__(self, data: bytes, fail_after: int | None = None) -> None:
        super().__init__(data)
        self.fail_after = fail_after

    def read(self, size: int = -1) -> bytes:
        if self.fail_after is not None and self.tell() >= self.fail_after:
            raise ConnectionResetError("cut")
        return super().read(size)


class Opener:
    def __init__(self, data: bytes, fail_after: int | None = None) -> None:
        self.data, self.fail_after, self.urls = data, fail_after, []

    def __call__(self, url: str) -> Source:
        self.urls.append(url)
        return Source(self.data, self.fail_after)


@pytest.fixture
def pinned(monkeypatch):
    pin = download.Pin("toy.ckpt", len(BLOB), hashlib.sha256(BLOB).hexdigest())
    monkeypatch.setitem(download.PINS, "toy", (pin,))
    return pin


def leftovers(directory):
    return sorted(path.name for path in directory.iterdir() if path.name.startswith("."))


def test_fetch_verifies_then_places_the_file(tmp_path, pinned):
    opener = Opener(BLOB)
    (path,) = download.fetch("toy", tmp_path, url="https://example.invalid", opener=opener)
    assert path == tmp_path / "toy.ckpt" and path.read_bytes() == BLOB
    assert opener.urls == ["https://example.invalid/toy.ckpt"]
    assert leftovers(tmp_path) == []
    # Present and verified: nothing is downloaded again.
    again = Opener(b"")
    download.fetch("toy", tmp_path, opener=again)
    assert again.urls == []


def test_fetch_never_places_a_wrong_or_cut_file(tmp_path, pinned):
    with pytest.raises(download.ChecksumError, match=pinned.sha256):
        download.fetch("toy", tmp_path, opener=Opener(BLOB[:-1] + b"x"))
    with pytest.raises(ConnectionResetError):
        download.fetch("toy", tmp_path, opener=Opener(BLOB, fail_after=len(BLOB) // 2))
    assert list(tmp_path.iterdir()) == []


def test_fetch_replaces_a_broken_file(tmp_path, pinned):
    (tmp_path / "toy.ckpt").write_bytes(BLOB[: len(BLOB) // 3])  # what a cut gigaam download leaves
    opener = Opener(BLOB)
    download.fetch("toy", tmp_path, opener=opener)
    assert len(opener.urls) == 1 and (tmp_path / "toy.ckpt").read_bytes() == BLOB


def test_hash_note_is_ignored_once_the_file_changes(tmp_path):
    path = tmp_path / "file"
    path.write_bytes(b"one")
    assert download.file_sha256(path) == hashlib.sha256(b"one").hexdigest()
    assert (tmp_path / "file.sha256").is_file()
    path.write_bytes(b"three")
    assert download.file_sha256(path) == hashlib.sha256(b"three").hexdigest()
