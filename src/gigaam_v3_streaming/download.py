"""Pinned, atomic download of GigaAM checkpoints and tokenizers.

``gigaam`` downloads straight into the final file name and checks the
checkpoint's md5 only when it loads it; a download cut midway leaves a broken
file that every later load trips over, and the tokenizer is not checked at
all. :func:`fetch` downloads into a temporary file in the same directory,
checks a pinned sha256 and size, and only then renames it into place, so a
file under its final name is always complete and verified. ``gigaam`` then
finds the files present and loads them as usual.

The pins were taken from the files on GigaAM's CDN whose md5 matches the
hashes ``gigaam`` 0.2.0 itself pins (``_MODEL_HASHES``, and the tokenizer hash
in the checkpoint's own config).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

URL = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM"
DEFAULT_ROOT = Path("~/.cache/gigaam")  # gigaam's own default


@dataclass(frozen=True, slots=True)
class Pin:
    file: str
    size: int
    sha256: str


PINS: dict[str, tuple[Pin, ...]] = {
    "v3_e2e_rnnt": (
        Pin(
            "v3_e2e_rnnt.ckpt",
            448_929_252,
            "f60c62fe45902d967000770a12f260de7c5ac4d4b8e5e852f1df8c548e3958b5",
        ),
        Pin(
            "v3_e2e_rnnt_tokenizer.model",
            255_336,
            "828c12c991019eef952a960661f25a92d6ad279591e2ea466b4aeddf1d20a18a",
        ),
    ),
    "v3_rnnt": (
        Pin(
            "v3_rnnt.ckpt",
            446_390_032,
            "6b902433bfdc707c0995f3cb94f033bddc16bd5d0a1cf3bf0ff10d6b5e36ed12",
        ),
    ),
}


class ChecksumError(RuntimeError):
    """A downloaded or cached file does not match its pin."""


def file_sha256(path: Path) -> str:
    """sha256 of a file, remembered beside it while its size and mtime stay the same.

    The note is ``<file>.sha256``; it is only a shortcut, never trusted for a
    file whose size or mtime changed. A read-only directory just means the hash
    is computed every time.
    """
    path = Path(path)
    stat = path.stat()
    stamp = f"{stat.st_size} {stat.st_mtime_ns}"
    note = path.with_name(path.name + ".sha256")
    try:
        digest, noted = note.read_text(encoding="ascii").split(" ", 1)
        if noted.strip() == stamp and len(digest) == 64:
            return digest
    except (OSError, ValueError):
        pass
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    try:
        _write_atomic(note, f"{digest} {stamp}\n".encode("ascii"))
    except OSError:
        pass
    return digest


def fetch(
    name: str,
    root: Optional[os.PathLike | str] = None,
    *,
    url: str = URL,
    opener: Callable[[str], object] = urllib.request.urlopen,
) -> list[Path]:
    """Make the pinned files of model ``name`` present and verified in ``root``.

    Returns their paths. A file already present is checked against its pin and
    downloaded again if it does not match; a new download that does not match
    raises :class:`ChecksumError` and leaves nothing behind. ``KeyError`` for a
    model without pins.
    """
    directory = Path(root if root is not None else DEFAULT_ROOT).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for pin in PINS[name]:
        path = directory / pin.file
        if not (path.is_file() and _matches(path, pin)):
            _download(f"{url}/{pin.file}", path, pin, opener)
        paths.append(path)
    return paths


def _matches(path: Path, pin: Pin) -> bool:
    return path.stat().st_size == pin.size and file_sha256(path) == pin.sha256


def _download(url: str, path: Path, pin: Pin, opener: Callable[[str], object]) -> None:
    hasher = hashlib.sha256()
    size = 0
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(handle, "wb") as output, opener(url) as source:
            for block in iter(lambda: source.read(1 << 20), b""):
                hasher.update(block)
                size += len(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o644)  # mkstemp makes it private; a model cache is shared
        if size != pin.size or hasher.hexdigest() != pin.sha256:
            raise ChecksumError(
                f"{url}: expected {pin.size} bytes sha256 {pin.sha256}, "
                f"got {size} bytes sha256 {hasher.hexdigest()}"
            )
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)


def _write_atomic(path: Path, data: bytes) -> None:
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as output:
            output.write(data)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:  # not possible on every platform
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
