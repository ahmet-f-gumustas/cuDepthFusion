"""Resumable, retried downloads and archive extraction that refuses unsafe members.

Only the standard library is used so the downloader works before any optional
dependency is installed.
"""

from __future__ import annotations

import contextlib
import fcntl
import gzip
import hashlib
import http.client
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TypeVar

T = TypeVar("T")

CHUNK_BYTES = 1 << 20
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_RETRIES = 5
BACKOFF_BASE_S = 2.0
BACKOFF_MAX_S = 60.0
DISK_MARGIN_BYTES = 512 * 1024 * 1024
PROGRESS_STEP = 0.05
USER_AGENT = "cuDepthFusion-downloader"
RETRYABLE_HTTP_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})

Logger = Callable[[str], None]
Opener = Callable[..., object]


class DownloadError(RuntimeError):
    """A download or extraction failed; the message names the official source page."""


class UnsafeArchiveError(DownloadError):
    """An archive member would escape the destination or is not a regular file/dir/link."""


@dataclass(frozen=True)
class DownloadResult:
    url: str
    path: Path
    size_bytes: int
    sha256: str
    reused_existing: bool


def log_to_stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_bytes(count: float) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"


def check_disk_space(directory: Path, needed_bytes: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(directory).free
    if free < needed_bytes + DISK_MARGIN_BYTES:
        raise DownloadError(
            f"not enough disk space in {directory}: need {format_bytes(needed_bytes)} "
            f"plus a {format_bytes(DISK_MARGIN_BYTES)} margin, {format_bytes(free)} free"
        )


def remote_size(url: str, timeout_s: float, opener: Opener = urllib.request.urlopen) -> int | None:
    """Content-Length from a HEAD request, or None when the server does not report it."""
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with opener(request, timeout=timeout_s) as response:
        length = response.headers.get("Content-Length")
    return int(length) if length is not None else None


@contextlib.contextmanager
def exclusive_download(dest: Path) -> Iterator[None]:
    """Hold an exclusive lock for ``dest``: two writers on one ``.part`` corrupt it.

    The lock file is left in place on purpose; unlinking it would let a third process
    lock a fresh inode while a second one still holds the old one.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    lock_path = dest.with_name(dest.name + ".lock")
    with lock_path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise DownloadError(
                f"another process is already downloading {dest} (lock: {lock_path})"
            ) from error
        yield


def download(
    url: str,
    dest: Path,
    *,
    official_page: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
    reserve_for_extraction: bool = False,
    log: Logger = log_to_stderr,
    opener: Opener = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> DownloadResult:
    """Download ``url`` to ``dest``, resuming a previous ``dest.part`` when possible.

    ``reserve_for_extraction`` also requires free space for an extracted copy of about the
    archive's size (PNG content barely compresses). A second concurrent download of the
    same ``dest`` is refused instead of interleaving writes.
    """
    with exclusive_download(dest):
        return _download_locked(
            url,
            dest,
            official_page=official_page,
            timeout_s=timeout_s,
            retries=retries,
            reserve_for_extraction=reserve_for_extraction,
            log=log,
            opener=opener,
            sleep=sleep,
        )


def _download_locked(
    url: str,
    dest: Path,
    *,
    official_page: str,
    timeout_s: float,
    retries: int,
    reserve_for_extraction: bool,
    log: Logger,
    opener: Opener,
    sleep: Callable[[float], None],
) -> DownloadResult:
    def retrying(action: Callable[[], T]) -> T:
        return _with_retries(action, url, official_page, retries, log, sleep)

    total = retrying(lambda: remote_size(url, timeout_s, opener))

    if dest.exists() and total is not None and dest.stat().st_size == total:
        log(f"reusing {dest} ({format_bytes(total)})")
        return DownloadResult(url, dest, total, sha256_file(dest), reused_existing=True)

    part = dest.with_name(dest.name + ".part")
    already = part.stat().st_size if part.exists() else 0
    remaining = (total - already) if total is not None else 0
    extraction = total if (reserve_for_extraction and total is not None) else 0
    check_disk_space(dest.parent, max(remaining, 0) + extraction)
    log(f"downloading {url} -> {dest} (size: {format_bytes(total) if total else 'unknown'})")

    retrying(lambda: _transfer(url, part, total, timeout_s, opener, log))

    size = part.stat().st_size
    if total is not None and size != total:
        raise DownloadError(f"{url}: downloaded {size} bytes, server announced {total}")
    part.replace(dest)
    return DownloadResult(url, dest, size, sha256_file(dest), reused_existing=False)


def _with_retries(
    action: Callable[[], T],
    url: str,
    official_page: str,
    retries: int,
    log: Logger,
    sleep: Callable[[float], None],
) -> T:
    """Run ``action``; retry network errors and transient HTTP codes with backoff."""
    failures = 0
    while True:
        try:
            return action()
        except DownloadError:
            raise
        except (urllib.error.URLError, OSError, http.client.HTTPException) as error:
            code = getattr(error, "code", None)
            if code is not None and code not in RETRYABLE_HTTP_CODES:
                raise DownloadError(_failure(url, official_page, error)) from error
            failures += 1
            if failures > retries:
                raise DownloadError(_failure(url, official_page, error, retries)) from error
            delay = min(BACKOFF_BASE_S**failures, BACKOFF_MAX_S)
            log(f"attempt {failures}/{retries} failed ({error}); retrying in {delay:.0f} s")
            sleep(delay)


def _transfer(
    url: str, part: Path, total: int | None, timeout_s: float, opener: Opener, log: Logger
) -> None:
    offset = part.stat().st_size if part.exists() else 0
    if total is not None and offset > total:
        part.unlink()
        offset = 0
    if total is not None and offset == total:
        return

    headers = {"User-Agent": USER_AGENT}
    if offset > 0:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with opener(request, timeout=timeout_s) as response:
        if offset > 0 and response.status == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {offset}-"):
                raise DownloadError(f"{url}: unexpected Content-Range {content_range!r}")
            mode = "ab"
            log(f"resuming at {format_bytes(offset)}")
        else:
            offset = 0  # server ignored the range request: start over
            mode = "wb"
        written = offset
        # First report at the next whole step above a resumed offset, not at every chunk.
        done = written / total if total else 0.0
        next_report = (int(done / PROGRESS_STEP) + 1) * PROGRESS_STEP
        with part.open(mode) as stream:
            for chunk in iter(lambda: response.read(CHUNK_BYTES), b""):
                stream.write(chunk)
                written += len(chunk)
                if total and written / total >= next_report:
                    log(f"  {written / total:5.0%}  {format_bytes(written)}")
                    next_report += PROGRESS_STEP
    if total is not None and written < total:
        raise http.client.IncompleteRead(b"", total - written)


def _failure(url: str, official_page: str, error: BaseException, retries: int = 0) -> str:
    after = f" after {retries} retries" if retries else ""
    return (
        f"download of {url} failed{after}: {error}. Check the official page {official_page}; "
        "no mirror is tried automatically"
    )


def safe_extract(archive: Path, dest: Path) -> list[str]:
    """Extract a tar archive into ``dest``; returns the sorted top-level entry names.

    Every member is checked before anything is written: absolute paths, ``..``
    components, links resolving outside ``dest`` and device/FIFO entries are refused.
    """
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    try:
        with tarfile.open(archive, "r:*") as tar:
            members = tar.getmembers()
            for member in members:
                _check_member(member, root)
            if hasattr(tarfile, "data_filter"):
                tar.extractall(root, members=members, filter="data")
            else:  # pragma: no cover - Python without PEP 706 filters
                tar.extractall(root, members=members)
    except (tarfile.TarError, EOFError, zlib.error, gzip.BadGzipFile) as error:
        raise DownloadError(
            f"{archive} is corrupt or truncated ({error}); delete it and download it again"
        ) from error
    return sorted({PurePosixPath(m.name).parts[0] for m in members if m.name not in ("", ".")})


def _check_member(member: tarfile.TarInfo, root: Path) -> None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts:
        raise UnsafeArchiveError(f"archive member {member.name!r} escapes the destination")
    if not _inside(root / path, root):
        raise UnsafeArchiveError(f"archive member {member.name!r} escapes the destination")
    if member.ischr() or member.isblk() or member.isfifo():
        raise UnsafeArchiveError(f"archive member {member.name!r} is a device or FIFO")
    if member.issym() or member.islnk():
        link = PurePosixPath(member.linkname)
        if link.is_absolute():
            raise UnsafeArchiveError(f"link {member.name!r} -> {member.linkname!r} is absolute")
        base = path.parent if member.issym() else PurePosixPath()
        if not _inside(root / base / link, root):
            raise UnsafeArchiveError(
                f"link {member.name!r} -> {member.linkname!r} points outside the destination"
            )


def _inside(candidate: Path, root: Path) -> bool:
    resolved = candidate.resolve()
    return resolved == root or root in resolved.parents
