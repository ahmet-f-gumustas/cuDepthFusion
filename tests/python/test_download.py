from __future__ import annotations

import hashlib
import io
import tarfile
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from cudepthfusion.data import download as dl

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB
PAGE = "https://example.invalid/official"


class _Server(ThreadingHTTPServer):
    fail_head_next: int = 0  # respond 503 to HEAD this many times
    fail_next: int = 0  # respond 503 to GET this many times
    truncate_next: int = 0  # send half the body this many times
    ignore_range: bool = False
    missing: bool = False


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, *args: object) -> None:  # keep test output quiet
        pass

    def do_HEAD(self) -> None:
        if self.server.missing:
            self.send_error(404)
            return
        if self.server.fail_head_next > 0:
            self.server.fail_head_next -= 1
            self.send_error(503)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(PAYLOAD)))
        self.end_headers()

    def do_GET(self) -> None:
        if self.server.missing:
            self.send_error(404)
            return
        if self.server.fail_next > 0:
            self.server.fail_next -= 1
            self.send_error(503)
            return
        start = 0
        range_header = self.headers.get("Range")
        if range_header and not self.server.ignore_range:
            start = int(range_header.removeprefix("bytes=").split("-")[0])
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
        else:
            self.send_response(200)
        body = PAYLOAD[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.server.truncate_next > 0:
            self.server.truncate_next -= 1
            body = body[: len(body) // 2]
        self.wfile.write(body)


@pytest.fixture
def server() -> Iterator[_Server]:
    httpd = _Server(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def _url(server: _Server) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}/file.bin"


def _download(server: _Server, dest: Path, **kwargs: object) -> dl.DownloadResult:
    return dl.download(
        _url(server), dest, official_page=PAGE, log=lambda _: None, sleep=lambda _: None, **kwargs
    )


def test_downloads_and_hashes(server: _Server, tmp_path: Path) -> None:
    result = _download(server, tmp_path / "file.bin")
    assert result.path.read_bytes() == PAYLOAD
    assert result.sha256 == hashlib.sha256(PAYLOAD).hexdigest()
    assert not (tmp_path / "file.bin.part").exists()
    assert result.reused_existing is False


def test_reuses_complete_file(server: _Server, tmp_path: Path) -> None:
    (tmp_path / "file.bin").write_bytes(PAYLOAD)
    assert _download(server, tmp_path / "file.bin").reused_existing is True


def test_resumes_partial_download(server: _Server, tmp_path: Path) -> None:
    (tmp_path / "file.bin.part").write_bytes(PAYLOAD[:1000])
    result = _download(server, tmp_path / "file.bin")
    assert result.path.read_bytes() == PAYLOAD


def test_resumed_download_reports_progress_once_per_step(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dl, "CHUNK_BYTES", 1024)  # many chunks after the resume point
    (tmp_path / "file.bin.part").write_bytes(PAYLOAD[: len(PAYLOAD) // 2])
    messages: list[str] = []
    dl.download(
        _url(server),
        tmp_path / "file.bin",
        official_page=PAGE,
        log=messages.append,
        sleep=lambda _: None,
    )
    progress = [m for m in messages if "%" in m]
    # Resumed at 50 %: one line per 5 % step up to 100 %, not one per chunk.
    assert 1 <= len(progress) <= round(0.5 / dl.PROGRESS_STEP) + 1


def test_restarts_when_server_ignores_range(server: _Server, tmp_path: Path) -> None:
    server.ignore_range = True
    (tmp_path / "file.bin.part").write_bytes(b"garbage")
    assert _download(server, tmp_path / "file.bin").path.read_bytes() == PAYLOAD


def test_retries_transient_errors_and_truncation(server: _Server, tmp_path: Path) -> None:
    server.fail_next = 2
    server.truncate_next = 1
    result = _download(server, tmp_path / "file.bin", retries=5)
    assert result.path.read_bytes() == PAYLOAD


def test_retries_transient_failure_of_size_request(server: _Server, tmp_path: Path) -> None:
    server.fail_head_next = 2  # e.g. a temporary DNS or gateway failure before the transfer
    assert _download(server, tmp_path / "file.bin", retries=3).path.read_bytes() == PAYLOAD


def test_gives_up_after_retries(server: _Server, tmp_path: Path) -> None:
    server.fail_next = 10
    with pytest.raises(dl.DownloadError, match="after 2 retries.*official page"):
        _download(server, tmp_path / "file.bin", retries=2)


def test_http_404_fails_fast_and_names_official_page(server: _Server, tmp_path: Path) -> None:
    server.missing = True
    with pytest.raises(dl.DownloadError, match="official page https://example.invalid/official"):
        _download(server, tmp_path / "file.bin")


def test_concurrent_download_of_the_same_file_is_refused(server: _Server, tmp_path: Path) -> None:
    dest = tmp_path / "file.bin"
    with dl.exclusive_download(dest):  # another process holding the lock
        with pytest.raises(dl.DownloadError, match="already downloading"):
            _download(server, dest)
    assert _download(server, dest).path.read_bytes() == PAYLOAD  # free again afterwards


def test_corrupt_archive_raises_download_error(tmp_path: Path) -> None:
    archive = tmp_path / "broken.tar.gz"
    good = _make_tar(tmp_path / "good.tar.gz", [_file("seq/a.txt", b"a" * 50000)])
    data = good.read_bytes()
    archive.write_bytes(data[: len(data) // 2])  # truncated mid-stream
    with pytest.raises(dl.DownloadError, match="corrupt or truncated"):
        dl.safe_extract(archive, tmp_path / "out")


def test_refuses_when_disk_is_too_small(
    server: _Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda _: type("U", (), {"free": 1024})())
    with pytest.raises(dl.DownloadError, match="not enough disk space"):
        _download(server, tmp_path / "file.bin")


def _make_tar(path: Path, members: list[tuple[tarfile.TarInfo, bytes | None]]) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for info, data in members:
            tar.addfile(info, io.BytesIO(data) if data is not None else None)
    return path


def _file(name: str, data: bytes = b"x") -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    return info, data


def _link(name: str, target: str, kind: bytes = tarfile.SYMTYPE) -> tuple[tarfile.TarInfo, None]:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = target
    return info, None


def test_safe_extract_writes_regular_members(tmp_path: Path) -> None:
    archive = _make_tar(
        tmp_path / "ok.tar.gz",
        [_file("seq/depth/0.png", b"a"), _file("seq/depth.txt", b"b"), _link("seq/l", "depth.txt")],
    )
    top = dl.safe_extract(archive, tmp_path / "out")
    assert top == ["seq"]
    assert (tmp_path / "out/seq/depth/0.png").read_bytes() == b"a"


@pytest.mark.parametrize(
    "member",
    [
        _file("../evil.txt"),
        _file("/abs/evil.txt"),
        _file("seq/../../evil.txt"),
        _link("seq/link", "../../outside"),
        _link("seq/link", "/etc/passwd"),
        _link("seq/hard", "../outside", tarfile.LNKTYPE),
    ],
    ids=["dotdot", "absolute", "nested-dotdot", "symlink-out", "symlink-abs", "hardlink-out"],
)
def test_safe_extract_refuses_escaping_members(tmp_path: Path, member: tuple) -> None:
    archive = _make_tar(tmp_path / "bad.tar.gz", [_file("seq/ok.txt"), member])
    with pytest.raises(dl.UnsafeArchiveError):
        dl.safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "out/seq/ok.txt").exists()  # nothing written before the check
    assert not (tmp_path / "evil.txt").exists()


def test_safe_extract_refuses_fifo(tmp_path: Path) -> None:
    fifo = tarfile.TarInfo("seq/pipe")
    fifo.type = tarfile.FIFOTYPE
    archive = _make_tar(tmp_path / "fifo.tar.gz", [(fifo, None)])
    with pytest.raises(dl.UnsafeArchiveError, match="FIFO"):
        dl.safe_extract(archive, tmp_path / "out")
