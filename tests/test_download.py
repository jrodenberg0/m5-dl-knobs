import hashlib
import io
import os
import tarfile

import pytest

from tests.fixtures.make_fake_results import make


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_pack_then_fetch_roundtrip(tmp_path):
    from m5knobs import download
    src = str(tmp_path / "res")
    make(src)
    tgz = download.pack_results(src, str(tmp_path / "results.tar.gz"))
    dest = str(tmp_path / "new_results")
    download.fetch_results(dest=dest, src=tgz, sha256=_sha(tgz))
    for root, _, files in os.walk(src):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), src)
            if f.endswith((".log", ".png", ".svg")) or rel in download.DERIVED_SKIP:
                continue
            assert os.path.exists(os.path.join(dest, rel)), rel


def test_pack_is_deterministic(tmp_path):
    from m5knobs import download
    src = str(tmp_path / "res")
    make(src)
    a = download.pack_results(src, str(tmp_path / "a.tar.gz"))
    b = download.pack_results(src, str(tmp_path / "b.tar.gz"))
    assert _sha(a) == _sha(b)


def test_fetch_rejects_bad_checksum(tmp_path):
    from m5knobs import download
    src = str(tmp_path / "res")
    make(src)
    tgz = download.pack_results(src, str(tmp_path / "results.tar.gz"))
    with pytest.raises(SystemExit, match="checksum"):
        download.fetch_results(dest=str(tmp_path / "out"), src=tgz, sha256="0" * 64)
    assert not os.path.exists(tmp_path / "out")


def test_fetch_refuses_to_overwrite_existing_results(tmp_path):
    from m5knobs import download
    src = str(tmp_path / "res")
    make(src)
    tgz = download.pack_results(src, str(tmp_path / "results.tar.gz"))
    dest = tmp_path / "out"
    (dest / "runs").mkdir(parents=True)
    with pytest.raises(SystemExit, match="already"):
        download.fetch_results(dest=str(dest), src=tgz, sha256=_sha(tgz))
    download.fetch_results(dest=str(dest), src=tgz, sha256=_sha(tgz), force=True)
    assert (dest / "actuals.parquet").exists() and any((dest / "runs").iterdir())


def test_fetch_rejects_unsafe_paths(tmp_path):
    from m5knobs import download
    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, "w:gz") as t:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 2
        t.addfile(info, io.BytesIO(b"hi"))
    with pytest.raises(SystemExit, match="unsafe"):
        download.fetch_results(dest=str(tmp_path / "out"), src=str(bad), sha256=_sha(bad))
    assert not (tmp_path / "escape.txt").exists()


def test_check_available_points_to_download_script(tmp_path):
    from m5knobs import results
    with pytest.raises(SystemExit, match="scripts/download_results.py"):
        results.check_available(str(tmp_path / "nothing"))
    src = str(tmp_path / "res")
    make(src)
    results.check_available(src)
