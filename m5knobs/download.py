"""Get the finished results/ without retraining: pack them into one tarball (maintainers) or fetch and unpack it (users)."""
import argparse
import gzip
import hashlib
import io
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request

from m5knobs import paths

TAG = "results-v1"
URL = f"https://github.com/jrodenberg0/m5-dl-knobs/releases/download/{TAG}/results.tar.gz"
SHA256 = "51ae99da7d0e67091d3b005582dc49ec1f3c76ccecc3efb15a7e47531ca17f0d"  # dist/results.tar.gz for release results-v1

# Files the notebooks rebuild from the runs; shipping them could leave stale verdicts next to fresh runs.
DERIVED_SKIP = {"card_verdicts.json", "stress_outcomes.json", "es_sweep.json", "f4_table.csv"}
_SKIP_SUFFIXES = (".log", ".png", ".svg")


def _members(results_dir: str) -> list:
    out = []
    for root, dirs, files in os.walk(results_dir):
        dirs.sort()
        for f in sorted(files):
            rel = os.path.relpath(os.path.join(root, f), results_dir)
            if f.endswith(_SKIP_SUFFIXES) or rel in DERIVED_SKIP or f.startswith("."):
                continue
            out.append(rel)
    return out


def pack_results(results_dir: str = paths.RESULTS, out: str = "results.tar.gz") -> str:
    """Write a byte-for-byte reproducible tarball of results_dir (fixed order, mtimes and owners). Returns its path."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as t:
        for rel in _members(results_dir):
            info = t.gettarinfo(os.path.join(results_dir, rel), arcname=rel)
            info.mtime, info.uid, info.gid, info.uname, info.gname, info.mode = 0, 0, 0, "", "", 0o644
            with open(os.path.join(results_dir, rel), "rb") as fh:
                t.addfile(info, fh)
    with open(out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
        gz.write(buf.getvalue())
    return out


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, out: str):
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as r, open(out, "wb") as fh:
        total, done = int(r.headers.get("Content-Length") or 0), 0
        for chunk in iter(lambda: r.read(1 << 20), b""):
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r{done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
    print()


def _safe_extract(tgz: str, dest: str):
    with tarfile.open(tgz, "r:gz") as t:
        members = t.getmembers()
        root = os.path.realpath(dest)
        for m in members:
            target = os.path.realpath(os.path.join(dest, m.name))
            if not (m.isfile() or m.isdir()) or os.path.isabs(m.name) or not target.startswith(root + os.sep):
                raise SystemExit(f"unsafe path in results archive: {m.name!r}; not extracting")
        t.extractall(dest, members=members)


def fetch_results(dest: str = paths.RESULTS, url: str = URL, sha256: str = SHA256, src: str = None, force: bool = False) -> str:
    """Download (or take src) the results tarball, check its SHA-256, and unpack it into dest. Returns dest."""
    if os.path.isdir(os.path.join(dest, "runs")) and not force:
        raise SystemExit(f"{dest} already holds runs; pass --force to replace them")
    with tempfile.TemporaryDirectory() as tmp:
        tgz = src or os.path.join(tmp, "results.tar.gz")
        if not src:
            _download(url, tgz)
        got = _sha256(tgz)
        if sha256 and got != sha256:
            raise SystemExit(f"checksum mismatch for {tgz}: expected {sha256}, got {got}")
        staging = os.path.join(tmp, "unpacked")
        os.makedirs(staging)
        _safe_extract(tgz, staging)
        if force and os.path.isdir(dest):
            shutil.rmtree(dest)
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        shutil.copytree(staging, dest, dirs_exist_ok=True)
    print(f"results ready in {dest}")
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(description="Get the finished results/ for the notebooks (no training).")
    ap.add_argument("--dest", default=paths.RESULTS)
    ap.add_argument("--from", dest="src", default=None, help="use a local results.tar.gz instead of downloading")
    ap.add_argument("--force", action="store_true", help="replace an existing results/")
    ap.add_argument("--pack", metavar="OUT", default=None, help="maintainers: write the release tarball from --dest and print its SHA-256")
    a = ap.parse_args(argv)
    if a.pack:
        out = pack_results(a.dest, a.pack)
        print(f"{out}  sha256={_sha256(out)}  ({os.path.getsize(out) / 1e6:,.0f} MB)")
        return
    fetch_results(dest=a.dest, src=a.src, force=a.force)


if __name__ == "__main__":
    main(sys.argv[1:])
