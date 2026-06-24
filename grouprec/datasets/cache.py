"""Download / cache management for the dataset registry.

Cache location resolution order: ``GROUPREC_CACHE`` env var, else
``platformdirs.user_cache_dir("grouprec")``. We never bundle data in the wheel;
auto-fetched datasets are downloaded here on first use.
"""

from __future__ import annotations

import hashlib
import os
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import platformdirs
from tqdm import tqdm


def cache_dir() -> Path:
    """Root cache directory (honors ``GROUPREC_CACHE``)."""
    env = os.environ.get("GROUPREC_CACHE")
    root = Path(env) if env else Path(platformdirs.user_cache_dir("grouprec"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def dataset_dir(name: str) -> Path:
    d = cache_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, *, checksum: str | None = None) -> Path:
    """Download ``url`` to ``dest`` (skips if present and checksum matches)."""
    if dest.exists() and (checksum is None or sha256(dest) == checksum):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp:  # noqa: S310 - canonical dataset URLs
        total = int(resp.headers.get("Content-Length", 0))
        with open(tmp, "wb") as out, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"download {dest.name}"
        ) as bar:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                bar.update(len(chunk))
    if checksum is not None and sha256(tmp) != checksum:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"checksum mismatch for {url}")
    tmp.rename(dest)
    return dest


def extract(archive: Path, dest: Path) -> Path:
    """Extract a .zip / .tar.* archive into ``dest`` (idempotent-ish)."""
    dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            t.extractall(dest)  # noqa: S202 - trusted dataset archives
    else:
        raise ValueError(f"unsupported archive format: {archive}")
    return dest


def find_file(root: Path, *names: str) -> Path | None:
    """First file under ``root`` matching any of ``names`` (recursive)."""
    for name in names:
        hits = sorted(root.rglob(name))
        if hits:
            return hits[0]
    return None
