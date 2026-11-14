"""Cheap local file-version snapshots; these are not cryptographic content hashes."""

import hashlib
from pathlib import Path


def file_signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def media_id(path):
    signature = file_signature(path)
    value = f"{signature['path']}|{signature['size']}|{signature['mtime_ns']}"
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def file_version_status(path, expected):
    try:
        actual = file_signature(path)
    except OSError:
        return "missing"
    if expected is None:
        return "legacy"
    if hasattr(expected, "model_dump"):
        expected = expected.model_dump()
    return "current" if actual == expected else "changed"
