"""Read a public research ZIP by HTTP Range, without downloading its full payload."""

import argparse
import io
import json
import struct
import zipfile
import zlib
from collections import Counter
from pathlib import Path

import requests

REPOSITORY = "yeliudev/UMT"
ARCHIVE_PATH = "datasets/qvhighlights-a8559488.zip"
ARCHIVE_SIZE = 10032460766
ARCHIVE_SHA256 = "a8559488ddd8356562ca94005d0f519cfe3474dab53779818ee65e46e52d2545"
PUBLIC_URL = f"https://hf-mirror.com/{REPOSITORY}/resolve/main/{ARCHIVE_PATH}"
AUTHOR_SOURCE = "https://github.com/TencentARC/UMT"
ROOT = Path(__file__).resolve().parent


class PublicRangeReader(io.RawIOBase):
    def __init__(self, url=PUBLIC_URL, size=ARCHIVE_SIZE, byte_budget=1024**3):
        super().__init__()
        self.url, self.size, self.byte_budget = url, size, byte_budget
        self.position, self.transferred, self.calls = 0, 0, 0
        self.session = requests.Session()

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        position = offset if whence == io.SEEK_SET else (self.position + offset if whence == io.SEEK_CUR else self.size + offset)
        if position < 0:
            raise ValueError("Negative ZIP offset")
        self.position = position
        return position

    def read(self, size=-1):
        size = min(self.size - self.position, self.size if size < 0 else size)
        if size <= 0:
            return b""
        if self.transferred + size > self.byte_budget:
            raise ValueError("Public archive transfer budget reached")
        end = self.position + size - 1
        with self.session.get(self.url, headers={"Range": f"bytes={self.position}-{end}", "Accept-Encoding": "identity"},
                              stream=True, timeout=(10, 40)) as response:
            response.raise_for_status()
            expected = f"bytes {self.position}-{end}/{self.size}"
            if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError("Server did not honor exact Range; refusing a full archive transfer")
            payload = response.raw.read(size + 1)
        if len(payload) != size:
            raise ValueError("Unexpected public archive byte count")
        self.position += size
        self.transferred += size
        self.calls += 1
        return payload

    def close(self):
        self.session.close()
        super().close()


def fetch_member(entry, session=None):
    """One bounded request per member; validate local ZIP header, size and CRC."""
    session = session or requests.Session()
    start = entry["offset"]
    # Small slack covers local extra fields; metadata is checked before decode.
    length = 30 + len(entry["name"].encode("utf-8")) + entry["compressed_size"] + 8192
    if length > 16 * 1024**2:
        raise ValueError("Member exceeds per-request bound")
    end = min(ARCHIVE_SIZE - 1, start + length - 1)
    with session.get(PUBLIC_URL, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                     stream=True, timeout=(10, 45)) as response:
        response.raise_for_status()
        if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{ARCHIVE_SIZE}":
            raise ValueError("Exact public Range was not honored")
        data = response.raw.read(end - start + 2)
    if len(data) != end - start + 1:
        raise ValueError("Incomplete public Range")
    return decode_member(entry, data), len(data)


def decode_member(entry, data):
    header = struct.unpack("<4s5H3I2H", data[:30])
    if header[0] != b"PK\x03\x04" or header[2] & 1:
        raise ValueError("Invalid/encrypted ZIP member")
    method, name_length, extra_length = header[3], header[-2], header[-1]
    name = data[30:30 + name_length].decode("utf-8" if header[2] & 0x800 else "cp437")
    payload_start = 30 + name_length + extra_length
    if name != entry["name"] or payload_start + entry["compressed_size"] > len(data):
        raise ValueError("ZIP local header does not match central index")
    compressed = data[payload_start:payload_start + entry["compressed_size"]]
    if method == zipfile.ZIP_DEFLATED:
        payload = zlib.decompress(compressed, -15)
    elif method == zipfile.ZIP_STORED:
        payload = compressed
    else:
        raise ValueError("Unsupported compression")
    if len(payload) != entry["size"] or zlib.crc32(payload) & 0xffffffff != entry["crc"]:
        raise ValueError("Public ZIP member size/CRC mismatch")
    return payload


def fetch_members(entries, session=None, *, url=PUBLIC_URL, archive_size=ARCHIVE_SIZE, max_range_bytes=8 * 1024**2):
    """Coalesce nearby whitelisted members; preserve per-member CRC checks."""
    session = session or requests.Session()
    start = min(entry["offset"] for entry in entries)
    end = max(entry["offset"] + 30 + len(entry["name"].encode()) + entry["compressed_size"] + 8192 for entry in entries) - 1
    end = min(archive_size - 1, end)
    if end - start + 1 > max_range_bytes:
        raise ValueError("Coalesced range exceeds bound")
    with session.get(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                     stream=True, timeout=(10, 45)) as response:
        response.raise_for_status()
        if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{archive_size}":
            raise ValueError("Coalesced exact Range was not honored")
        data = response.raw.read(end - start + 2)
    if len(data) != end - start + 1:
        raise ValueError("Incomplete coalesced range")
    return {entry["name"]: decode_member(entry, data[entry["offset"] - start:]) for entry in entries}, len(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--save-index", action="store_true")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    index_path = ROOT / "data" / "qv_public_index.json"
    if index_path.exists():
        entries = json.loads(index_path.read_text(encoding="utf-8"))["entries"]
        print(f"Using existing validated central-index snapshot: {len(entries)} members")
    else:
        entries = None
    if entries is None:
      with PublicRangeReader(byte_budget=64 * 1024**2) as reader, zipfile.ZipFile(reader) as archive:
        files = archive.infolist()
        counts = Counter("/".join(info.filename.split("/")[:2]) for info in files)
        print(json.dumps({"archive_size": ARCHIVE_SIZE, "author_source": AUTHOR_SOURCE,
                          "index_bytes_read": reader.transferred, "entries": len(files),
                          "groups": dict(counts), "first_entries": [info.filename for info in files[:30]]}, indent=2))
        entries = [{"name": info.filename, "offset": info.header_offset, "compressed_size": info.compress_size,
                    "size": info.file_size, "crc": info.CRC}
                   for info in files if not info.is_dir() and not info.filename.startswith("qvhighlights/clip_sub_features/")]
        if args.save_index:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            with index_path.open("x", encoding="utf-8") as stream:
                json.dump({"source": AUTHOR_SOURCE, "public_transfer_url": PUBLIC_URL,
                           "full_archive_sha256_published": ARCHIVE_SHA256,
                           "full_archive_hash_verified": False,
                           "validation": "ZIP central index and each selected member's ZIP CRC; not full archive SHA256",
                           "entries": entries}, stream)
    if args.sample:
        import numpy as np
        by_name = {entry["name"]: entry for entry in entries}
        names = ["qvhighlights/clip_features/j7rJstUseKg_360.0_510.0.npz",
                 "qvhighlights/pann_features/j7rJstUseKg_360.0_510.0.npy",
                 "qvhighlights/clip_text_features/qid9769.npz"]
        for name in names:
            if name not in by_name:
                print(f"Missing sample member: {name}")
                continue
            payload, transferred = fetch_member(by_name[name])
            data = np.load(io.BytesIO(payload), allow_pickle=False)
            if isinstance(data, np.ndarray):
                arrays = {"array": {"shape": list(data.shape), "dtype": str(data.dtype)}}
            else:
                with data:
                    arrays = {key: {"shape": list(data[key].shape), "dtype": str(data[key].dtype)} for key in data.files}
            print(json.dumps({"name": name, "transferred": transferred, "arrays": arrays}, indent=2))


if __name__ == "__main__":
    main()
