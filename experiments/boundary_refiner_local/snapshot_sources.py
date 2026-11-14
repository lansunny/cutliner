"""Preserve exact executable sources beside experiment evidence before iteration."""

import argparse
import json
from pathlib import Path

from prepare_data import dump, file_hash, immutable_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    configs = list(args.round.glob("*/configuration.json"))
    for path in configs:
        config = json.loads(path.read_text(encoding="utf-8"))
        for name, expected in config.get("source_hashes", {}).items():
            if expected != file_hash(here / name):
                raise ValueError(f"Current {name} does not reconstruct {path}")
        for field, name in (("model_source_sha256", "model.py"), ("dataset_source_sha256", "dataset.py"),
                            ("train_source_sha256", "train.py")):
            if field in config and config[field] != file_hash(here / name):
                raise ValueError(f"Current {name} does not reconstruct {path}; do not create a misleading snapshot")
    entries = []
    for source in sorted(here.glob("*.py")):
        immutable_bytes(args.round / "source_snapshot" / source.name, source.read_bytes())
        entries.append({"file": source.name, "sha256": file_hash(source)})
    dump(args.round / "source_snapshot" / "sources.json", entries)
    print(f"Verified and snapshotted {len(entries)} executable source files")


if __name__ == "__main__":
    main()
