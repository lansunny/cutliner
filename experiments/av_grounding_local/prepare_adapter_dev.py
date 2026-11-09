"""Prepare a fresh adapter-dev set unseen by the existing QD-DETR's training."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from prepare_data import ANNOTATIONS, cached, download_group, feature_names, round_robin, source_group
from public_zip import ROOT


def main():
    base = ROOT / "data/av_v1"
    test = [json.loads(line) for line in (base / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    forbidden = {source_group(row) for row in test}
    rows = [json.loads(line) for line in (ANNOTATIONS / "highlight_val_release.jsonl").read_text(encoding="utf-8").splitlines()]
    index = json.loads((ROOT / "data/qv_public_index.json").read_text(encoding="utf-8"))["entries"]
    by_name = {entry["name"]: entry for entry in index}
    eligible = [row for row in rows if source_group(row) not in forbidden and all(name in by_name for name in feature_names(row))]
    selected = round_robin(eligible, 80, 20261108)
    folder = ROOT / "data/adapter_v1"
    folder.mkdir(exist_ok=True)
    plan = {"train_manifest": str(base / "train.jsonl"), "adapter_dev": selected,
            "heldout_manifest": str(base / "test.jsonl"), "heldout_not_repartitioned": True,
            "source_groups": len({source_group(row) for row in selected}),
            "scope": "Additional official-val sources, disjoint from frozen heldout; not used to train our adapters",
            "caveat": "Public pretrained checkpoint may have been selected using official validation; this is adapter-dev independence only"}
    plan_path = folder / "split_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Adapter split plan differs")
    else:
        with plan_path.open("x", encoding="utf-8") as stream:
            json.dump(plan, stream, indent=2)
    names = sorted({name for row in selected for name in feature_names(row)})
    pending = [by_name[name] for name in names if not cached(by_name[name])]
    print(f"Preparing {len(selected)} fresh source queries, {len(pending)} feature members; test stays sealed", flush=True)
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(download_group, [entry]) for entry in pending]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 40 == 0:
                print(f"Adapter dev features {i}/{len(pending)}", flush=True)
    for name, payload in (("train.jsonl", (base / "train.jsonl").read_text(encoding="utf-8")),
                          ("dev.jsonl", "\n".join(json.dumps(row) for row in selected) + "\n")):
        path = folder / name
        if path.exists() and path.read_text(encoding="utf-8") != payload:
            raise ValueError("Adapter manifest differs")
        if not path.exists():
            with path.open("x", encoding="utf-8") as stream:
                stream.write(payload)
    print(f"Fresh adapter-dev source plan sha256: {hashlib.sha256(plan_path.read_bytes()).hexdigest()}", flush=True)


if __name__ == "__main__":
    main()
