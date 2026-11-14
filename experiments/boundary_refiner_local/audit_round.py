"""Summarize terminal synthetic/real-candidate dev evidence and delivery guardrails."""

import argparse
import json
from pathlib import Path

from compare import paired_difference
from prepare_data import dump, file_hash, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads((args.round / "sweep_status.json").read_text(encoding="utf-8"))
    if state["status"] != "complete":
        raise ValueError("Audit requires an actually completed architecture round")
    controlled = json.loads((args.round / "comparison/comparison.json").read_text(encoding="utf-8"))
    real, checksums = {}, {}
    for name in ("no_query_s17", "film_s17", "cross_s17"):
        folder = args.round / ("actual_" + name)
        result = json.loads((folder / "results.json").read_text(encoding="utf-8"))
        if result["test_accessed"]:
            raise ValueError("Unexpected test-set access during architecture search")
        result["paired_iou_vs_coarse"] = paired_difference(read_jsonl(folder / "refined_predictions.jsonl"),
                                                          read_jsonl(folder / "coarse_predictions.jsonl"))
        result["passes_delivery_guardrail"] = (result["refined"]["coverage_at_90"] >= result["coarse"]["coverage_at_90"]
                                                and result["refined"]["iou"] >= result["coarse"]["iou"])
        real[name] = result
        checksums[name] = file_hash(args.round / name / "best.pt")
    report = {"status": "round_completed_not_project_completed", "seed_count": 1,
              "scope": "source-disjoint development data only; no Agent/export evaluation",
              "controlled_synthetic": controlled["summaries"], "actual_retrieval": real,
              "checkpoint_sha256": checksums, "test_accessed": False,
              "decision": {"retain_for_next_architecture": "film_s17",
                           "cross_attention_not_promoted": "more parameters, lower controlled mIoU and lower real-candidate coverage than FiLM in this seed",
                           "deploy_current_neural_refiner": False,
                           "reason": "actual-candidate coverage_at_90 is worse than leaving the coarse interval unchanged",
                           "next_architecture_proposal_not_implemented": "FiLM with joint interval quality/completeness modeling, explicit uncertainty/fallback; compare under the same cached observations",
                           "remaining_validation": "additional seeds, real-candidate guardrails, final frozen test after architecture selection"}}
    dump(args.round / "round_audit.json", report)
    compact = {name: {"controlled_iou": controlled["summaries"][name]["iou"],
                     "actual_iou": r["refined"]["iou"], "actual_coverage_at_90": r["refined"]["coverage_at_90"],
                     "delivery_guardrail": r["passes_delivery_guardrail"]} for name, r in real.items()}
    print(json.dumps({"results": compact, "decision": report["decision"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
