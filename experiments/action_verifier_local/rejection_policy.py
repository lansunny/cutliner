"""Conservative, closed-domain triage; never certify an arbitrary compound query."""

import torch


def eligible(opposite_scores, opposite_top1, minimum=0.5):
    return (opposite_scores >= minimum) & opposite_top1.bool()


def select_rejection_policy(head_scores, labels, opposite_scores, opposite_top1):
    # Fix strong-opposite evidence at .5 a priori. Only the head threshold is
    # selected on validation. Zero observed val errors is not a safety guarantee.
    mask = eligible(opposite_scores, opposite_top1)
    candidates = [0.0, *head_scores.tolist(), 1.0]
    feasible = []
    for threshold in candidates:
        rejected = (head_scores < threshold) & mask
        if not (rejected & labels.bool()).any():
            feasible.append((int((rejected & ~labels.bool()).sum()), -threshold, threshold))
    threshold = max(feasible)[2]
    return {"version": 1, "head_reject_below": float(threshold), "opposite_minimum": 0.5,
            "require_opposite_top1": True, "requires_explicit_closed_domain_opt_in": True,
            "positive_acceptance": False,
            "scope": "short continuous single-primary-action SSV2-like clips only",
            "selection": "max val negative rejection with zero observed val positive rejection"}


def rejection_metrics(scores, labels, opposite_scores, opposite_top1, policy):
    base = eligible(opposite_scores, opposite_top1, policy["opposite_minimum"])
    masks = {"backbone_strong_opposite": base,
             "head_plus_strong_opposite": base & (scores < policy["head_reject_below"])}
    result = {}
    for name, rejected in masks.items():
        positives, negatives = labels.bool(), ~labels.bool()
        pos_rejected, neg_rejected = int((rejected & positives).sum()), int((rejected & negatives).sum())
        result[name] = {"positive_pairs": int(positives.sum()), "negative_pairs": int(negatives.sum()),
                        "positives_rejected": pos_rejected, "negatives_rejected": neg_rejected,
                        "positive_rejection_rate": pos_rejected / max(1, int(positives.sum())),
                        "negative_rejection_rate": neg_rejected / max(1, int(negatives.sum())),
                        "deferred_to_existing_review": int((~rejected).sum())}
    return result


def decide(score, opposite_score, opposite_is_top1, policy, closed_domain=False):
    would_reject = (score < policy["head_reject_below"] and
                    opposite_score >= policy["opposite_minimum"] and opposite_is_top1)
    return {"decision": "reject" if would_reject and closed_domain else "uncertain",
            "would_reject_in_closed_domain": bool(would_reject),
            "closed_domain_opt_in": bool(closed_domain),
            "reason": "trained_head_and_strong_opposite_agree" if would_reject else "insufficient_rejection_evidence",
            "certifies_full_query": False}
