"""Select distinct source intervals in retrieval priority order."""

from itertools import chain
from operator import itemgetter


def _same_window(left, right):
    overlap = min(left[1], right[1]) - max(left[0], right[0])
    if overlap <= 0:
        return False
    union = max(left[1], right[1]) - min(left[0], right[0])
    shorter = min(left[1] - left[0], right[1] - right[0])
    # IoU catches shifted cuts; shorter-span coverage also catches nested cuts.
    return overlap >= .45 * union or overlap >= .85 * shorter


def select_distinct(ordered, limit, *, bounds=itemgetter("start_seconds", "end_seconds")):
    """Keep the first representative, and continue past duplicates to fill the limit."""
    selected, spans = [], []
    for item in ordered:
        span = bounds(item)
        if any(_same_window(span, previous) for previous in spans):
            continue
        selected.append(item)
        spans.append(span)
        if len(selected) == limit:
            break
    return selected


def finalize_refined_pool(report, limit):
    """Check the effective boundaries, then fill from the unused coarse ranking."""
    reserve = report.pop("candidate_reserve", [])
    refined = report["predictions"]
    selected = select_distinct(chain(refined, reserve), limit)
    next_id = len(report["coarse_predictions"]) + 1
    for priority, item in enumerate(selected, 1):
        if "candidate_id" not in item:
            item["candidate_id"] = f"neural_{next_id:04d}"
            next_id += 1
        item["review_priority"] = priority
    report["predictions"] = selected
