"""Rank locally; widening this pool does not increase the review API budget."""

import math
from operator import itemgetter

import numpy as np

from mini_agent.video.candidate_pool import select_distinct

_BOUNDS = itemgetter("start", "end")


def _ordinals(values):
    """Conservative tied ranks; a constant signal contributes equally."""
    ordered = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    return np.searchsorted(-ordered, -np.asarray(values, dtype=np.float64), side="right")


def mix_scales(long, short, top_k):
    """Keep the best long-context result; reserve room for shorter actions."""
    ordered = []
    for i in range(max(len(long), 2*len(short))):
        ordered.extend(long[i:i+1])
        if (i+1) % 2 == 0:
            ordered.extend(short[i//2:i//2+1])
    return ordered if top_k is None else select_distinct(ordered, top_k, bounds=_BOUNDS)


def rerank(proposals, frame_scores, cadence, start, top_k, *, mode, rank_all=False):
    """proposals[0] and frame_scores[:,0] describe the original full query."""
    if mode not in {"frame", "consensus"}:
        raise ValueError("Unknown visual ranker")
    tracks = proposals if mode == "consensus" else proposals[:1]
    pool = []
    for channel, ranked in enumerate(tracks):
        per_window = {}
        for i, candidate in enumerate(ranked, 1):
            window = candidate["window"]
            per_window[window] = per_window.get(window, 0) + 1
            if i <= max(48, top_k * 5) or per_window[window] <= 2:
                pool.append({**candidate, "track": channel, "track_rank": i})
    if not pool:
        return [], 0

    def clip_support(candidate):
        lo = max(0, math.ceil((candidate["start"] - start) / cadence - .5))
        hi = min(len(frame_scores), max(lo + 1, math.ceil((candidate["end"] - start) / cadence - .5)))
        if lo >= hi:
            lo, hi = min(lo, len(frame_scores) - 1), min(lo, len(frame_scores) - 1) + 1
        scores = frame_scores[lo:hi]
        count = min(3, len(scores))
        peaks = np.partition(scores, -count, axis=0)[-count:].mean(axis=0)
        return .75 * peaks + .25 * scores.mean(axis=0)

    support = np.stack([clip_support(row) for row in pool])
    visual = _ordinals(support[:, 0])
    if len(tracks) > 1:
        # The auxiliary anchors only help when supported in the same interval.
        secondary = _ordinals(support[:, 1:len(tracks)].mean(axis=1))
    else:
        secondary = visual
    ranked_primary = tracks[0]
    bounds = np.array([(c["start"], c["end"]) for c in ranked_primary], dtype=np.float64).reshape(-1, 2)
    for i, p in enumerate(pool):
        if p["track"] == 0:
            primary = p["track_rank"]
        else:
            inter = np.maximum(0., np.minimum(p["end"], bounds[:, 1]) - np.maximum(p["start"], bounds[:, 0]))
            union = np.maximum(p["end"], bounds[:, 1]) - np.minimum(p["start"], bounds[:, 0])
            overlap = np.flatnonzero(inter >= .4 * union)
            primary = int(overlap[0]) + 1 if len(overlap) else len(ranked_primary) + 1
        # Different head and cosine scores do not share a confidence scale.
        # Their ranks do. Preserve the original question as the strongest signal.
        p["rank_score"] = (.50 / (20 + primary) + .35 / (20 + visual[i])
                           + .15 / (20 + secondary[i]))
        p["primary_rank"] = primary
    ordered = sorted(pool, key=lambda p: (-p["rank_score"], p["primary_rank"], p["track"], p["start"]))
    return (ordered if rank_all else select_distinct(ordered, top_k, bounds=_BOUNDS)), len(pool)
