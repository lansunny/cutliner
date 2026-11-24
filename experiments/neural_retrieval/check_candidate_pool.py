"""Offline checks for the candidate limit before and after boundary refinement."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mini_agent.video.candidate_pool import finalize_refined_pool, select_distinct


def candidate(start, end, index=None):
    row = {"start_seconds": start, "end_seconds": end, "model_evidence": []}
    if index is not None:
        row["candidate_id"] = f"neural_{index:04d}"
    return row


class CandidatePoolTest(unittest.TestCase):
    def test_duplicates_do_not_consume_slots(self):
        first = candidate(10, 30)
        tail = [candidate(50 + i * 10, 55 + i * 10) for i in range(11)]
        ranked = [first, candidate(11, 15), candidate(17, 38), *tail]
        self.assertEqual(select_distinct(iter(ranked), 12), [first, *tail])

    def test_small_overlap_is_not_a_duplicate(self):
        ranked = [candidate(10, 20), candidate(19, 25), candidate(25, 30)]
        self.assertEqual(select_distinct(ranked, 12), ranked)

    def test_refined_overlap_backfills_from_reserve(self):
        coarse = [candidate(10, 20, 1), candidate(22, 31, 2), candidate(40, 50, 3)]
        first = {**coarse[0], "end_seconds": 30, "boundary_refinement": {"example": True}}
        report = {
            "coarse_predictions": coarse,
            "predictions": [first, coarse[1].copy(), coarse[2].copy()],
            "candidate_reserve": [candidate(44, 47), candidate(60, 70)],
        }
        finalize_refined_pool(report, 3)
        rows = report["predictions"]
        self.assertEqual([r["candidate_id"] for r in rows], ["neural_0001", "neural_0003", "neural_0004"])
        self.assertEqual([r["review_priority"] for r in rows], [1, 2, 3])
        self.assertEqual(rows[0]["boundary_refinement"], {"example": True})
        self.assertEqual(coarse[0]["end_seconds"], 20)
        self.assertNotIn("candidate_reserve", report)


if __name__ == "__main__":
    unittest.main()
