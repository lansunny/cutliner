"""Optional training-only partial-context augmentation, with unmodified GT spans."""

import hashlib
import json
import random

from dataset import BoundaryDataset, coarse_window
from prepare_data import read_jsonl


class IntervalTrainingDataset(BoundaryDataset):
    def __init__(self, *args, partial_context=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.partial_context = partial_context

    def __getitem__(self, index):
        row, variant = self.rows[index // 3], index % 3
        seed = f"train:{self.seed}:epoch:{self.epoch}"
        coarse, window, kind = coarse_window(row, variant, seed)
        if self.partial_context:
            rng = random.Random(hashlib.sha256(f"partial:{seed}:{index}".encode()).hexdigest())
            if rng.random() < .25:
                start, end = row["target_interval_seconds"]
                duration, width = row["duration_seconds"], coarse[1] - coarse[0]
                cut = rng.uniform(.1, .3) * (end - start)
                sides = ["left", "right"]
                rng.shuffle(sides)
                for side in sides:
                    if side == "left":
                        a = start + cut + 6
                        b = min(duration, max(a + width, end - 5.9))
                    else:
                        b = end - cut - 6
                        a = max(0., min(b - width, start + 5.9))
                    if not 0 <= a < b <= duration:
                        continue
                    lower, upper = max(0., a - 6), min(duration, b + 6)
                    intersection = max(0., min(upper, end) - max(lower, start))
                    if .65 <= intersection / (end - start) < .999:
                        coarse, window = (a, b), (lower, upper)
                        break
        # The entire original target remains in the labels. interval_objective
        # masks unobserved endpoints instead of making the crop edge a fake GT.
        return self.observe(row, coarse, window, kind)


class RepresentativeTrainingDataset(BoundaryDataset):
    """Equal mixtures: synthetic perturbation, nearly correct anchor, real top1.

Actual proposals were ranked without GT. GT-derived nearly-correct anchors are
TRAINING augmentation only. No semantic/exhaustive NULL labels are generated.
"""

    def __init__(self, *args, candidates, **kwargs):
        super().__init__(*args, **kwargs)
        if self.split != "train":
            raise ValueError("Representative proposal augmentation is training-only")
        config = json.loads((candidates / "configuration.json").read_text(encoding="utf-8"))
        if config["split"] != "train" or config["ground_truth_used_for_candidate_selection"]:
            raise ValueError("Require independently ranked TRAINING proposals")
        self.proposals = {x["case_id"]: x for x in read_jsonl(candidates / "candidates.jsonl")}
        if self.proposals.keys() != {x["case_id"] for x in self.rows}:
            raise ValueError("Training candidate identity/coverage mismatch")
        for row in self.rows:
            proposal = self.proposals[row["case_id"]]
            if proposal["video_id"] != row["video_id"]:
                raise ValueError("Training proposal points to a different video")

    def __getitem__(self, index):
        row, mode = self.rows[index // 3], index % 3
        seed = f"train:{self.seed}:epoch:{self.epoch}"
        rng = random.Random(hashlib.sha256(f"representative:{seed}:{index}".encode()).hexdigest())
        if mode == 0:
            coarse, window, kind = coarse_window(row, rng.randrange(3), seed)
        else:
            if mode == 1:
                start, end = row["target_interval_seconds"]
                a = max(0., start + max(-.5, min(.5, rng.gauss(0., .2))))
                b = min(row["duration_seconds"], end + max(-.5, min(.5, rng.gauss(0., .2))))
                kind = "nearly_correct_training_anchor"
            else:
                a, b = self.proposals[row["case_id"]]["coarse"]
                kind = "real_training_top1"
            if not 0 <= a < b <= row["duration_seconds"]:
                raise ValueError("Invalid representative training anchor")
            coarse, window = (a, b), (max(0., a - 6), min(row["duration_seconds"], b + 6))
        return self.observe(row, coarse, window, kind)
