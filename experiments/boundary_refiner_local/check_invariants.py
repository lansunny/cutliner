"""Code/data invariants; passing these is not evidence of localization quality."""

import numpy as np
import torch

from dataset import collate, coarse_window
from model import BoundaryRefiner, joint_decode, objective


def main():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    rows = []
    for t in (11, 7):
        edges = np.linspace(0, 10, t + 1, dtype=np.float32)
        rng = np.random.default_rng(t)
        rows.append({"visual": rng.normal(size=(t, 512)).astype(np.float32), "times": (edges[:-1] + edges[1:]) / 2,
                     "edges": edges, "tokens": rng.normal(size=(8, 512)).astype(np.float32),
                     "query": rng.normal(size=512).astype(np.float32),
                     "membership": ((edges[:-1] >= 2) & (edges[:-1] < 7)).astype(np.float32),
                     "target": np.array([2., 7.], np.float32), "coarse": np.array([1., 8.], np.float32),
                     "window": np.array([0., 10.], np.float32), "case_id": str(t), "video_id": str(t), "kind": "redundant"})
    batch = collate(rows)
    for variant in ("film", "cross", "no_query", "cross_delta"):
        model = BoundaryRefiner(variant)
        output = model(batch)
        loss, _ = objective(output, batch)
        loss.backward()
        prediction = joint_decode(output, batch)
        assert torch.isfinite(loss) and torch.all(prediction[:, 1] > prediction[:, 0])
        assert model.edge[-1].weight.grad.abs().sum() > 0
        model.eval()
        with torch.no_grad():
            old = model(batch)
            perturbed = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
            perturbed["visual"][~batch["mask"]] = 500
            perturbed["tokens"][~batch["token_mask"]] = 500
            new = model(perturbed)
            assert torch.allclose(old["start"][batch["edge_mask"]], new["start"][batch["edge_mask"]], atol=1e-5)
            if variant == "no_query":
                perturbed["query"] = torch.randn_like(batch["query"])
                perturbed["tokens"] = torch.randn_like(batch["tokens"])
                assert torch.allclose(old["start"], model(perturbed)["start"], atol=1e-5)
        print(f"{variant}: gradients, legal joint spans, padding and query ablation invariants pass")
    for length in (2.5, 7., 15.):
        for start in (.2, 5., 15.):
            row = {"case_id": f"{start}:{length}", "target_interval_seconds": [start, start + length],
                   "duration_seconds": start + length + .3}
            for seed in range(100):
                for kind in range(3):
                    anchor, window, _ = coarse_window(row, kind, str(seed))
                    assert window[0] <= start < start + length <= window[1]
                    assert window[0] <= anchor[0] < anchor[1] <= window[1]
    print("Coarse-derived context and contained positive target invariants pass; no performance claim.")


if __name__ == "__main__":
    main()
