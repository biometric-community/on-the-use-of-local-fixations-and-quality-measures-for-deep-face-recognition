"""LFW/CALFW/CPLFW: global vs quality-weighted regional matching."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from locfix.data.pairs import build_calfw, build_cplfw, build_lfw
from locfix.models.fr_net import FixationFaceNet
from locfix.quality.measures import quality_weighted_score, region_qualities


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def best_acc_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    best_acc, best_eer = 0.0, 1.0
    for t in thresholds:
        pred = (scores >= t).astype(np.int32)
        best_acc = max(best_acc, float((pred == labels).mean()))
        pos, neg = labels == 1, labels == 0
        if pos.any() and neg.any():
            fpr = float((pred[neg] == 1).mean())
            fnr = float((pred[pos] == 0).mean())
            best_eer = min(best_eer, 0.5 * (fpr + fnr))
    return best_acc, best_eer


@torch.no_grad()
def eval_dataset(model, ds, device, batch_size, nw, tag) -> Dict:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=nw)
    global_scores, qw_scores, labels = [], [], []
    model.eval()
    for a, b, y in tqdm(loader, desc=tag, leave=False):
        a, b = a.to(device), b.to(device)
        ea, ra = model.forward_regions(a)
        eb, rb = model.forward_regions(b)
        global_scores.append((ea * eb).sum(dim=1).cpu().numpy())
        qa, qb = region_qualities(a), region_qualities(b)
        qw = quality_weighted_score(ra, rb, qa, qb)
        qw_scores.append(qw.cpu().numpy())
        labels.append(y.numpy())
    g = np.concatenate(global_scores)
    q = np.concatenate(qw_scores)
    y = np.concatenate(labels)
    g_acc, g_eer = best_acc_eer(g, y)
    q_acc, q_eer = best_acc_eer(q, y)
    print(f"[{tag}] global={g_acc:.4f} qw={q_acc:.4f} n={len(y)}")
    return {
        "global": {"accuracy": g_acc, "eer": g_eer, "n_pairs": int(len(y))},
        "quality_weighted": {"accuracy": q_acc, "eer": q_eer, "n_pairs": int(len(y))},
    }


def evaluate(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    ckpt = Path(cfg["paths"]["checkpoint_dir"]) / "last.pt"
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    emb_dim = int(blob.get("emb_dim", cfg["model"].get("emb_dim", 128)))
    model = FixationFaceNet(int(blob["num_classes"]), emb_dim=emb_dim, pretrained=False).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    image_size = int(cfg["model"].get("image_size", 112))
    bs = int(cfg["eval"].get("batch_size", 128))
    nw = int(cfg["train"].get("num_workers", 4))
    max_p = cfg["eval"].get("max_lfw_pairs")
    max_p = int(max_p) if max_p is not None else None

    results: Dict = {"verification": {}}
    for name, builder in (("lfw", build_lfw), ("calfw", build_calfw), ("cplfw", build_cplfw)):
        try:
            ds = builder(cfg, image_size, max_p)
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}")
            continue
        results["verification"][name] = eval_dataset(model, ds, device, bs, nw, name)

    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / "eval.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"Wrote {out}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    evaluate(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
