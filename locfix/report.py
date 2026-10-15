"""Figures + REPORT.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from locfix.plot_style import FIGSIZE, LINEWIDTH_MAIN, MULTI_SERIES_COLORS, apply_rcparams, apply_style


def plot_train(hist, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot([h["epoch"] for h in hist], [h["loss"] for h in hist], color=MULTI_SERIES_COLORS[0], linewidth=LINEWIDTH_MAIN)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Local-fixation FR training (CelebA)")
    apply_style(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_verify(eval_res: dict, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ver = eval_res.get("verification", {})
    labels, gvals, qvals = [], [], []
    for name, v in ver.items():
        labels.append(name)
        gvals.append(v["global"]["accuracy"] * 100)
        qvals.append(v["quality_weighted"]["accuracy"] * 100)
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, gvals, w, label="global", color=MULTI_SERIES_COLORS[0], edgecolor="#002080")
    ax.bar(x + w / 2, qvals, w, label="quality-weighted", color=MULTI_SERIES_COLORS[1], edgecolor="#002080")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Global vs quality-weighted regional matching")
    ax.legend()
    apply_style(ax, grid_axis="y")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def write_report(cfg, train_s, eval_res, fig_dir: Path, path: Path) -> None:
    protocol = cfg["train"].get("protocol", "full")
    lines = [
        "# REPORT — Local Fixations and Quality Measures for Deep Face Recognition",
        "",
        f"**Protocol:** `{protocol}`",
        "",
        "Lab twin of Méndez-Llanes et al. (TBIOM): eyes/nose/mouth fixation embeddings",
        "with Laplacian-sharpness quality weights for verification.",
        "",
        "## Training",
        "",
        f"- Epochs: {train_s.get('epochs')}",
        f"- Train images: {train_s.get('n_train')}",
        f"- Identities: {train_s.get('num_classes')}",
        f"- Elapsed (s): {train_s.get('elapsed_sec', 0):.1f}",
        "",
        f"![Training]({fig_dir.as_posix()}/train_loss.svg)",
        "",
        "## Verification",
        "",
        "| Dataset | Global acc | Global EER | QW acc | QW EER | N |",
        "|---------|------------|------------|--------|--------|---|",
    ]
    for name, v in sorted(eval_res.get("verification", {}).items()):
        g, q = v["global"], v["quality_weighted"]
        lines.append(
            f"| {name} | {g['accuracy']*100:.2f}% | {g['eer']*100:.2f}% | "
            f"{q['accuracy']*100:.2f}% | {q['eer']*100:.2f}% | {g['n_pairs']} |"
        )
    lines += [
        "",
        f"![Verification]({fig_dir.as_posix()}/verification.svg)",
        "",
        "## Notes",
        "",
        "- Paper uses pretrained deep nets + landmark fixations; this rebuild trains ResNet-18 on CelebA",
        "  with horizontal band fixations and sharpness quality.",
        "- See `DEVIATIONS.md`.",
        "",
    ]
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def generate(cfg: dict) -> None:
    log_dir = Path(cfg["paths"]["log_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    train_s = json.loads((log_dir / "train_summary.json").read_text())
    eval_res = json.loads((log_dir / "eval.json").read_text())
    plot_train(train_s["history"], fig_dir / "train_loss.svg")
    plot_verify(eval_res, fig_dir / "verification.svg")
    write_report(cfg, train_s, eval_res, fig_dir, Path("REPORT.md"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    generate(yaml.safe_load(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
