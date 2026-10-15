"""Simple face-region quality measures (sharpness / activation energy)."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from locfix.models.fr_net import REGION_SLICES


def region_qualities(images: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Per-region quality in [0,1] from Laplacian variance (sharpness).
    images: (B,C,H,W) normalized roughly in [-1,1].
    """
    # to single channel [0,1]
    x = images.mean(dim=1, keepdim=True)
    x = (x + 1.0) * 0.5
    lap_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=images.dtype, device=images.device
    ).view(1, 1, 3, 3)
    lap = F.conv2d(x, lap_kernel, padding=1)
    b, _, h, w = lap.shape
    out = {}
    for name, (y0, y1) in REGION_SLICES.items():
        a, b_ = int(y0 * h), max(int(y1 * h), int(y0 * h) + 1)
        patch = lap[:, :, a:b_, :]
        var = patch.var(dim=(1, 2, 3)).clamp(min=0)
        # squash to (0,1)
        q = 1.0 - torch.exp(-var * 10.0)
        out[name] = q
    return out


def quality_weighted_score(
    emb_a: Dict[str, torch.Tensor],
    emb_b: Dict[str, torch.Tensor],
    qa: Dict[str, torch.Tensor],
    qb: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Weighted sum of regional cosines; weights = min(qa,qb)."""
    score = None
    wsum = None
    for name in emb_a:
        cos = (emb_a[name] * emb_b[name]).sum(dim=1)
        w = torch.minimum(qa[name], qb[name]).clamp(min=1e-3)
        score = cos * w if score is None else score + cos * w
        wsum = w if wsum is None else wsum + w
    return score / wsum
