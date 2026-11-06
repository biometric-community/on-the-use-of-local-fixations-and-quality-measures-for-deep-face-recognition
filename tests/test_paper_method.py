"""Regions, quality measures and metrics, checked against the method description."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from locfix.eval import best_acc_eer
from locfix.models.fr_net import REGION_SLICES, FixationFaceNet
from locfix.quality.measures import quality_weighted_score, region_qualities


@pytest.mark.parametrize("region", list(REGION_SLICES))
def test_pooled_rows_cover_the_declared_band(region):
    """Each band must pool every feature-map row it overlaps at the real input size."""
    model = FixationFaceNet(10, pretrained=False).eval()
    with torch.no_grad():
        h = model.stem(torch.zeros(1, 3, 112, 112)).shape[2]
    y0, y1 = REGION_SLICES[region]
    expected = {r for r in range(h) if (r + 1) / h > y0 and r / h < y1}
    pooled = set()
    for r in range(h):
        feat = torch.zeros(1, 1, h, h)
        feat[:, :, r, :] = 1.0
        if model._region_pool(feat, y0, y1).item() > 0:
            pooled.add(r)
    assert pooled == expected


def test_blurred_region_gets_lower_quality():
    torch.manual_seed(0)
    image = 0.02 * torch.randn(1, 3, 112, 112)
    image[:, :, 70:, :] = F.avg_pool2d(image, 5, stride=1, padding=2)[:, :, 70:, :]
    q = region_qualities(image)
    assert q["mouth"].item() < q["eyes"].item()


def test_quality_values_stay_in_unit_range():
    torch.manual_seed(1)
    q = region_qualities(torch.rand(4, 3, 112, 112) * 2 - 1)
    for value in q.values():
        assert ((value >= 0) & (value <= 1)).all()


def unit_emb(sign: float) -> torch.Tensor:
    return torch.full((1, 4), sign * 0.5)


def test_score_ignores_a_region_with_no_quality():
    """Abstract: the representation is built from the regions that score well."""
    a = {"eyes": unit_emb(1), "nose": unit_emb(1), "mouth": unit_emb(1)}
    b = {"eyes": unit_emb(1), "nose": unit_emb(1), "mouth": unit_emb(-1)}
    good = torch.tensor([0.5])
    qa = {"eyes": good, "nose": good, "mouth": torch.tensor([0.0])}
    qb = {"eyes": good, "nose": good, "mouth": good}
    assert quality_weighted_score(a, b, qa, qb).item() == pytest.approx(1.0, abs=0.01)


def test_equal_quality_gives_the_mean_regional_cosine():
    a = {"eyes": unit_emb(1), "nose": unit_emb(1), "mouth": unit_emb(1)}
    b = {"eyes": unit_emb(1), "nose": unit_emb(1), "mouth": unit_emb(-1)}
    q = {name: torch.tensor([0.5]) for name in a}
    assert quality_weighted_score(a, b, q, q).item() == pytest.approx(1.0 / 3.0, abs=1e-5)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def test_eer_is_the_point_where_fmr_equals_fnmr():
    lo, hi = 0.0, 1.0
    for _ in range(80):
        t = 0.5 * (lo + hi)
        fmr, fnmr = 1.0 - _norm_cdf(t), _norm_cdf((t - 1.0) / 0.1)
        lo, hi = (t, hi) if fmr > fnmr else (lo, t)
    expected = 1.0 - _norm_cdf(t)

    rng = np.random.RandomState(3)
    scores = np.concatenate([rng.normal(1.0, 0.1, 200000), rng.normal(0.0, 1.0, 200000)])
    labels = np.concatenate([np.ones(200000, dtype=int), np.zeros(200000, dtype=int)])
    _, eer = best_acc_eer(scores, labels)
    assert eer == pytest.approx(expected, abs=0.01)
