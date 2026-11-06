"""The paper's premise, reproduced with this repo's matching on synthetic regions.

Abstract: illumination, pose and occlusion hit facial regions unequally, so local
quality should decide which regions the comparison relies on.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from locfix.eval import best_acc_eer
from locfix.models.fr_net import REGION_SLICES
from locfix.quality.measures import quality_weighted_score

PAIRS, DIM = 2000, 32


def make_pairs(seed: int = 0):
    """Half the pairs are mated; in every pair one random region is corrupted."""
    g = torch.Generator().manual_seed(seed)
    names = list(REGION_SLICES)
    labels = torch.arange(PAIRS) % 2
    corrupted = torch.randint(0, len(names), (PAIRS,), generator=g)
    a, b, quality = {}, {}, {}
    for k, name in enumerate(names):
        base = torch.randn(PAIRS, DIM, generator=g)
        other = torch.randn(PAIRS, DIM, generator=g)
        noise = torch.randn(PAIRS, DIM, generator=g)
        partner = torch.where(labels[:, None] == 1, base + 1.5 * noise, other)
        hit = corrupted == k
        partner[hit] = torch.randn(int(hit.sum()), DIM, generator=g)
        a[name], b[name] = F.normalize(base, dim=1), F.normalize(partner, dim=1)
        quality[name] = torch.where(hit, torch.tensor(0.05), torch.tensor(0.9))
    return a, b, quality, labels.numpy()


def test_quality_weighting_beats_uniform_regions_under_local_degradation():
    a, b, quality, labels = make_pairs()
    uniform = {name: torch.ones(PAIRS) for name in a}
    weighted_acc, _ = best_acc_eer(quality_weighted_score(a, b, quality, quality).numpy(), labels)
    uniform_acc, _ = best_acc_eer(quality_weighted_score(a, b, uniform, uniform).numpy(), labels)
    assert weighted_acc > uniform_acc


def test_matching_is_chance_level_without_identity_signal():
    a, _, quality, _ = make_pairs(seed=1)
    _, b, _, _ = make_pairs(seed=2)
    labels = np.random.RandomState(0).randint(0, 2, size=PAIRS)
    acc, _ = best_acc_eer(quality_weighted_score(a, b, quality, quality).numpy(), labels)
    assert acc <= 0.56
