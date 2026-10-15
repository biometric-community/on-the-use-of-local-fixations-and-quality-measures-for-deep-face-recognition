"""ResNet-18 with regional fixation features from last conv map."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Horizontal bands approximating eyes / nose / mouth on aligned faces
REGION_SLICES = {
    "eyes": (0.15, 0.45),
    "nose": (0.40, 0.70),
    "mouth": (0.65, 0.95),
}


class FixationFaceNet(nn.Module):
    def __init__(self, n_classes: int, emb_dim: int = 128, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        # keep up to layer4 (before avgpool)
        self.stem = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)
        self.classifier = nn.Linear(emb_dim, n_classes)
        self.region_fc = nn.ModuleDict(
            {name: nn.Linear(512, emb_dim) for name in REGION_SLICES}
        )
        self.region_bn = nn.ModuleDict(
            {name: nn.BatchNorm1d(emb_dim) for name in REGION_SLICES}
        )

    def _global_emb(self, feat: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(self.gap(feat), 1)
        return F.normalize(self.bn(self.fc(x)), p=2, dim=1)

    def _region_pool(self, feat: torch.Tensor, y0: float, y1: float) -> torch.Tensor:
        _, _, h, _ = feat.shape
        a, b = int(y0 * h), max(int(y1 * h), int(y0 * h) + 1)
        return torch.flatten(feat[:, :, a:b, :].mean(dim=(2, 3)), 1)

    def region_embeddings(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.stem(x)
        out = {}
        for name, (y0, y1) in REGION_SLICES.items():
            pooled = self._region_pool(feat, y0, y1)
            out[name] = F.normalize(self.region_bn[name](self.region_fc[name](pooled)), p=2, dim=1)
        return out

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self._global_emb(self.stem(x))

    def forward(self, x: torch.Tensor):
        feat = self.stem(x)
        emb = self._global_emb(feat)
        return self.classifier(emb), emb

    def forward_regions(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        feat = self.stem(x)
        emb = self._global_emb(feat)
        regs = {}
        for name, (y0, y1) in REGION_SLICES.items():
            pooled = self._region_pool(feat, y0, y1)
            regs[name] = F.normalize(self.region_bn[name](self.region_fc[name](pooled)), p=2, dim=1)
        return emb, regs
