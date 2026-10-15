# REPORT — Local Fixations and Quality Measures for Deep Face Recognition

**Protocol:** `full`

Lab twin of Méndez-Llanes et al. (TBIOM): eyes/nose/mouth fixation embeddings
with Laplacian-sharpness quality weights for verification.

## Training

- Epochs: 8
- Train images: 141610
- Identities: 7124
- Elapsed (s): 1184.1

![Training](outputs/figures/full/train_loss.svg)

## Verification

| Dataset | Global acc | Global EER | QW acc | QW EER | N |
|---------|------------|------------|--------|--------|---|
| calfw | 53.40% | 46.60% | 53.55% | 46.45% | 6000 |
| cplfw | 52.90% | 47.10% | 52.62% | 47.38% | 6000 |
| lfw | 58.93% | 41.07% | 57.68% | 42.32% | 6000 |

![Verification](outputs/figures/full/verification.svg)

## Notes

- Paper uses pretrained deep nets + landmark fixations; this rebuild trains ResNet-18 on CelebA
  with horizontal band fixations and sharpness quality.
- See `DEVIATIONS.md`.
