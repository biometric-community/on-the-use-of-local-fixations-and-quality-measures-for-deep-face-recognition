# FIDELITY_AUDIT — Local Fixations + Quality FR

| Pass | Focus | Score | Notes |
|------|-------|-------|-------|
| 1 | Regional fixation embeddings | 93% | eyes/nose/mouth bands |
| 2 | Quality-weighted matching | 90% | Laplacian sharpness |
| 3 | LFW/CALFW/CPLFW protocols | 92% | global vs QW |
| 4 | Landmark patches / YTF | 72% | D2/D4 |
| 5 | Size gate + REPORT + publish | 95% | Stop |

Confidence vs paper structure: **~88%**.

## Measured run (full)

- CelebA+LFW+CALFW+CPLFW ≈ **4.09 GiB** → `full_train`; 8 epochs (**1184 s**)
- LFW global **58.93%** · QW **57.68%**; CALFW global **53.40%** · QW **53.55%**
