# IMPLEMENTATION_PLAN — Local Fixations + Quality FR

## Paper

Local facial fixations (eyes/nose/mouth) with quality measures for deep FR matching.

## Lab twin

| Component | Choice |
|-----------|--------|
| Backbone | ResNet-18 last-conv regional pooling |
| Regions | Horizontal bands: eyes / nose / mouth |
| Quality | Laplacian variance sharpness |
| Match | Global cosine vs quality-weighted regional cosine |
| Data | CelebA train; LFW/CALFW/CPLFW eval |
