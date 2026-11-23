# Phase 3B: Final Research Summary

**Status:** CLOSED  
**Dataset:** Owkin PLISM  
**Scanner pair:** AT2 / GT450  
**Selected regions:** 10  
**Matched tile positions:** 360  
**Original PNG images:** 720  
**Encoder:** DINOv2-Small (384-dimensional CLS embeddings)

## Research objective

Investigate cross-scanner graph topology instability and
representation consistency under controlled graph construction.

## Confirmed findings

1. Global and spatial feature-similarity graphs select different
   edge sets between scanner acquisitions.
2. Spatial graph construction generally improves raw edge and
   node-neighborhood stability at matched edge budgets.
3. Connected spatial graphs preserve a residual-edge stability
   advantage in the selected cohort.
4. Geometry-conditioned adjustment reduces the size and
   consistency of the spatial advantage.
5. Deterministic spatial propagation exhibits smaller
   topology-induced representation differences.
6. Spatial propagation also leads to greater node-variance loss.
7. Equal-strength propagation retains the observed topology
   sensitivity advantage.

## Reference comparison

At **72 edges**, depth **2**, alpha **0.50**:

| Metric | Result |
|---|---:|
| Global topology L2 | 3.681058 |
| Spatial topology L2 | 2.452613 |
| Relative L2 reduction | 33.37% |
| Regions favoring spatial | 10/10 |
| Geometry-adjusted regions favoring spatial | 7/10 |
| Regions with lower spatial variance retention | 10/10 |

These are descriptive within-acquisition results.

## Limitations

- Only one registered acquisition pair was investigated.
- Ten spatial regions are not verified independent specimens.
- Manual registration quality review remains unconfirmed.
- Graph-construction and smoothing effects remain interrelated.
- No downstream clinical labels or trained GNN have been evaluated.
- Novelty and broader generalization have not been established.

## Phase 4 direction

Develop and assess a stability-aware graph construction approach
that balances cross-scanner robustness with representation diversity.

Before making downstream performance claims, establish a defensible
training objective, non-collapse controls, baseline comparisons,
and independent evaluation where possible.

## Research preservation

- Frozen manifest SHA256: `69d11e1a0dd1107f2e6534632727e2c2291ff83d6b17cf953ed5056bea9020f9`
- Phase 3B feature archive and experiment tables are unchanged.
- Previous research results remain historical records.
- Phase 3B is closed after Step 3B.28.
