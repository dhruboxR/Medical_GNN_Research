# Phase 3A: PLISM Pilot Experimental Research

## Project

**Cross-Scanner Graph Topology Stability in Computational Histopathology**

This folder contains pilot experiments investigating graph topology
and representation stability across matched histopathology tiles
captured using different scanners.

## Status

**Phase 3A experimental diagnostics completed.**

The current findings are exploratory and limited to one
90-location tissue region.

## Dataset and Feature Representation

- Dataset: PLISM, using registered tiles distributed by Owkin.
- Scanner conditions: AT2 and GT450.
- Staining condition: GIVH.
- Matched nodes: 90.
- Feature encoder: `facebook/dinov2-small`.
- Feature dimensionality: 384.

The two scanner acquisitions correspond to the same selected
tile locations.

## Original Annotation Validation

Official PLISM WSI annotation analysis found:

| Property | Result |
|---|---:|
| Total WSI annotation rows | 310,947 |
| Acquisition groups | 91 |
| GIVH AT2 annotation coordinates | 3,417 |
| GIVH GT450 annotation coordinates | 3,417 |
| Shared coordinate strings | 3,417 |
| Shared coordinate tissue-label agreement | 100% |
| Label conflicts | 0 |

Original WSI coordinates have not been verified against the
registered Owkin tile-coordinate system.

Tissue-type identifiers do not establish patient-level or
specimen-level biological independence.

## Experiment Overview

### 3A.1: Dataset and Provenance Audits

Scripts 01-08 examine:

- Paired tile inventory.
- Scanner acquisition metadata.
- Specimen identity availability.
- Official PLISM annotation resources.
- Annotation schemas.
- Tissue labels and coordinate consistency.

### 3A.2: Matched Edge-Budget Stability

Script 09 compares global and spatial-feature graph construction
with equal edge budgets.

| Budget | Global Jaccard | Spatial Jaccard |
|---|---:|---:|
| 90 | 0.241379 | 0.538462 |
| 135 | 0.285714 | 0.708861 |
| 180 | 0.304348 | 0.791045 |
| 225 | 0.347305 | 0.829268 |

The unrestricted top-M construction resulted in disconnected graphs.

### 3A.3: Random and Connectivity Diagnostics

Script 10 assesses:

- Candidate-pool effects.
- Empirical random-overlap references.
- Graph fragmentation.
- Isolated nodes.

Raw stability scores alone are insufficient to compare methods
with different eligible candidate-pool sizes.

### 3A.4: Connectivity-Preserving Graph Construction

Script 11 introduces a common 89-edge spatial spanning tree
and adds feature-selected edges.

All tested graph configurations are connected.

Residual feature-edge agreement excludes the fixed backbone:

| Budget | Global residual Jaccard | Spatial residual Jaccard |
|---|---:|---:|
| 135 | 0.179487 | 0.483871 |
| 180 | 0.213333 | 0.716981 |
| 225 | 0.276995 | 0.813333 |

Spatially constrained graph construction produced greater
residual edge agreement at all three evaluated budgets.

### 3A.5: Node-Level Structural Stability

Script 12 measures:

- Residual degree distributions.
- Nodes connected only through the backbone.
- Neighbor-set consistency.
- Long-distance versus local edges.
- Cross-scanner degree variation.

### 3A.6: Representation Consistency

Script 13 uses deterministic row-normalized propagation
with self-loops and no learned GNN weights.

At propagation depth 2:

| Method | Budget | Mean paired-node cosine |
|---|---:|---:|
| Global + MST | 135 | 0.984529 |
| Spatial + MST | 135 | 0.985045 |
| Global + MST | 180 | 0.985101 |
| Spatial + MST | 180 | 0.985820 |
| Global + MST | 225 | 0.985559 |
| Spatial + MST | 225 | 0.986704 |
| Backbone-only | 89 | 0.984597 |

These differences are descriptive and have not been assessed
through independent biological replication.

### 3A.7: Oversmoothing and Topology Sensitivity

Script 14 evaluates:

- Within-graph embedding dispersion.
- Cross-scanner representation differences.
- Fixed-feature topology sensitivity.
- Dispersion retention at increasing propagation depths.

Repeated propagation increased cross-scanner similarity
but reduced node embedding dispersion.

This is consistent with smoothing and means that greater
similarity alone should not be interpreted as improved
representation quality.

## Final Figures

All final figures are saved in `outputs/figures/`.

| Figure | Description |
|---|---|
| 01 | Matched-budget graph stability |
| 02 | Stability above random reference |
| 03 | Graph fragmentation |
| 04 | Connected residual-edge stability |
| 05 | Node-neighborhood stability |
| 06 | Topology sensitivity |
| 07 | Representation consistency |
| 08 | Oversmoothing and dispersion retention |

Every figure is generated as a PNG and vector PDF.

The exact files are listed in
`outputs/figures/FIGURE_INVENTORY.csv`.

## Research Findings

The pilot provides evidence that spatially constrained
feature-based graphs have more stable scanner-paired
neighborhood structure under the tested conditions.

The effect remains visible after matching edge budgets,
enforcing connectivity, and excluding a fixed backbone.

However, representation differences are substantially smaller
than raw edge differences under deterministic propagation.

The observed representation consistency is accompanied
by decreased embedding dispersion at greater depths.

## Important Limitations

1. Only one 90-node tissue region was analyzed.
2. Biological specimen identities have not been verified.
3. The original WSI and Owkin coordinate systems have not
   been linked by a validated transform.
4. Candidate edge pools differ between global and spatial methods.
5. Fixed backbone edges artificially raise full-graph agreement.
6. Graph representation analysis is deterministic, not trained GNN evaluation.
7. Graph coarsening and pooling stability remain untested.
8. No downstream classification performance is established.
9. No independent specimen-level inferential conclusions are supported.

## Directory Structure

```text
experimental_research/
├── README.md
├── scripts/
├── Docs/
└── outputs/
    ├── tables/
    ├── reports/
    └── figures/
```

## Reproducing Figures

From the project root:

```bash
python experimental_research/scripts/15_finalize_phase3a.py
```

Requires NumPy, pandas, and Matplotlib.

## Next Research Phase

**Phase 3B: Multi-Region Controlled Experiments**

Location: `../multi_region_research/`

The next phase should prioritize spatially separated matched
regions, tissue attribution, independent analysis-unit verification,
and multi-region experimental replication before strong
generalization claims.

## Research References

- Official PLISM dataset:
  https://doi.org/10.1038/s41597-024-03122-5
- Owkin registered PLISM tiles:
  https://huggingface.co/datasets/owkin/plism-dataset-tiles

## Conclusion

Phase 3A establishes a reproducible exploratory baseline
for cross-scanner graph topology stability.

Its results motivate expanded evaluation, but do not yet
demonstrate clinically useful prediction improvements or
generalization across independent specimens.
