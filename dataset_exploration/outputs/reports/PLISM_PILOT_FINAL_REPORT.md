# PLISM Pilot Dataset Exploration Report

## 1. Research Context

**Research direction:** Cross-scanner graph topology
stability in computational histopathology.

**Dataset:** PLISM

**Staining condition:** GIVH

**Scanners:** AT2 and GT450

**Feature extractor:** facebook/dinov2-small

**Report generated:** 2026-11-22 19:52 UTC

### Objective

Investigate whether scanner-dependent image feature
differences alter graph connections when corresponding
tissue locations are represented as graph nodes.

This is a pilot feasibility analysis, not a diagnostic
classification or GNN training experiment.

## 2. Dataset Preparation

| Property | Result |
|---|---|
| Matched tissue locations | 90 |
| Scanners | AT2, GT450 |
| Staining condition | GIVH |
| Valid paired images | 180 |
| Grid dimensions | 10 x 10 |
| Grid coverage | 90.00% |
| Tile resolution | 224 x 224 |

Matching tile IDs were identified across both scanner
groups. All selected PNG images were checked for valid
dimensions and readable image contents.

The coordinates were interpreted using structured
tile identifiers. Fine registration accuracy was not
quantitatively measured.

## 3. Image-Level Findings

| Metric | Value |
|---|---:|
| Mean paired pixel MAE | 47.6779 |
| AT2 average brightness | 179.8487 |
| GT450 average brightness | 216.6711 |

GT450 images were visibly paler than AT2 images.

The brightness-based tissue coverage estimate is
sensitive to scanner appearance and must not be
interpreted as a reliable segmentation measurement.

Pixel differences do not directly establish graph
topology instability.

## 4. Feature Extraction

**Model:** DINOv2-Small

**Embedding dimensionality:** 384

**Matched feature pairs:** 90

| Metric | Result |
|---|---:|
| Mean paired cosine similarity | 0.973113 |
| Minimum paired cosine similarity | 0.940725 |
| Maximum paired cosine similarity | 0.986593 |
| Mean Euclidean feature distance | 11.410013 |

The paired embeddings showed high cosine similarity.
However, feature similarity alone cannot establish
stability of nearest-neighbor relationships.

## 5. Cross-Scanner Graph Comparison

Three graph constructions were examined:

1. Spatial-only graph using fixed grid adjacency.
2. Feature-based cosine k-nearest-neighbor graph.
3. Hybrid graph using feature similarity within
   spatially restricted candidate neighbors.

Feature k: 5

Hybrid k: 4

| Graph | AT2 edges | GT450 edges | Shared | Edge Jaccard | Neighbor Jaccard | AT2 components | GT450 components |
|---|---:|---:|---:|---:|---:|---:|---:|
| Spatial-only | 306 | 306 | 306 | 1.0000 | 1.0000 | 1 | 1 |
| Feature-kNN | 328 | 340 | 176 | 0.3577 | 0.3796 | 1 | 1 |
| Hybrid | 231 | 238 | 208 | 0.7969 | 0.8154 | 1 | 1 |

### Interpretation

**Spatial-only graph:** Identical graph connectivity
was expected because both scanners used the same
grid coordinates and fixed adjacency rule.

**Feature-based graph:** Different scanners produced
substantially different feature-selected edges within
the same matched region.

**Hybrid graph:** Restricting candidate neighbors
spatially produced greater edge agreement in this
specific experiment.

The hybrid and feature-based approaches use different
neighborhood restrictions and k values. Their
differences cannot be attributed solely to one factor.

All graph types remained connected in both scanners.

## 6. Main Pilot Finding

Paired tissue images had high feature cosine
similarity, yet feature-based nearest-neighbor graph
construction produced substantially different edges
between AT2 and GT450.

This establishes measurable graph construction
sensitivity under the tested configuration.

It does not establish graph coarsening failure,
causal effects on classification, or generalization
across additional tissue samples and scanners.

## 7. Limitations

1. Only one selected tissue region was analyzed.
2. Only two scanner conditions were evaluated.
3. The pilot used 90 matched patch locations.
4. DINOv2-Small is not pathology-specific.
5. Registration accuracy was not measured quantitatively.
6. Graph hyperparameter sensitivity was not tested.
7. No learned graph pooling or coarsening was evaluated.
8. No diagnostic classification labels were used.
9. No GNN was trained.
10. No inferential statistical testing was conducted.

## 8. Recommendations for Phase 2

The next research phase should:

1. Define explicit hypotheses and evaluation protocols.
2. Expand to independently sampled tissue regions.
3. Include additional scanners and staining conditions.
4. Compare multiple feature extractors.
5. Test different graph k values and neighborhood rules.
6. Evaluate registered graph topology consistency.
7. Investigate pooling/coarsening stability separately.
8. Evaluate clinical classification only when suitable
   labeled data and benchmarks are available.

## 9. Exploration Phase Decision

**Pilot dataset exploration: COMPLETE.**

The observations justify further investigation of
cross-scanner graph topology stability.

The evidence is preliminary and does not yet
demonstrate a new model contribution or establish
clinical improvement.

The next stage is research protocol design,
followed by controlled experiments.
