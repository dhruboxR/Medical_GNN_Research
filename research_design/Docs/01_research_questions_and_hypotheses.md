
# Research Questions, Hypotheses, and Scope

## 1. Working Research Title

Cross-Scanner Graph Topology Stability in Computational Histopathology

**Research phase:** Phase 2 — Research Design and Experimental Protocol

**Status:** Initial research design; pending literature and novelty verification.

---

## 2. Research Background

Whole-slide histopathology images can exhibit differences in
color, brightness, contrast, and learned feature representations
when scanned using different devices.

Graph-based histopathology methods represent image patches as
nodes and construct edges using spatial relationships, feature
similarity, or combinations of both.

Scanner-dependent feature variation may influence graph
construction, potentially altering neighborhood relationships
between corresponding tissue patches.

The impact of these changes on graph topology, coarsening,
and downstream model behavior requires systematic evaluation.

---

## 3. Preliminary Evidence

Phase 1 examined 90 matched PLISM tissue locations across
the AT2 and GT450 scanners under the GIVH staining condition.

Feature extraction used the pretrained DINOv2-Small model.

### Pilot Results

| Metric | Value |
|---|---:|
| Matched tissue locations | 90 |
| Feature dimensions | 384 |
| Mean paired cosine similarity | 0.973113 |
| Spatial graph edge Jaccard | 1.000000 |
| Feature-kNN edge Jaccard | 0.357724 |
| Hybrid graph edge Jaccard | 0.796935 |

These findings suggest that high feature similarity does not
necessarily imply stable nearest-neighbor graph construction.

However, the pilot involved one tissue region, one encoder,
and different graph construction constraints.

The findings are preliminary and cannot establish general
scanner robustness or diagnostic performance.

---

## 4. Primary Research Objective

To investigate how scanner-dependent differences in image
features influence histopathology graph construction and
whether spatial constraints can improve cross-scanner
graph topology stability.

---

## 5. Research Questions

### RQ1: Cross-Scanner Feature Stability

How consistently do pretrained visual feature extractors
represent corresponding histopathology patches across
different scanners?

### RQ2: Graph Topology Stability

How do scanner-dependent feature differences affect
graph topology under different graph construction methods?

### RQ3: Spatial Constraint Effectiveness

Under controlled graph construction settings, do spatial
constraints improve cross-scanner edge and neighborhood
consistency?

### RQ4: Downstream Representation Stability

Does improved graph topology stability lead to more
consistent graph-level representations or predictions
across scanners?

RQ4 will be investigated only when appropriate graph-level
models and evaluation data are available.

---

## 6. Research Hypotheses

### H1: Feature Variation

Registered patches scanned by different devices exhibit
measurable differences in their extracted feature vectors.

**Measurements:**
- Paired cosine similarity
- Euclidean feature distance
- Feature neighborhood consistency

### H2: Graph Sensitivity

Feature-based graph construction exhibits measurable
cross-scanner edge instability, even when matched patch
features have high cosine similarity.

**Measurements:**
- Edge Jaccard similarity
- Neighbor Jaccard similarity
- Edge disagreement rate
- Graph connectivity changes

### H3: Spatial Constraints

Spatially constrained feature graphs achieve higher
cross-scanner edge consistency than unrestricted
feature-based graphs under controlled graph budgets.

**Measurements:**
- Edge Jaccard similarity
- Neighbor consistency
- Edge density
- Sensitivity to neighborhood parameters

### H4: Downstream Consistency

Graph construction methods with greater topology stability
may produce more consistent graph-level representations
across scanners.

**Measurements:**
- Graph embedding cosine similarity
- Graph representation distance
- Prediction agreement, when labels are available

H4 is exploratory and does not assume that greater
topology stability guarantees better diagnostic accuracy.

---

## 7. Research Scope

### Included

- Computational histopathology images
- Registered or spatially corresponding image patches
- Multi-scanner comparisons
- Frozen pretrained image feature extractors
- Spatial, feature-based, and hybrid graph construction
- Graph topology stability evaluation
- Controlled graph construction experiments

### Conditional Extensions

- Learned graph pooling and coarsening
- Graph-level representation consistency
- Diagnostic classification experiments

These extensions require suitable datasets and separate
experimental validation.

### Excluded from the Initial Study

- Developing a new whole-slide segmentation method
- Training image encoders from scratch
- Claiming clinical diagnostic improvements without labels
- Treating adjacent tiles as independent experimental samples

---

## 8. Unit of Analysis

The main experimental unit should be an independently
sampled tissue region or slide-level group.

Individual patches are graph nodes, not independent
replicates for statistical significance testing.

Scanner comparisons must use matched tissue locations
and consistent preprocessing protocols.

---

## 9. Initial Baselines

1. Fixed spatial adjacency
2. Unrestricted feature-based kNN
3. Spatially constrained feature-based kNN
4. Additional graph construction baselines identified
   through literature review

Graph methods should be compared using controlled
edge densities, neighborhood sizes, or clearly reported
construction budgets.

---

## 10. Evaluation Strategy

### Feature-Level Metrics

- Paired cosine similarity
- Feature distance
- Cross-scanner neighborhood agreement

### Graph-Level Metrics

- Edge Jaccard similarity
- Node neighborhood overlap
- Connected components
- Average node degree
- Edge density
- Graph-level structural differences

### Statistical Evaluation

- Multiple independently sampled tissue regions
- Paired comparisons across scanner conditions
- Confidence intervals
- Appropriate paired statistical tests
- Sensitivity analysis across graph hyperparameters

---

## 11. Threats to Validity

- Limited scanner diversity
- Differences in image registration quality
- Tissue-region selection bias
- Scanner and staining confounding
- Feature extractor dependence
- Graph hyperparameter sensitivity
- Non-independent neighboring tissue patches
- Insufficient sample size
- Lack of diagnostic labels

These limitations should be explicitly addressed
in the experimental protocol.

---

## 12. Expected Research Contributions

Potential contributions, subject to experimental evidence:

1. A reproducible evaluation protocol for cross-scanner
   histopathology graph topology stability.

2. A quantitative characterization of how scanner-related
   feature differences influence graph construction.

3. A controlled comparison of spatial, feature-based,
   and spatially constrained graph construction.

4. An investigation of whether graph stability relates
   to downstream representation consistency.

These are proposed contributions, not established findings
or verified novelty claims.

---

## 13. Next Research Step

Step 2.2: Literature Review and Novelty Verification.

The review must establish:

- Which existing studies examine scanner robustness
  in histopathology representations.
- Which histopathology GNN studies evaluate graph
  construction stability.
- Whether cross-scanner graph topology consistency
  has already been investigated.
- Which methodological gaps remain scientifically
  meaningful and experimentally testable.

No novel model architecture will be proposed until
the relevant prior work has been assessed.
