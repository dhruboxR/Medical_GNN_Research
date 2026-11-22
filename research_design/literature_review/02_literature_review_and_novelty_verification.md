# Step 2.2: Literature Review and Novelty Verification

**Research project:** Cross-Scanner Graph Topology Stability in Computational Histopathology  
**Phase:** 2, Research Design and Experimental Protocol  
**Review date:** 23 November 2026  
**Status:** Preliminary targeted literature review. Not a systematic review or a verified novelty claim.

## 1. Review objective

Evaluate how existing research covers (a) scanner-dependent image or feature variation, (b) spatial and feature-based graph construction in histopathology, (c) GNN domain generalization, and (d) learned pooling/coarsening. Determine whether a narrower question on **registered, cross-scanner graph stability** merits a controlled study.

The review keeps the following distinctions separate:

- Paired-image feature agreement is not equivalent to agreement between nearest-neighbor graphs.
- Stable graph edges do not imply stable coarsened assignments.
- Stable graphs or representations do not prove improved diagnostic performance.
- A benchmark, an empirical characterization, and a new trainable method are different kinds of research contributions.

## 2. Targeted literature matrix

| ID | Work | Directly relevant contribution | What it does **not** establish for our study |
|---|---|---|---|
| P1 | Ochi et al., **Registered multi-device/staining histology image dataset for domain-agnostic machine learning models** (2024) | Introduces PLISM and registration across imaging conditions. | It does not, by itself, validate our specific tile-coordinate interpretation or demonstrate graph edge stability. |
| P2 | **Lightweight self-supervised learning framework for domain generalization in histopathology** (2025) | Measures differences between embeddings of same-slide regions scanned on different devices and investigates scanner-robust representation learning. | Cross-scanner embedding robustness does not answer our edge-level topology question. |
| P3 | **SPECTRA: Contrastive fine-tuning against stain and scanner bias improves robustness and downstream performance across pathology foundation models** (arXiv preprint, 6 Nov 2026) | Uses spatially matched PLISM image locations to train more scanner- and stain-robust embeddings; evaluates downstream tasks. | Its feature-alignment focus must not be confused with our proposed controlled graph-edge and coarsening stability analysis. Its full methods and supplementary experiments require close reading. |
| P4 | **Scanner-Induced Domain Shifts Undermine the Robustness of Pathology Foundation Models** (2026 preprint) | Assesses scanner effects with a multi-scanner cohort and representation/prediction analyses. | Does not automatically establish the stability of kNN graph edges built over matched tissue locations. |
| P5 | **Reliability of foundation models for image retrieval in histopathology** (2026) | Evaluates cross-scanner consistency of image retrieval and foundation-model representations. | Image retrieval stability differs from global graph adjacency stability. Closely relevant because both depend on nearest-neighbor ranking. |
| P6 | **Enhancing Cross-Institute Generalisation of GNNs in Histopathology** (earlier work) | Directly addresses domain generalization for histopathology GNNs. | Prevents a broad 'first robust pathology GNN' novelty claim; exact adjacency/coarsening tests need further inspection. |
| P7 | **Structure-aware generalization for heterogeneous histopathology via prototype-based multiple instance learning** (2026) | Uses graph-based structural encoding, pooling, and domain-generalization objectives. | Does not, on the evidence reviewed here, establish scanner-paired edge Jaccard benchmarks for the same physical tissue. |
| P8 | **HiGATE: hierarchical graph attention for multi-scale tissue encoder in computational pathology** (2026) | Uses spatially constrained graphs and differentiable pooling to learn tissue-region structure. | Restrained graph construction and pooling already exist; cross-scanner stability of learned assignments remains an empirical question. |
| P9 | **WSI-GT: Pseudo-Label Guided Graph Transformer for Whole-Slide Histology** (2026) | Examines spatial versus feature kNN and neighborhood size in histopathology graph design. | These choices are not novel alone. A scanner-matched stability protocol would need to be a separately demonstrated contribution. |
| P10 | **A protocol for evaluating robustness to H&E staining variation in computational pathology models** (2026) | Provides robustness-evaluation framing and uses PLISM as a reference library. | Stain robustness and scanner-induced topology stability must not be conflated. |

## 3. Source links

- **P1:** https://doi.org/10.1038/s41597-024-03122-5
- **P2:** https://doi.org/10.1038/s41598-025-20411-0
- **P3:** https://arxiv.org/abs/2611.10093
- **P4:** https://arxiv.org/abs/2601.04163
- **P5:** https://doi.org/10.1038/s44303-026-00174-7
- **P6:** https://ora.ox.ac.uk/objects/uuid:1cfc9dc2-357b-48a0-808b-9fc5b6278eb4
- **P7:** https://doi.org/10.1038/s41746-025-02289-4
- **P8:** https://doi.org/10.3389/fonc.2026.1798040
- **P9:** https://www.mdpi.com/2504-4990/8/1/8
- **P10:** https://pmc.ncbi.nlm.nih.gov/articles/PMC13545398/

## 4. Preliminary gap assessment

### Already established in the literature

1. Scanner-dependent changes can affect histopathology image appearance and learned representations.
2. Histopathology graphs commonly use spatial proximity, feature similarity, or both.
3. Graph neural networks, domain-generalization techniques, and hierarchical pooling have already been used in histopathology.
4. Registered multi-scanner data, including PLISM, are being used to learn or evaluate acquisition-robust representations.

### Candidate gap requiring deeper verification

**How scanner-induced embedding changes translate into matched-node graph edge disagreements and hierarchical coarsening differences, under controlled graph budgets and independent tissue-region replication, is not clearly resolved by the targeted papers examined here.**

This is a **provisional gap**, not a claim that no previous paper has investigated it. Citation chaining and full-text review may reveal closely related work.

### Claim boundaries

Do **not** claim:

- First use of PLISM for scanner robustness.
- First demonstration of scanner-related embedding differences.
- First spatially constrained histopathology graph.
- First use of differentiable pooling in histopathology.
- Proven improvement to diagnostic classification through topology stability.

## 5. Pilot evidence motivating the question

The completed Phase 1 PLISM pilot used 90 matching tile IDs across AT2 and GT450 under the GIVH staining condition. A frozen DINOv2-Small encoder generated 384-dimensional representations.

| Pilot metric | Value |
|---|---:|
| Paired cosine similarity, mean | 0.973113 |
| Spatial-only edge Jaccard | 1.000000 |
| Feature-kNN edge Jaccard | 0.357724 |
| Hybrid spatial-feature edge Jaccard | 0.796935 |

These values come from one region and one encoder. Spatial graph agreement is guaranteed by identical coordinate-based construction. Feature-kNN and hybrid construction used different candidate sets and k settings, so their numerical difference is not a controlled causal comparison.

## 6. Refined research questions

**RQ1: Representation drift.** How do frozen general-purpose versus pathology-specific image encoders differ in paired scanner feature agreement and neighbor ranking stability?

**RQ2: Graph construction sensitivity.** Under matched nodes and controlled graph budgets, how much graph adjacency differs across scanners for spatial, feature, and spatially constrained feature graphs?

**RQ3: Coarsening stability.** Do scanner-induced changes in node features or graph edges alter clustering/pooling assignments, coarsened adjacency, or region-level representations?

**RQ4: Downstream relevance (conditional).** When paired labels and appropriate graph tasks are available, do improvements in graph/coarsening stability associate with improved cross-scanner prediction consistency or generalization?

## 7. Experimental controls needed before claiming a contribution

1. Independently sampled tissue regions, preferably from multiple slides and scanning combinations.
2. Slide-grouped partitions and paired within-location comparisons.
3. Verified coordinate semantics and registration quality checks.
4. Fixed feature extractor preprocessing and deterministic inference.
5. Matched edge density or matched candidate-neighborhood budgets; sweep k and spatial radius.
6. Distinguish directed kNN, symmetrized kNN, and mutual kNN.
7. Compute edge Jaccard, per-node neighbor Jaccard, rank correlation, and component/degree statistics.
8. Use uncertainty intervals and paired region-level statistics rather than treating tiles as independent replicates.
9. Compare at least one pathology-specific encoder to DINOv2 where access and licensing permit.
10. Treat graph pooling as a separate stage, with matched coarsening cardinality and reproducible assignment comparisons.


