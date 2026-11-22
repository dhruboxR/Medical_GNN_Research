# Phase 2, Step 2.2B: Focused Literature Comparison and Research-Gap Assessment

**Project:** Medical GNN Research  
**Working topic:** Cross-scanner graph-topology stability in computational histopathology  
**Status:** Focused screening completed; novelty not yet established  
**Literature check:** 23 November 2026

## 1. Purpose

Assess whether prior research already addresses the proposed study's components: (i) registered scanner-paired histopathology tiles, (ii) image-feature instability, (iii) changes in feature-derived patch-graph edges, (iv) spatially constrained graph construction, (v) hierarchical pooling/coarsening consistency, and (vi) downstream prediction consistency. This is a targeted source comparison, **not** a systematic literature review or proof of absence of earlier work.

## 2. Phase 1 pilot evidence (observed, not literature claims)

- Source: `owkin/plism-dataset-tiles`; staining condition `GIVH`; scanners `AT2` and `GT450`.
- 90 matched locations, 180 validated RGB tiles of 224 × 224 pixels, and a provisional 10 × 10 spatial layout.
- Frozen `facebook/dinov2-small`: 384-dimensional embeddings; mean paired cosine similarity **0.973113**.
- Undirected spatial graph: 306 edges per scanner, edge Jaccard **1.000000** (by construction).
- Undirected feature kNN (k=5): 328 AT2 edges, 340 GT450 edges, 176 shared edges, edge Jaccard **0.357724**.
- Spatially constrained feature graph (up to k=4 per node): 231 AT2 edges, 238 GT450 edges, 208 shared edges, edge Jaccard **0.796935**.

**Interpretation:** Matching patch embeddings can retain high cosine similarity even when feature-selected graph edges change considerably. The pilot does not establish statistical generality, graph-pooling behavior, causality, or diagnostic benefit. Graph budgets and candidate sets differed in the pilot; comparisons require controls.

## 3. Focused paper comparison

| Source | Key verified contribution | Matched scanner data? | Graph topology / pooling? | Direct relevance and remaining question |
|---|---|---|---|---|
| **Ochi et al., 2024, PLISM** [1] | Introduces a registered multi-device and staining histology dataset, including WSI scanner alignment. | **Yes** | Dataset paper, not a scanner-paired graph-topology benchmark. | Foundation for matched tissue experiments; registration quality and tile-coordinate semantics should still be checked for the particular extended tile release. |
| **Owkin PLISM tiles / PLISM benchmark** [2,3] | Provides registered tiles and robustness benchmarking for pathology encoders under scanner/stain changes. | **Yes** | Reported emphasis is feature-model robustness rather than patch-graph edge agreement. | Reuse benchmark matching and reporting conventions where applicable; distinguish graph stability from existing embedding robustness metrics. |
| **SPECTRA, Kim et al., 2026 preprint** [4] | Contrastive post-training to reduce stain/scanner bias in pathology foundation-model embeddings; evaluates robustness and downstream tasks, including held-out PLISM retrieval. | **Yes, PLISM used** | Graph adjacency/coarsening comparisons were **not identified in the abstract/search materials reviewed**; full methods must be checked. | Direct competitor for feature stability and feature-alignment baselines, not necessarily for graph-stability measurement. |
| **FEATMAP, 2026 bioRxiv preprint** [5] | Harmonizes scanner-related feature signatures using large aligned multi-scanner embedding sets; evaluates on PLISM. | **Yes** | Graph-edge/coarsening analysis **not established from materials reviewed**. | Important control: test whether feature harmonization also stabilizes graph adjacency. |
| **Reliability of foundation models for image retrieval in histopathology, 2026** [6] | Evaluates histopathology feature retrieval using spatially co-registered tissue scanned by different scanners. | **Yes** | Retrieval-neighbor consistency is related to kNN graph stability; exact graph evaluation must be checked in the full text. | Highest-priority check before calling cross-scanner neighborhood instability novel. |
| **HiGATE, Dad et al., 2026** [7] | Spatially constrained hierarchical graph construction, differentiable graph pooling, and cross-level attention. | No matched cross-scanner graph stability study established in reviewed summary. | **Yes** | Spatial constraints and pooling themselves are not novel; test their *paired-scanner consistency* explicitly. |
| **StructMIL, 2026** [8] | Structure-aware graph priors, prototype-enhanced pooling, and domain-generalization objectives for histopathology. | Evaluates cross-scanner/domain robustness, but matched physical-node correspondence is not established in reviewed materials. | **Yes, graph features and pooling** | Strong overlap with robustness-plus-graph modeling; avoid any claim that combining graph structure and scanner robustness is unprecedented. |

### Evidence standard for the table

“Not established” means the materials examined do not verify that property. It does **not** mean the authors omitted it. A full-paper methods review, references/citation search, and supplementary-material check are needed before asserting novelty.

## 4. Preliminary overlap assessment

| Proposed contribution | Assessment | Action |
|---|---|---|
| Scanner shifts affect histopathology embeddings | **Already studied** | Treat as motivation/baseline, not primary novelty. |
| Scanner-paired image retrieval / nearest-neighbor changes | **Strong existing overlap** | Read [6], [4], [5] completely; compare exact definitions. |
| Spatially constrained histopathology graphs | **Already studied** | Cite HiGATE and related graph literature. |
| Graph-structured domain generalization | **Already studied** | Cite StructMIL and related work. |
| Registered, same-node **edge-set agreement** across scanner conditions | **Candidate gap, not verified** | Search for paired-scanner graph-adjacency robustness metrics. |
| Stability of **learned pooling/coarsening assignments** for matched nodes | **Candidate gap, not verified** | Inspect hierarchical histopathology GNN/domain-generalization studies closely. |
| Predictive utility of stable graphs on a labeled external test set | **Unproven within this project** | Requires a labeled benchmark and split protocol, separate from PLISM pilot. |

## 5. Most defensible research direction at present

**Provisional focus:** A reproducible, paired-scanner graph-structure robustness evaluation that compares node-aligned edge agreement, spatial constraints under controlled edge budgets, and (if feasible) coarsening-assignment consistency. Test whether increased structural agreement predicts representation or prediction consistency, rather than assuming it does.

Avoid the premature claims “first scanner-robust graph method” or “first evaluation of scanner-induced nearest-neighbor drift.” The literature already addresses closely related questions.

## 6. Required follow-up reading checklist

For each of [4]–[8], record from the **full paper and supplement**:

1. Exact datasets and whether the **same physical locations** are paired across scanners.
2. Models/encoders and whether they are frozen or fine-tuned.
3. Graph definition, whether a graph is actually constructed, and whether edges are based on features, coordinates, or both.
4. Whether edge/neighborhood agreement is computed **with node correspondence preserved**.
5. Whether pooling assignments are compared on matched scanner pairs (including permutation/cluster-label alignment).
6. What independence unit is used for uncertainty (slide, specimen, region, or patch).
7. Whether external classification, stain-normalization, and feature-harmonization baselines are included.
8. What code and datasets are publicly reusable.

Only after this audit should the claim of novelty be finalized.

## 7. Handoff to Phase 2, Step 2.3: Dataset and sampling protocol

Design a preregistered-style sampling protocol, **without downloading new data yet**:

- **Sampling unit:** independent parent slide/specimen; select multiple spatially separated regions *within* each. Never use adjacent patches as independent statistical replicates.
- **Scanner pairs:** same tissue and staining condition, with a known shared reference frame; record device metadata and registration procedure.
- **Layout validation:** verify `tile_id` row/column order against dataset documentation and visual seams; inspect registration error quantitatively where possible.
- **Pilot versus confirmatory samples:** freeze the existing 90-location region as exploratory and exclude it from any future held-out confirmatory test set.
- **Primary outcome:** graph edge Jaccard on matched nodes. **Secondary:** node-neighbor Jaccard, edge turnover, degree agreement, components, neighborhood rank agreement.
- **Controls:** spatial-only (expected ceiling of 1 by fixed construction), feature kNN, spatially constrained feature kNN; report symmetric union rules and compare compatible edge budgets/densities.
- **Robustness checks:** vary k, spatial radius, graph symmetrization, encoders, tissue regions, and scanners. Avoid selecting parameters after inspecting held-out results.
- **Inference:** paired, slide-clustered bootstrap confidence intervals or another justified slide-level method; report number of independently sampled slides.
- **Coarsening extension:** define matched-node, permutation-invariant assignment metrics before training pooling models.
- **Downstream extension:** only where appropriate labels and a leakage-resistant slide/patient split exist.

**Decision gate:** Do not start large-scale downloading or GNN development until literature overlap and the data sampling plan are reviewed.

## References and original links

1. Ochi et al. (2024). *Registered multi-device/staining histology image dataset for domain-agnostic machine learning models.* Scientific Data. https://doi.org/10.1038/s41597-024-03122-5
2. Owkin, *PLISM dataset tiles* (dataset card). https://huggingface.co/datasets/owkin/plism-dataset-tiles
3. Owkin, *Official PLISM robustness benchmark*. https://github.com/owkin/plism-benchmark
4. Kim et al. (2026). *Contrastive fine-tuning against stain and scanner bias improves robustness and downstream performance across pathology foundation models* (SPECTRA; preprint). https://arxiv.org/abs/2611.10093
5. *FEATMAP: Targeted Correction of Acquisition Signatures Harmonizes Medical Foundation Model Embeddings and Enables Robust Task Generalization* (2026; preprint). https://www.biorxiv.org/content/10.64898/2026.07.02.736184v1.full
6. *Reliability of foundation models for image retrieval in histopathology* (2026). https://www.nature.com/articles/s44303-026-00174-7
7. Dad, He, and Shen (2026). *HiGATE: hierarchical graph attention for multi-scale tissue encoder in computational pathology.* Frontiers in Oncology. https://doi.org/10.3389/fonc.2026.1798040
8. *Structure-aware generalization for heterogeneous histopathology via prototype-based multiple instance learning* (2026; StructMIL). https://www.nature.com/articles/s41746-025-02289-4

---

*Scope note: This document summarizes verified abstracts, dataset documentation, and available full-text descriptions as of the check date. A complete systematic review has not been performed.*
