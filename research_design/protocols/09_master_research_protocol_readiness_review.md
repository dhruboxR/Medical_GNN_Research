# Phase 2 Master Research Protocol and Readiness Review

**Project:** Cross-Scanner Graph Topology Stability in Computational Histopathology  
**Document:** Step 2.8, master protocol  
**Status:** Design package consolidated; **conditional readiness** for engineering and feasibility checks, **not approved** for confirmatory experiments  
**Prepared:** 2026-11-23  
**Pilot:** PLISM GIVH, AT2 versus GT450, 90 registered tile-ID locations

## 1. Executive decision

Phase 2 research-design documentation is substantially complete. The project may advance to **Phase 3A: Feasibility and implementation validation**, subject to the readiness gates in Section 12. Do **not** start confirmatory cross-specimen evaluation, claim methodological novelty, or train a new clinical GNN merely because this document exists. The primary experimental configuration and independent specimen availability remain to be validated and frozen.

## 2. Provenance and upstream protocols

This master protocol consolidates the decisions recorded in the project’s research questions and the following authored Phase 2 documents. Retain the originals and treat their more detailed specifications as supporting references:

| Step | File and role |
|---|---|
| 2.1 | `research_design/Docs/01_research_questions_and_hypotheses.md` — aims, scope, hypotheses |
| 2.2A | `research_design/literature_review/02_literature_review_and_novelty_verification.md` — preliminary related-work assessment |
| 2.2B | `research_design/literature_review/03_focused_literature_comparison.md` — focused overlap and gap review |
| 2.3 | `research_design/protocols/04_experimental_dataset_sampling_protocol.md` — data sourcing, specimen grouping, leakage control |
| 2.4 | `research_design/protocols/05_controlled_graph_construction_baseline_protocol.md` — graph methods, matched edge budgets |
| 2.5 | `research_design/protocols/06_feature_extractor_selection_comparison_protocol.md` — encoder candidates and frozen extraction |
| 2.6 | `research_design/protocols/07_evaluation_methodology_statistical_analysis_plan.md` — endpoints and statistics |
| 2.7 | `research_design/protocols/08_graph_coarsening_downstream_evaluation_protocol.md` — hierarchical and downstream experiments |

This synthesis uses previously recorded project results and protocol decisions; it does **not** certify the novelty or current availability of external datasets/models.

## 3. Research aim and testable questions

**Aim:** Determine how scanner-associated variation in patch embeddings affects the topology of graphs built from corresponding histopathology regions, and whether spatially constrained graph construction produces more consistent topology under fair, fixed edge budgets.

- **RQ1, representation:** How stable are registered patch embeddings across scanners, and how does stability vary by encoder and specimen?
- **RQ2, topology:** How stable are feature-derived graph edges and node neighborhoods across scanners?
- **RQ3, control:** Do spatially constrained feature graphs exhibit greater stability than unrestricted feature graphs at the **same number of nodes and edges**, beyond stability attributable to a smaller eligible-edge universe?
- **RQ4, optional extension:** Do graph differences affect coarsening assignments, pooled topology, frozen graph representations, or predictions?

**Primary directional hypothesis, provisional:** The specimen-level paired difference in edge Jaccard, `J(spatial-constrained) − J(unrestricted)`, is positive at a preregistered jointly feasible edge budget. Because eligibility restrictions can trivially increase overlap, a null-adjusted or candidate-universe-aware interpretation is required.

## 4. Pilot evidence (development only)

| Metric | Observed value |
|---|---:|
| Matched tile-ID locations | 90 |
| Valid paired images | 180 |
| DINOv2-Small feature vectors | 90 × 384 per scanner |
| Mean paired cosine similarity | 0.973113 |
| Mean paired pixel MAE (0–255) | 47.6779 |
| Spatial-only edges: AT2 / GT450 | 306 / 306 |
| Spatial-only edge Jaccard | 1.000000 |
| Feature-kNN edges: AT2 / GT450 | 328 / 340 |
| Feature-kNN edge Jaccard | 0.357724 |
| Hybrid edges: AT2 / GT450 | 231 / 238 |
| Hybrid edge Jaccard | 0.796935 |

**Interpretation:** Paired feature vectors were similar while selected feature-kNN edges differed. The spatial-only graph is identical by construction. The original feature-kNN and hybrid methods were **not** matched for edge count or candidate sets; their Jaccard values are descriptive, not a clean causal comparison. One tissue region is insufficient for specimen-level inference. The apparent 10 × 10 grid is inferred from tile IDs; physical coordinate semantics and registration quality still require verification.

## 5. Dataset, sampling, and isolation

**Source:** PLISM registered scanner-paired tiles. The initial study holds staining condition fixed at GIVH and compares AT2 with GT450. Additional acquisitions and stains are extensions only after feasible metadata coverage is established.

1. Inventory PLISM metadata **without bulk image download**. Confirm specimen/slide identifiers, scanner and stainer combinations, registered location keys, tile coordinate semantics, and shard coverage.
2. Identify independently sampled source specimens; where available, sample several separated tissue regions per specimen but retain specimen as the inferential cluster.
3. Prespecify region size, tissue-content QC, correspondence checks, and minimum viable connected spatial coverage. QC must not favor one scanner's brightness or appearance.
4. Keep paired acquisitions of a physical region and all regions of the same underlying specimen **in one split**. If specimen independence cannot be established, label the work exploratory and do not claim held-out generalization.
5. Pilot region is **development-only**. A provisional 60/20/20 group split may be used when feasible; exact counts and the held-out partition must be frozen before tuning.
6. Candidate target of 20–30 independent specimens and about 80–150 paired tiles/region is **aspirational**, pending an availability audit and a statistical precision/power justification.

**Excluded:** unverified image correspondences, mislabeled scanner/stainer conditions, inconsistent node sets, empty/failed image tiles, or physically ambiguous coordinates under a documented exclusion rule. Store all exclusions and reasons.

## 6. Feature-extraction protocol

- **Primary development encoder:** frozen `facebook/dinov2-small`, CLS embedding, 384 dimensions. Pin exact model revision, processing config, package versions, and hardware.
- **Priority pathology-specific comparator:** Phikon-v2, only after checking model access, licensing, input transformations, and runtime feasibility.
- **Optional:** UNI2-h and CONCH if model access, licensing, and GPU requirements permit. No access is presumed.
- Apply each model's documented deterministic inference preprocessing, no data augmentation at evaluation, same image IDs and node order across scanner conditions, no fine-tuning in the primary topology comparison.
- Use L2-normalized embeddings for cosine-based graph ranking. Save original embeddings, normalized embeddings or provenance for reconstruction, image-to-feature map, error flags, hashes, and inference settings locally.
- Compare encoders **within their own feature space**. Do not directly compare uncalibrated Euclidean distances across different embedding dimensions or feature scales.

## 7. Graph baselines and fair comparator

All graphs use the same `N` matched physical locations and common deterministic node ordering. Edges are undirected, unique, with no self-loops; graph construction is deterministic at fixed inputs.

| ID | Graph | Purpose |
|---|---|---|
| B0 | Fixed spatial adjacency (8-neighbor after geometry validation) | Identity/control check; Jaccard must be 1 on paired coordinates |
| B1 | Global cosine feature-kNN, symmetrized | Descriptive pilot baseline; realized edge counts may differ |
| B2 | Spatial-restricted feature-kNN, symmetrized | Descriptive pilot hybrid baseline |
| B3 | Global top-`M` unique edges by feature cosine | **Primary unrestricted comparator** |
| B4 | Spatially eligible top-`M` unique edges by feature cosine | **Primary constrained comparator** |
| N0 | Random `M` edges from respective eligible candidate universe | Candidate-size-adjusted null benchmark |

**Primary contrast:** `B4 − B3` in edge Jaccard on paired scanner graphs, using identical `N`, identical integer `M`, fixed cosine metric, and fixed deterministic tie-breaking. A setting is eligible only if both methods can select `M` edges for **both** scanners. Never silently adjust `M` for a single method or scanner.

**Prespecified development sweeps:** feature-kNN `k ∈ {3,5,8,10}` (secondary); spatial radius `r ∈ {1,2,3}` grid steps after coordinate verification; fixed eligible density targets `5%, 10%, 15%` converted by one documented rounding convention. These are **candidate ranges**, not finalized primary settings. Freeze exactly one primary encoder, radius, and edge-budget selection before testing; report all sensitivity results without selecting a best test configuration.

**Critical caution:** Restricting possible edges can yield artificially high overlap without preserving tissue-relevant relationships. Always record eligible candidate counts and compare against matching random-null constructions. Report connectivity and isolated nodes; do not force connectivity without recording and controlling that operation.

## 8. Outcomes and statistical plan

**Primary outcome per independent specimen:** paired difference `D = J_B4 − J_B3`, where Jaccard compares the AT2 versus GT450 undirected edge sets for a given method. If multiple regions are drawn from a specimen, aggregate to a predefined specimen-level summary or use specimen-clustered inference.

**Secondary:** node-neighborhood Jaccard, shared-edge fraction (`|A∩B|/M`), edge counts, candidate universe size, degree distribution, components, isolates, largest-component share, paired feature cosine, and stability versus edge budget.

**Analysis:** Report specimen-level effect estimates and 95% uncertainty intervals, with specimen-clustered bootstrap when sample size permits. Use an appropriate paired test for a prespecified comparison, with caveats for small sample size. Correct or explicitly distinguish exploratory multiplicity from confirmatory inference. Avoid treating patches, nodes, edges, or regions from one specimen as independent biological replicates.

**Negative controls:** identical-graph identity check; deterministic same-input rerun; independently sampled matched-budget random graphs within each eligible universe; registration/coordinate sensitivity analyses. Verify graph implementations against tiny synthetic graphs with known Jaccard values.

**Interpretation rule:** Stability is a reproducibility property, **not** a proxy for semantic accuracy, diagnosis, or clinical utility. Report null-adjusted effects and representational quality checks before advocating a constrained approach.

## 9. Coarsening and downstream extension (separate stage)

Only pursue after matched-budget topology analysis is technically validated and an independent data sample is available.

- **Baselines:** no pooling, fixed spatial partition, feature-based clustering; optional learned DiffPool and MinCutPool with fixed training/initialization protocols and comparable pooling ratios.
- **Assignment agreement:** Adjusted Rand Index (ARI), Adjusted Mutual Information (AMI) on the same registered node set; account for arbitrary cluster labels.
- **Pooled topology:** align clusters meaningfully before edge comparison, or use permutation-invariant structural metrics if alignments are not trustworthy. Report number of clusters, occupancy and collapsed/degenerate assignments.
- **Factorial diagnostics:** `{AT2 features, AT2 edges}`, `{GT450 features, GT450 edges}`, `{GT450 features, AT2 edges}`, `{AT2 features, GT450 edges}` in an otherwise identical frozen model to separate sensitivity to node attributes and edge sets.
- **Representation consistency:** paired graph-embedding similarity under the same frozen graph network, with fixed weights and normalization. No claim of diagnostic improvement without an independent labeled task, suitable ground truth and held-out evaluation.

This extension remains **conditional** and is not a gate for beginning core Phase 3A engineering validation.

## 10. Planned implementation sequence

**Phase 3A: Feasibility and engineering validation**

1. Build lightweight metadata inventory across selected PLISM shards; validate independent specimen groups and scanner-pair availability.
2. Validate tile-ID coordinates against source documentation and visual adjacency; audit registered correspondence and scanner-neutral tissue masks.
3. Implement B3/B4 fixed-`M` graph builders, spatial eligibility checks, and null controls with unit tests on synthetic graphs.
4. Run these builders first on the **existing 90-node pilot**, explicitly labeling results development-only.
5. Define and freeze eligible independent specimen regions, preregister the comparison and splits before broader download.

**Phase 3B: Multi-specimen exploratory experiment**

6. Download only verified matched regions, hash/store a local manifest, and extract frozen embeddings.
7. Assess feasible sample size, region variability, registration QC, graph budget feasibility, and null-corrected stability; document deviations.
8. Lock primary configuration and held-out specimen partition only after predeclared development criteria are satisfied.

**Phase 3C: Confirmatory test (conditional)**

9. Evaluate the frozen endpoint once on held-out independent specimens. Report uncertainty, negative findings, exclusions, and any protocol deviations.
10. Consider pooling/downstream studies as a separately specified experiment, not as evidence guaranteed by topology stability.

## 11. Reproducibility artifacts

Keep in Git: Python source scripts, pinned dependency manifest/lockfile where feasible, README, methodology documents, figures, small result tables, metadata manifests **without secrets**, configuration YAML/JSON, saved random seeds and source revision identifiers.

Keep locally or under approved secure data storage: `datasets/`, original PLISM PNGs, downloaded pretrained weights and large feature arrays. Respect the repository-root `.gitignore` rule `/datasets/`. Raw data must not be committed accidentally.

Proposed Phase 3 layout:

```text
Medical_GNN_Research/
├── datasets/                                  # local, gitignored
├── dataset_exploration/                      # Phase 1, completed
├── research_design/
│   ├── Docs/
│   ├── literature_review/
│   └── protocols/
│       └── 09_master_research_protocol_readiness_review.md
└── experimental_research/                    # Phase 3; create when starting
    ├── configs/
    ├── scripts/
    ├── tests/
    ├── outputs/
    │   ├── tables/
    │   ├── figures/
    │   └── reports/
    └── README.md
```

## 12. Readiness review and decision gates

| Gate | State at Phase 2 close | Required evidence/action |
|---|---|---|
| Research question and descriptive pilot | **Complete** | Phase 1 output plus RQ document |
| Targeted literature gap assessment | **Provisional** | Full-text review of closest studies; update novelty statement |
| Source metadata supports independent specimens | **Unverified** | Specimen/slide IDs and counts verified |
| Correct scanner/stainer pairing | **Pilot verified; broader inventory pending** | Matched acquisition metadata for each sampled region |
| Tile coordinate semantics and fine registration | **Partial** | Documentation and quantitative QC protocol |
| Scanner-neutral region QC | **Specified, not validated** | Test on multiple acquisition pairs |
| Fixed-budget graph builders and nulls | **Specified, not implemented** | Synthetic tests and pilot reproducibility |
| Primary encoder, `M`, and spatial radius | **Not frozen** | Select on development groups; record versioned config |
| Specimen-disjoint test partition | **Not established** | Independent specimen inventory and frozen split manifest |
| Specimen-level statistical analysis | **Specified, not executed** | Validate analysis code and effective sample size |
| Coarsening/downstream task | **Optional future extension** | Separate feasible protocol and data justification |

**Readiness decision:** **GO** for Phase 3A feasibility and engineering validation. **NO-GO** for confirmatory experiments or broad novelty claims until the unresolved gates pass.

## 13. Research claims: allowed and disallowed

**Allowed today:** “In a 90-location PLISM pilot, paired DINOv2-Small features had mean cosine similarity 0.9731, while feature-kNN graph edge Jaccard across AT2 and GT450 was 0.3577. A spatially restricted hybrid achieved 0.7969 under a different edge budget and candidate set.”

**Not allowed today:** “The new hybrid graph is generally scanner-robust”; “scanner variation caused classification failure”; “coarsening is unstable”; “our approach improves diagnostic accuracy”; “we are the first to study scanner-robust histopathology GNNs.” None is established by the current data.

## 14. Phase decision and change control

Phase 2 is **documentation-complete** when this file is saved and reviewed. Phase 3A may begin immediately as a validation stage. Maintain a `protocol_changes.md` log identifying changes to sampler, scanner selection, encoder, graph controls, statistical testing, or exclusion rules. Once the primary confirmatory protocol is frozen, treat material deviations as amendments and report them explicitly.


