# Phase 2, Step 2.3: Experimental Dataset and Sampling Protocol

**Project:** Cross-Scanner Graph Topology Stability in Computational Histopathology  
**Status:** Proposed protocol, pending metadata audit and feasibility review  
**Version:** 0.1 (design only; no additional data downloaded)

## 1. Purpose

Build a reproducible, leakage-resistant set of matched histopathology tissue regions for evaluating scanner-dependent embedding drift and graph-topology stability. This document defines the *sampling plan*, not an assertion that every proposed sample is available in PLISM.

The Phase 1 pilot used 90 matching patch locations from `GIVH_AT2_to_GMH_S60` and `GIVH_GT450_to_GMH_S60`. It is retained for method development and excluded from independent final testing.

## 2. Dataset and observation hierarchy

**Dataset:** PLISM (`owkin/plism-dataset-tiles`, Hugging Face), with registered or matched tile identifiers across scan conditions. The local dataset resides under `~/Medical_GNN_Research/datasets/`, which must remain ignored by Git.

Maintain the following distinction:

| Level | Meaning | Experimental role |
|---|---|---|
| Image tile | One 224 × 224 scanner-specific image | Graph node / paired observation |
| Matched tile location | Same tile ID in two verified scan groups | Paired node across scanners |
| Spatial region | Set of adjacent, tissue-bearing matched tiles | One candidate graph observation |
| Original specimen / slide identity | Underlying tissue source, if verified from dataset provenance | Preferred unit for data splitting and statistical independence |
| Scanner/stainer condition | Acquisition configuration | Domain factor / comparison condition |

**Critical verification:** A scanner-specific `slide_id` is not automatically an independent physical specimen. Before splitting, validate how PLISM encodes original specimen identity, slide groups, scanner names, stainer names, and coordinate axes. Tile IDs that resemble `tile_16_row_col` provide provisional grid indices; physical adjacency and orientation need further checking against mosaics and dataset documentation.

## 3. Scope of the first controlled experiment

- **Primary comparison:** AT2 versus GT450 under a fixed GIVH staining condition, using verified matched specimen and tile identifiers.
- **Initial encoder:** frozen DINOv2-Small to reproduce the pilot.
- **Later encoder check:** at least one accessible pathology-specialized encoder, subject to model license/access and compute feasibility.
- **Primary outcome:** paired cross-scanner edge Jaccard and node-neighborhood Jaccard.
- **Additional outcomes:** paired embedding similarity, number of edges, degree distributions, connected components, and sensitivity to graph parameters.
- **Not in initial scope:** clinical classification performance or claims about pooling/coarsening. Those require separate datasets/models/protocols.

## 4. Metadata inventory before download (mandatory gate)

Create an index from metadata alone. Do not download all 293 Parquet shards or the complete tile dataset to build the index. Stream candidate shards with bounded memory and log provenance; adjust shard selection after verifying group locations. Record:

- `source_shard` and remote dataset revision/commit identifier
- raw `slide_id`, `tile_id`, `scanner`, `stainer`, and any available physical specimen identifier
- parsed tile-level and grid coordinates (with a parse-validity flag)
- scanner-pair linkage and confirmed stainer equivalence
- per-group counts and coordinate ranges
- candidate region availability, overlap, and number of unique original specimens

**Gate A:** Proceed only if original specimen/group linkage and pair correspondence are adequately documented. If multiple candidate regions belong to one specimen, keep them in the same split and treat them as clustered measurements.

## 5. Proposed sampling targets (planning targets, not available-data claims)

Use an explicit staged plan, governed by the metadata inventory:

| Stage | Intended sample | Role |
|---|---|---|
| Historical pilot | Existing 90-node region | Debugging and method development only |
| Feasibility extension | 3–5 *independent* specimens if available, one region/specimen initially | Check coverage, grouping, and preprocessing |
| Main evaluation target | 20–30 independent specimens/physical slide groups if available | Estimate between-specimen variation |
| Per-specimen region | At least one connected region, target 80–150 matched tissue tiles | One matched graph per scanner |
| Robustness extension | Multiple separated regions per specimen | Assess within-specimen variability without pretending independence |
| Scanner generalization | Additional verified same-stain scanner pairs | Test reproducibility beyond AT2/GT450 |

These are preliminary targets, **not a power calculation**. After the inventory, revise the target based on the number of genuinely independent specimens and expected confidence-interval precision. If 20–30 independent specimens cannot be obtained, report an exploratory study with appropriately limited generalization claims.

## 6. Region selection algorithm

1. Group records by verified original specimen identity and acquisition condition, not scanner-specific `slide_id` alone.
2. Identify tile IDs present in *both* scan groups within each specimen.
3. Parse and validate grid coordinates; reject malformed IDs or ambiguous coordinate systems.
4. Build a coordinate-adjacency graph on the intersection of paired IDs (8-neighbor connectivity as an initial option).
5. Identify connected components and generate candidate windows of 80–150 nodes (or an explicitly justified alternate range).
6. Evaluate tissue-content heuristics on **both** scans, and use a scanner-symmetric inclusion rule. Brightness-only thresholds are unsuitable as the sole tissue criterion because our pilot found major scanner brightness differences.
7. Select candidates using a frozen random seed (e.g., `42`) after applying predeclared quality criteria, avoiding overlapping windows as separate independent samples.
8. Save selection and rejection reasons for each candidate. Do not choose samples based on graph stability outcomes.

**Gate B:** Confirm each region is contiguous by the validated coordinate rule and visually inspect a small blinded/random subset before finalizing inclusion criteria.

## 7. Splits and leakage prevention

**Preferred splitting key:** independently verified physical specimen ID. When available, patient/source-group identity takes precedence over slide and region. Never allow adjacent tiles, overlapping spatial windows, repeated scans, or multiple regions from the same specimen to appear across different splits.

Proposed division of independent specimen groups:

- **Development:** 60% for graph design and parameter tuning.
- **Validation:** 20% for selecting graph settings.
- **Locked test:** 20% for one final evaluation after choices are frozen.

Ratios are provisional; with few independent specimens, favor grouped cross-validation and an explicitly defined untouched test set if feasible. Use grouped, seeded splits and store IDs in CSV files. Do **not** split tile rows randomly.

**Phase 1 pilot:** Explicitly label as development-only. Exclude its physical source specimen and any overlapping registered slide groups from the test set, once specimen identity is confirmed.

## 8. Graph-construction comparison controls

Construct both scanner graphs from the same ordered node IDs, graph population, and coordinate system. Fix all preprocessing choices and encoder weights before comparison.

- **Spatial-only control:** Identical coordinate-based adjacency; Jaccard = 1 by construction.
- **Unrestricted feature-kNN:** Fixed `k`, same similarity metric, same graph symmetrization rule.
- **Spatially constrained feature graph:** Rank feature similarity among spatial candidates; report the candidate radius and achievable node degree.
- **Matched-budget analysis:** Compare versions with controlled average edge count/density or degree, plus sensitivity sweeps. Merely using a nominal `k` does not ensure equal undirected edge counts.

Pre-register a small parameter grid, for example `k ∈ {3,5,8}` and spatial radii `r ∈ {1, √2, 2, 3}` in tile-grid units, where feasible. Keep the final test set untouched during tuning. Report graph construction tie-breaking and symmetrization policy (`union` or `mutual`), which affect edge counts.

## 9. Evaluation and statistical unit

| Outcome | Unit and summary |
|---|---|
| Matched-node feature cosine | Per tile; summarize within each region/specimen |
| Edge Jaccard | Per paired region graph |
| Mean node-neighborhood Jaccard | Per paired region graph |
| Shared / discordant edges | Per paired region graph |
| Components, mean degree, edge density | Per scanner graph; paired differences by region |
| Hyperparameter sensitivity | Across graph settings, summarized by specimen |

**Primary comparison:** feature-kNN versus spatially constrained feature graphs, paired within the same region/specimen, under explicitly controlled construction budgets. Summarize effects at the independent **specimen level**, not as thousands of independent tile observations.

Report median/mean and dispersion, paired effect sizes, and uncertainty intervals. Use specimen-level bootstrap confidence intervals or a justified paired alternative. If multiple regions per specimen are included, aggregate by specimen or use cluster-aware hierarchical analysis. Avoid treating the known spatial-only Jaccard of 1 as an empirical success benchmark for classification.

## 10. Robustness and negative controls

- Repeat with another pretrained encoder when available.
- Check stability across `k`, spatial radius, graph symmetrization, and region sizes.
- Check whether tissue content, brightness, feature norms, and registration quality relate to edge disagreement.
- Include an identity/self-pair control (same image embeddings paired with themselves), which should yield Jaccard = 1 when all graph-building operations are deterministic.
- Optionally use same-scanner repeat runs to verify deterministic feature extraction.
- Do not claim causality for scanner effects unless specimen, stain, registration, and preprocessing confounders are sufficiently controlled.

## 11. Download and storage policy

- Start with metadata and specific shard indices, then download *only* approved matched tiles.
- Resume from locally validated images; verify integrity, dimensions, and scanner/slide/tile IDs.
- Keep all original tiles under `datasets/` and excluded from Git with `/datasets/` in repository `.gitignore`.
- Track scripts, manifests, metadata (as licensing permits), summaries, figures, reports, environment requirements, and repository/revision provenance.
- Avoid committing pretrained model weights, raw PNG tiles, `.npz` embeddings, or secrets.
- Follow dataset and model licensing and attribution requirements before redistribution.

## 12. Planned deliverables

```text
research_design/
├── Docs/
│   └── 01_research_questions_and_hypotheses.md
├── literature_review/
│   ├── 02_literature_review_and_novelty_verification.md
│   └── 03_focused_literature_comparison.md
├── protocols/
│   └── 04_experimental_dataset_sampling_protocol.md  # this file
└── README.md
```

Following acceptance of this protocol, the implementation phase should produce:

- `specimen_acquisition_index.csv`
- `candidate_region_inventory.csv`
- `sample_selection_log.csv`
- `grouped_split_manifest.csv`
- `matched_region_manifest.csv`
- `dataset_provenance.json`
- A short audit report explaining any changed targets and why

