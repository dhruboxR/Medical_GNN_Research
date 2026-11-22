# Phase 2, Step 2.6: Evaluation Methodology and Statistical Analysis Plan

**Project:** Cross-Scanner Graph Topology Stability in Computational Histopathology  
**Status:** Preregisterable draft; finalize after metadata feasibility checks  
**Research workspace:** `~/Medical_GNN_Research/research_design/protocols/`  
**Version:** 1.0 | 2026-11-23

## 1. Purpose and boundaries

This protocol specifies **what is measured, at what unit of replication, how methods are compared, and how uncertainty is reported**. It extends Phase 1's 90-location PLISM pilot; it is **not** evidence that graph stability improves clinical classification. The pilot region and its results are designated exploratory/development-only.

Primary target: paired registered tissue regions scanned under AT2 and GT450 with matching staining condition and node identities. Confirm physical specimen identity, registration quality, and independence using official PLISM metadata before full-scale analysis. Tile-ID-derived coordinates are provisional until verified.

## 2. Questions and analysis priorities

| Priority | Question | Main estimand | Status |
|---|---|---|---|
| Primary | Does adding a spatial candidate restriction change scanner-to-scanner edge agreement at a **matched edge budget**? | Within-region difference in edge Jaccard (spatial-feature minus feature-only) | Confirmatory **only after** methods, sampling, and hypotheses are frozen |
| Secondary | How stable are patch embeddings across scanners? | Within-region mean matched-patch cosine similarity | Descriptive |
| Secondary | Are node neighborhoods preserved? | Mean per-node neighbor Jaccard, aggregated per region | Descriptive / paired |
| Secondary | Is instability sensitive to encoder or graph budget? | Stability curves across prespecified encoder and density settings | Sensitivity |
| Exploratory | Are pooling/coarsening assignments or graph embeddings stable? | Matched assignment agreement / pooled representation similarity | Deferred to Step 2.7 |
| Out of scope initially | Does higher topology stability improve disease classification? | Labeled downstream task performance | Requires separate labeled dataset and held-out evaluation |

**Primary comparator:** global feature-similarity graph vs spatially restricted feature-similarity graph, both using the same node set, encoder features, and fixed **M undirected edges** within each region and scanner. The fixed-coordinate spatial graph is a deterministic control, not an eligible winner in the primary claim.

## 3. Definitions and graph alignment

For each region `r`, scanner `s`, encoder `e`, and graph method `m`, construct an undirected, simple edge set `E(r,s,e,m)` over the same registered patch IDs. Remove self-loops; canonicalize each edge as `(min(node_i,node_j), max(node_i,node_j))`; fix tie-breaking by stable node ID; document feature normalization and candidate selection.

- **Primary edge stability:** `J = |E_AT2 ∩ E_GT450| / |E_AT2 ∪ E_GT450|`; define J = 1 if both sets are empty, but exclude degenerate empty graphs from the planned primary comparison.
- **Per-node neighbor stability:** `J_i = |N_AT2(i) ∩ N_GT450(i)| / |N_AT2(i) ∪ N_GT450(i)|`; explicitly report handling for nodes isolated in both graphs (default J_i = 1; sensitivity analysis excludes them).
- **Edge disagreement:** symmetric-difference count `|E_AT2 △ E_GT450|`, plus rate relative to edge union; do not equate with error without a ground-truth graph.
- **Structural descriptors:** edge count, density `2|E|/[n(n−1)]`, degree distribution, isolated nodes, connected components, and largest-component fraction.
- **Feature descriptors:** cosine similarity between embeddings of the same registered location, paired Euclidean distance (within encoder only), cross-scanner local-neighborhood overlap.

All metrics must retain the region/scanner/encoder/parameter identifiers. Compare methods **at identical numbers of nodes and edges**. If the spatial candidate pool contains fewer than M feasible edges, mark the region-setting as infeasible; do not silently lower M for only one method. Report feasibility frequency.

## 4. Primary endpoint and effect

For each independent region `r` (or predeclared specimen-level aggregation if regions share a specimen), calculate

`D_r = J_r(spatially restricted, M) − J_r(global feature-only, M)`.

The **primary reported effect** is the mean of `D_r` across independent specimens, with a 95% confidence interval. Positive `D` means greater edge agreement for spatially restricted graphs at the chosen settings; it does not imply superior diagnostic utility.

**One primary configuration must be frozen in advance:** encoder, tile filter, graph size, candidate spatial radius and edge budget or target density, tie rules, and population of eligible specimen groups. Suggested development choices should not be treated as finalized after inspecting held-out outcomes.

## 5. Independent experimental unit and leakage prevention

- **Patch:** matched node; not an independent statistical replicate.
- **Region:** graph-level measurement; regions from the same slide/specimen are correlated.
- **Specimen/slide group:** preferred independent unit if independently sampled; verify slide-to-patient or specimen provenance before treating different slide IDs as independent.
- **Primary inference:** one aggregate paired effect per independent specimen (e.g., average region-level D within that specimen). Equal-specimen weighting is default; record alternative weighting as sensitivity analysis.
- **Splits:** reserve original tissue specimen groups, keeping all scanner versions and all extracted regions from one group in the same split. Keep Phase 1's 90 matched locations in development only.
- **No leakage:** do not tune encoder, graph settings, pooling, thresholds, or stopping choices using held-out test results.

## 6. Sample size and feasibility

Phase 1's 90 locations form **one region**, not 90 independent observations. A previous exploratory target of 20–30 independent specimens is only a planning placeholder, **not a power calculation**. Verify the number of independent PLISM specimen groups and usable registered regions first. Once a development set provides between-specimen variance for D, estimate precision/power for a prespecified minimum meaningful effect; use scenario calculations and explicitly report limitations if too few clusters exist. Never manufacture statistical significance by treating patches or multiple graph settings as independent replicates.

## 7. Confidence intervals and inference

**Primary analysis (after sufficient independent specimen groups exist):**

1. Compute each specimen's paired difference `D_g`, aggregating its regions as prespecified.
2. Report `G` (number of independent specimens), mean/median `D_g`, standard deviation, and individual paired effects.
3. Obtain a **specimen-level paired bootstrap** 95% confidence interval for mean D by resampling specimens with replacement while carrying both methods and all within-specimen observations together (e.g., 5,000–10,000 replicates; fixed RNG seed). If G is small, acknowledge unreliable bootstrap coverage and provide sensitivity methods.
4. A prespecified paired sign-flip permutation test on specimen-level differences may be used if exchangeability/symmetry assumptions are defensible; otherwise use a suitable paired analysis. Report confidence intervals and magnitude before p-values.
5. For a single pilot region, report **descriptive results only**, with no inferential claims.

If some scanner comparisons or replicate regions are repeated within specimens, preserve their dependency during resampling. For many-level data, consider hierarchical analysis as a sensitivity check, not as an ad hoc substitute for verified independent groups.

## 8. Multiple comparisons

One predeclared **primary endpoint × primary encoder × primary graph budget × primary scanner comparison** is confirmatory. Other encoders, radii, edge densities, scanners, and pooling metrics are secondary or exploratory. Report all tested settings, not only favorable results. If making confirmatory secondary claims, preselect a family and adjust for multiplicity (e.g., Holm); otherwise label them exploratory and avoid binary significance claims.

## 9. Sensitivity and robustness checks

Predefine grids after feasibility and literature review, preferably using development data only:

- Graph density / edge budget: low, medium, high **feasible common** budgets.
- Spatial candidate radius and neighborhood definitions: 4- vs 8-neighbor, optionally larger validated radii.
- Encoders: DINOv2-Small baseline and at least one accessible pathology encoder, with their official preprocessing.
- Regions: tissue-rich vs mixed tissue content, excluding empty/background regions via a **scanner-neutral** paired rule.
- Registration: measure sensitivity to questionable correspondence; use quality flags and a prespecified exclusion rule.
- Similarity metric: cosine as primary; Euclidean on normalized embeddings only as planned robustness check.
- Graph construction: deterministic global sorting for fixed M and stable tie-breaks; record cases where constraints force edges or affect connectivity.
- Spatial geometry: verify coordinates represent physical adjacency and record orientation/scaling before computing Euclidean grid distances.
- Generalizability: stratify by specimen, scanner pair, and staining condition when the sample allows; do not pool incompatible acquisitions without accounting for condition.

## 10. Null and reference baselines

- **Identical spatial graph:** J = 1 when the same fixed grid is applied to both scanners, so it only validates graph alignment/code.
- **Random matched-budget graphs:** use shared node identities and same M; generate independently with fixed seeds. They provide a reference distribution, not tissue ground truth.
- **Matched-edge-budget feature graphs:** primary reference comparison.
- Optionally analyze node-level edge selection frequencies over repeated randomized tie-breaks or mild input perturbations as an auxiliary stability assessment; report perturbation design explicitly.

## 11. Reporting and visualization

Per region and scanner: node count, edge budget, possible spatial candidate edges, completeness, tissue and registration QC, and connectivity. Per pair: edge Jaccard, mean neighbor Jaccard, shared/unique edges, difference D. Across specimen groups: paired scatterplot, distributions of D, stability-vs-edge-budget curve with specimen-level uncertainty, and per-scanner graph examples. Keep Phase 1 figures separated from confirmatory results.

Recommended result tables:

- `metadata_feasibility.csv`: specimen ID, slide ID, scanner/stainer, region ID, matching and quality flags.
- `graph_metrics_per_region.csv`: specimen, region, scanner pair, encoder, method, density, Jaccard, neighbor overlap, connectivity.
- `primary_paired_differences.csv`: one line per independent specimen with both method scores and D.
- `bootstrap_summary.csv`: sample size, effect estimate, CI limits, bootstrap method, seed.
- `sensitivity_results.csv`: settings, coverage, point estimates, uncertainty.

## 12. Implementation and validation checks

Before the larger study:

1. Verify node ordering and tile ID correspondence; edges must use registered node IDs, not unrelated array indices.
2. Confirm graph symmetry, no self-loops, no duplicate edges, and identical eligible nodes across scanners.
3. Test Jaccard with identical graphs (expected 1), disjoint nonempty edge sets (expected 0), and controlled intersections.
4. Check fixed-budget graph builders return exactly M edges, or an explicit infeasible status.
5. Repeat all computations with a fixed seed and confirm same outputs; log package versions, model IDs/revisions, and hardware.
6. Run sensitivity analyses without opening the held-out test split.
7. Freeze the final protocol and code hash **before** the confirmatory run.

## 13. Interpretation rules

**Evidence supporting the primary hypothesis:** a consistently positive specimen-level D in a preregistered matched-budget comparison, with uncertainty adequately characterized and no selective reporting. **Evidence not supporting it:** a near-zero, negative, or highly uncertain D; a finding confined to one encoder or parameter range must be described narrowly.

A stable graph may be uninformative (e.g., a fixed spatial grid); therefore **stability is not equivalent to semantic fidelity or diagnostic accuracy**. Coarsening/pooling assignment stability and downstream model behavior require separate experiments and endpoints.

## 14. Decision gates before Phase 3

- [ ] PLISM specimen identity and scanner pairing verified.
- [ ] Spatial coordinate interpretation verified.
- [ ] Sufficient independent specimens for the intended claim identified.
- [ ] Scanner-neutral image quality control rule frozen.
- [ ] One primary encoder, graph density, spatial radius, and endpoint frozen.
- [ ] Same-node, same-edge-budget graph comparator implemented and validated.
- [ ] Specimen-group holdout designated with no overlap.
- [ ] Statistical resampling/test code validated on synthetic examples.
- [ ] Coarsening/downstream protocol (Step 2.7) separately defined.
- [ ] Final master protocol (Step 2.8) signed off before confirmatory experiments.

## 15. Sources and methodological references

The following are methodological references, not claims that these studies tested our exact PLISM graph experiment:

1. Wills & Meyer, **Metrics for graph comparison: A practitioner's guide** (2020). https://doi.org/10.1371/journal.pone.0228728
2. **Avoiding Blunders When Analyzing Correlated Data, Clustered Data or Repeated Measures** (2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10543393/
3. **Simple Approaches for Dealing With Correlated Data** (2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC10893847/
4. **Cluster-robust inference: A guide to empirical practice** (2023). https://doi.org/10.1016/j.jeconom.2022.04.001
5. **Reliability of foundation models for image retrieval in histopathology** (2026). https://www.nature.com/articles/s44303-026-00174-7

---

**Next:** Phase 2, Step 2.7: Hierarchical Graph Coarsening and Downstream Task Evaluation Protocol.
