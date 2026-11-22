# Phase 2 · Step 2.7: Graph Coarsening and Downstream Evaluation Protocol

**Project:** Cross-Scanner Graph Topology Stability in Computational Histopathology  
**Status:** Proposed protocol, not executed  
**Input from Phase 1:** PLISM GIVH, AT2 vs GT450, 90 paired 224×224 tiles, frozen DINOv2-Small features (384 dimensions)  
**Version:** 1.0 · 23 November 2026

## 1. Aim

Determine whether scanner-related differences in patch features and graph edges persist, diminish, or amplify when graphs are coarsened into clusters or hierarchical representations. Separately test whether coarsening stability predicts stable graph-level outputs. **Topology consistency is not a substitute for diagnostic accuracy.**

## 2. Evidence motivating this stage

In the *single-region* Phase 1 pilot, corresponding tiles had mean embedding cosine similarity **0.973113**, while feature-kNN graphs had edge Jaccard **0.357724** and the spatially restricted hybrid graphs **0.796935**. Fixed spatial adjacency was identical by construction (**1.000000**). The pilot used different neighborhood rules and edge counts, so it establishes a phenomenon worth testing, **not** superiority of one graph model.

These values are descriptive; no GNN, learned pooling, or coarsening was evaluated in Phase 1.

## 3. Research questions

- **RQ-C1:** For matched scanner graphs, how often do the same physical patches receive corresponding cluster assignments after pooling?
- **RQ-C2:** Does a change in graph topology lead to an additional change in the pooled adjacency structure, beyond feature differences alone?
- **RQ-C3:** Under matched graph budgets and frozen models, do spatial constraints affect pooling stability?
- **RQ-C4 (conditional):** Is higher pooling stability associated with consistent graph embeddings or predictions on an appropriately labeled, independent dataset?

### Hypotheses to test, not assume

- **HC1:** Pooling assignments differ across scanners for at least some independent matched regions.
- **HC2:** The extent of disagreement depends on graph construction, pooling algorithm, and pooling ratio.
- **HC3:** Higher edge overlap does not necessarily imply higher cluster agreement or predictive performance.

## 4. Experimental unit and data requirements

- **Primary independent unit:** An original specimen/slide group, subject to verification of PLISM's sample identifiers and specimen independence.
- **Within-unit observations:** Multiple matched, registered patches constituting a graph. Do not count these nodes as independent study replicates.
- **Pairing:** Same tissue coordinates and node ordering for AT2 and GT450; exclude unmatched locations from both graphs.
- **Pilot:** Use the existing 90-node region only for code verification and exploratory figures. Never include it in held-out evaluation.
- **Scaling gate:** Verify specimen identities, independence, registered coordinates, and sufficient region sizes *before* selecting additional regions or claiming population-level results.

## 5. Controlled input-factor design

| Factor | Proposed levels | Control requirement |
|---|---|---|
| Scanner | AT2, GT450 initially | Precisely matched tissue nodes |
| Encoder | DINOv2-Small, Phikon-v2 initially | Identical images per condition; frozen weights |
| Graph construction | Fixed spatial, feature-only, spatial-feature | Same node set; report exact edge budget and degree distributions |
| Edge budget | e.g. 2N, 3N, and 4N undirected edges | Feasibility of candidates checked before fixing M |
| Pooling model | Fixed deterministic control, graph clustering, learned method | Same weights/seed for both scanners |
| Pooling ratio | e.g. 25%, 50%, 75% of N nodes retained or cluster targets | Explicit rounded target count; do not compare incompatible definitions |
| Repetition | At least 5 seeds for learned methods (provisional) | Pair scanner inputs within each seed |

An edge-budget example such as **M = 3N = 270** edges for N=90 is feasible only when every compared construction has at least 270 eligible edges. The original 8-neighbor spatial graph contained 306; stricter neighborhood rules may have fewer candidates. Record and resolve any infeasibility rather than silently changing M.

## 6. Pooling and coarsening baselines

1. **Identity/no pooling:** Required control; measures input graph differences without pooling.
2. **Coordinate-only deterministic region pooling:** Partition nodes into spatial blocks with fixed boundaries, shared across scanners. **Expected identical assignments by construction**; tests whether pooled edge weights/topology nevertheless change.
3. **Feature-based fixed clustering:** Run a deterministic clustering method with fixed cluster count and seed on each scanner's features; use an identical configuration and evaluate label-invariant assignment agreement. This is an analysis baseline, not necessarily GNN pooling.
4. **DiffPool:** Learned soft node-to-cluster assignments; evaluate hard and soft agreement. Use shared frozen weights at comparison time, trained only on development specimens.
5. **MinCutPool:** Alternative learned clustering/pooling objective, included when code and compute permit; likewise freeze one model for both scanners.

Do not introduce learned pooling as the primary experiment until the no-pooling and deterministic controls pass. Compare DiffPool and MinCutPool under their own documented capacity and objective differences; avoid pretending they are functionally identical.

## 7. Assignment comparison with matched nodes

Given assignment matrices `S_AT2` and `S_GT450`, with rows aligned by **tile_id**, report:

- **Adjusted Rand Index (ARI):** Primary hard-assignment agreement metric after taking `argmax` for soft assignments, if the chosen model defines clusters this way. ARI is invariant to arbitrary cluster label permutations and adjusts for chance.
- **Adjusted Mutual Information (AMI):** Secondary chance-adjusted assignment similarity. Optional NMI may be reported descriptively, but NMI is not chance-adjusted.
- **Soft assignment agreement:** Align cluster columns using an explicitly specified matching method, e.g. maximum-weight bipartite matching on assignment overlap, then report mean per-node Jensen–Shannon divergence or another bounded divergence. The method for unequal cluster counts must be prespecified.
- **Cluster-size distribution and collapse rate:** Report empty clusters, size imbalance, and effective number of occupied clusters.
- **Spatial coherence:** Evaluate whether clustered tiles are spatially contiguous. This is a *separate* desideratum and should not be treated as scanner agreement.

**Critical:** Comparing raw cluster IDs across separately constructed graphs is invalid because cluster labels are arbitrary. ARI/AMI handle label permutations; soft comparisons require explicit column alignment.

## 8. Pooled graph topology comparison

For a soft pooling matrix `S`, graph node features `X`, and adjacency `A`, the standard form is:

- `X_pool = Sᵀ X`
- `A_pool = Sᵀ A S`

These are representative operations for assignment-based pooling; details vary among models.

Pooled clusters may not have a direct one-to-one identity across scanners. **Align or compare clusters in a permutation-invariant way before reporting pooled edge overlap.** Where alignment is unstable or sizes differ, report invariant statistics instead: edge-weight distributions, graph density, Laplacian spectral summaries, component counts, and graph edit-style measures with their assumptions stated.

If weighted adjacencies are thresholded, pre-register thresholding and edge-budget rules; do not choose thresholds after seeing which method looks most stable.

## 9. Factorial ablations to separate causes

For each matched specimen and trained frozen pooling model, compare four input conditions:

| Condition | Node features | Adjacency | Purpose |
|---|---|---|---|
| AA | AT2 | AT2 | Reference |
| BB | GT450 | GT450 | Full scanner change |
| BA | GT450 | AT2 | Feature change only |
| AB | AT2 | GT450 | Edge change only |

This isolates **input interventions within a fixed model**. Differences may interact and are not necessarily additive. Ensure all adjacency matrices use identical ordering and the same edge-count settings.

## 10. Graph-level representations and predictions

### 10.1 Unlabeled evaluation (available earlier)

- Freeze a graph encoder and pooling component trained solely on development data, or clearly label a randomly initialized network as a *sanity check*, not a meaningful model.
- For matched scanners, compare graph embeddings using cosine similarity and Euclidean distance (with a standardized feature scale).
- Examine output sensitivity to graph construction and coarsening choice, across multiple independent specimens and model seeds.

### 10.2 Diagnostic evaluation (conditional; separate dataset gate)

Only if a dataset provides valid **specimen-level pathology labels** with appropriate scanner pairing or trustworthy external scanner-domain evaluation:

- Train models on prespecified training subjects and validate without subject/slide leakage.
- Report classification AUROC, balanced accuracy, macro-F1, calibration, and uncertainty as appropriate to label structure.
- Report scanner prediction agreement **alongside** accuracy; equal predictions can be consistently wrong.
- Analyze trade-offs: graph stability versus clinical utility. Never infer improved diagnosis from topology Jaccard alone.

**PLISM tile matching alone does not establish a suitable diagnostic classification benchmark.**

## 11. Analysis plan

### Primary exploratory outcomes

1. Per-specimen ARI for paired scanner pooling assignments.
2. Per-specimen AMI as corroboration.
3. Edge Jaccard before pooling, and aligned pooled edge agreement when identifiable.

### Secondary outcomes

- Soft assignment divergence; cluster collapse; graph embedding cosine; connectivity; spatial coherence; runtime and memory.

### Statistical discipline

- Calculate matched scanner-pair comparisons within a specimen and aggregate at the **specimen level**.
- Use confidence intervals obtained by resampling **independent specimens**, not individual tiles.
- Treat encoder, graph method, pooling ratio, and random seed as structured repeated factors.
- State the primary contrast before main experiments; correct across confirmatory secondary tests where warranted.
- Report effect sizes with intervals and negative results. Phase 1 numbers are not hypothesis tests.

## 12. Reproducibility and failure checks

For every experiment save: original specimen and slide IDs, scanner/stainer, selected tile IDs and coordinates, split membership, encoder revision and preprocessing, graph construction, edge count, graph seed, pooling architecture and checkpoint, optimizer/training split, pooling ratio, and software versions.

Required checks:

- Pairwise tile IDs and order identical between scanner inputs.
- Identical spatial adjacency yields edge Jaccard 1.0.
- Identical input features and adjacency under deterministic frozen inference produce identical outputs (within numerical tolerance).
- Random cluster-ID permutation leaves ARI/AMI unchanged.
- No train/test specimen overlap, even through adjacent regions.
- No fitting of preprocessing, encoder tuning, graph rules, or pooling on held-out test specimens.
- Flag NaNs, degenerate one-cluster outputs, and empty/unused clusters.

## 13. Decision gates

**Gate A: Dataset readiness.** Independent specimen metadata and valid matched coordinates confirmed.  
**Gate B: Graph readiness.** Fair matched-budget baselines and permutation/unit tests pass.  
**Gate C: Coarsening readiness.** Deterministic controls pass; trained pooling does not collapse and reports stable losses.  
**Gate D: Inference readiness.** One frozen model processes both scanners under fixed rules.  
**Gate E: Clinical extension.** Independent labeled data are available, with a leakage-resistant evaluation design.

Do not move to large-scale inference or diagnostic claims if these gates fail.

## 14. Recommended implementation sequence after Phase 2

1. Reproduce Phase 1 graph baselines using the saved 90-node PLISM pilot.
2. Add deterministic spatial-block pooling and verify ARI = 1 for its fixed paired assignments.
3. Add a deterministic feature-clustering baseline and calculate ARI/AMI.
4. Perform the four-condition adjacency/features ablation.
5. Expand to independent specimens only after dataset-identity and pairing validation.
6. Add learned DiffPool/MinCutPool using development-only trained checkpoints.
7. Evaluate downstream representation consistency and conditional classification on appropriate labeled data.

The protocol defines procedures; **none of these coarsening experiments has yet been run**.

## 15. Primary references

- Ying et al. (2018). *Hierarchical Graph Representation Learning with Differentiable Pooling.* NeurIPS. https://proceedings.neurips.cc/paper_files/paper/2018/hash/e77dbaf6759253c7c6d0efc5690369c7-Abstract.html
- Bianchi, Grattarola & Alippi (2020). *Spectral Clustering with Graph Neural Networks for Graph Pooling.* ICML/PMLR. https://proceedings.mlr.press/v119/bianchi20a.html
- Liu et al. (2023). *Graph Pooling for Graph Neural Networks: Progress, Challenges, and Opportunities.* IJCAI survey. https://www.ijcai.org/proceedings/2023/752
- scikit-learn documentation. *Clustering metrics: adjusted Rand index and adjusted mutual information.* https://scikit-learn.org/stable/modules/clustering.html

---