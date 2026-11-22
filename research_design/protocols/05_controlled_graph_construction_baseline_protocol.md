# Phase 2, Step 2.4: Controlled Graph Construction and Baseline Protocol

**Project:** Cross-Scanner Graph Topology Stability in Computational Histopathology  
**Status:** Proposed protocol, pending feasibility checks and literature review validation  
**Dataset:** PLISM, registered scanner-paired tissue patches  
**Initial scanner comparison:** AT2 versus GT450; GIVH staining condition  
**Pilot reference:** 90 matched patch locations; frozen DINOv2-Small, 384-dimensional CLS embeddings

## 1. Objective

Test how graph construction choices influence cross-scanner adjacency consistency **for the same registered tissue locations**, while controlling graph size, edge count, candidate neighborhoods, encoder, and preprocessing. Separately assess whether increased edge consistency helps representation stability or downstream performance. No novel architecture or clinical benefit is assumed.

## 2. Observations motivating this protocol

| Pilot construction | AT2 edges | GT450 edges | Shared edges | Edge Jaccard | Mean node-neighbor Jaccard |
|---|---:|---:|---:|---:|---:|
| Fixed spatial eight-neighbor | 306 | 306 | 306 | 1.000000 | 1.000000 |
| Feature cosine kNN, k=5; symmetrized union | 328 | 340 | 176 | 0.357724 | 0.379607 |
| Feature ranking within spatial eight-neighbor candidates, k=4; symmetrized union | 231 | 238 | 208 | 0.796935 | 0.815357 |

Paired cosine similarity averaged **0.973113**. These results demonstrate pilot graph-edge disagreement, but the construction methods had **different edge counts, candidate sets, and k values**. The fixed spatial result is deterministic because coordinates were shared; it is not independent scanner-robustness evidence.

## 3. Definitions and experimental unit

- **Node:** One matched 224 × 224 tissue patch at a verified tile ID. The AT2 and GT450 graphs use exactly the same node ordering.
- **Region graph:** An independently sampled, spatially coherent tissue region with at least a prespecified minimum number of matched, usable nodes.
- **Pair:** Two scanner acquisitions of that same set of tissue locations.
- **Replicate for inference:** An independent original specimen/slide grouping, **not an edge, node, or multiple regions from the same slide**. Multiple regions in one specimen must remain in the same data partition and should be aggregated or modeled as clustered observations.
- **Node coordinates:** Parsed tile-ID grid coordinates, used only after checking the tiling convention and physical spatial correspondence.
- **Edge:** Undirected, unweighted for primary topology metrics; self-loops excluded. Weighted edges are a secondary analysis.

Verify that identifiers represent independent specimens before inferring biological sample sizes. Do not equate scanner-group labels or shard numbers with specimen IDs.

## 4. Frozen input and preprocessing

1. Construct a fixed eligible-node mask for each paired region; apply it identically to both scanners. Do **not** independently drop nodes using scanner-specific brightness-based tissue thresholds.
2. Validate correspondence, image dimensions, orientation, and spatial assembly. Record registration limitations.
3. Use identical encoder weights, resizing, normalization, and inference settings for all conditions. Run inference in evaluation mode, with no stochastic augmentation.
4. Extract embeddings separately per scanner, preserving matched tile ID order. Normalize each feature vector to unit L2 norm for cosine-based edge ranking; retain original embeddings for separate feature metrics.
5. Cache per-region input manifests, model name/revision, software versions, random seeds, and embedding arrays. Raw image tiles and binary embeddings remain outside Git unless a separate data publication plan is approved.
6. Do not tune methods on held-out test specimens.

## 5. Primary graph baselines

### B0. Fixed spatial graph (control)

Connect nodes in the verified 2D grid if their Chebyshev distance is one (`max(|Δrow|, |Δcol|) = 1`), corresponding to eight-neighbor adjacency. This construction is identical across the two scanners by definition. Treat its stability of 1.0 as an **identity check**. Report edge budget and density; do not interpret perfect overlap as better learning capacity.

### B1. Global feature kNN (pilot replication)

For each node, select its **k** highest-cosine-similarity other nodes anywhere in the region. Break ties deterministically by the common node index. Convert the directed neighbors to one undirected edge set using the **union** rule. Report resulting edge counts because this method does **not** guarantee an identical undirected edge count across scanners.

### B2. Spatially constrained feature kNN (pilot replication)

For each node, rank only neighbors within a fixed spatial candidate radius, then select up to **k** by cosine similarity. Symmetrize using the same undirected union rule. Report varying node degrees and edge counts. The primary local candidate set must use **identical node coordinates** across scanners.

### B3. Edge-budget-matched global feature graph (primary controlled baseline)

Build a candidate set of all unordered node pairs, calculate cosine similarity for each pair, and retain exactly **M** edges with the highest scores. Use deterministic tie breaking. Choose M **before evaluating the held-out test set**. This ensures AT2 and GT450 have equal numbers of undirected edges. Connectivity is not guaranteed and must be measured, not silently repaired.

### B4. Edge-budget-matched spatially constrained graph (primary controlled comparator)

Build a candidate set of unordered node pairs within spatial radius **r**, score them by cosine similarity, and retain exactly **M** highest-ranked edges. Use the *same* M for B3 and B4, and the same candidate radius for the two scanners. Require `M ≤ number of eligible spatial candidate edges` for **every included region**; otherwise mark the configuration infeasible and select a smaller preregistered M or larger r. Do not alter M separately for AT2 and GT450.

**Caution:** B4 may score better partly because it has a smaller set of possible edges and a strong shared coordinate prior. Report this clearly and add null controls (Section 7). Equal edge count alone does not isolate every causal factor.

### B5. Fixed-spatial candidate graph (optional context)

Use all candidate spatial edges without feature ranking. It helps characterize the effect of candidate restriction but should not be treated as a head-to-head matched-budget method unless the edge count is deliberately aligned.

## 6. Edge-budget controls and hyperparameter grid

Use a predefined grid on development specimens, then freeze choices before testing:

| Parameter | Candidate settings | Notes |
|---|---|---|
| Original pilot k | 3, 5, 8, 10 | For directed kNN + union, report realized undirected edges |
| Spatial radius r | 1, 2, 3 grid steps | Chebyshev-radius candidate neighborhoods; validate coordinate units |
| Edge budget M | Fixed target edge density 5%, 10%, 15% **when feasible** | Convert density to integer M per region by a single prespecified rule; enforce same M across scanners and B3/B4 |
| Cosine ranking | L2-normalized features | Primary feature similarity |
| Encoder | DINOv2-Small first | Later add pathology-specific encoders under a frozen comparison protocol |
| Ties | Common deterministic node index | Do not use random tie breaking |

**Feasibility rule:** A local candidate graph may have fewer edges than an unconstrained density target, particularly in small/irregular regions. Check feasibility per region before running a configuration. For fair analysis across methods, compare only configurations jointly feasible for that region; report exclusions and never substitute a different M for one scanner.

**Selection rule:** The pilot region is development-only. Select primary M and r using independent development/validation groups and predefined criteria. Avoid choosing the best-looking stability number from the held-out test set.

## 7. Required null and diagnostic controls

1. **Spatial identity check:** B0 must have Jaccard 1.0 for paired graphs sharing coordinates. If not, investigate a node-order or masking bug.
2. **Same-scanner determinism:** Recompute graph edges from identical saved features; edge sets must match exactly.
3. **Matched-budget random graph null:** Sample M unordered edges independently for each condition, using a fixed seed and the same *eligible candidate universe* as each method. Repeat across seeds and estimate the expected accidental overlap.
4. **Coordinate-restricted random null:** Independently sample M edges from spatially eligible candidates to assess how much agreement is explained by the smaller candidate universe rather than learned features.
5. **Candidate-space reporting:** Always record the size of each method's eligible edge universe so spatial restrictions cannot masquerade as feature robustness.
6. **Registration sensitivity:** If correspondence is uncertain, stratify by registration quality or exclude ambiguous region pairs under documented rules.

## 8. Stability and structure metrics

For scanner edge sets A and B, computed on the same N matched nodes:

- **Edge Jaccard:** `|A ∩ B| / |A ∪ B|` (1.0 if both empty, but avoid degenerate configurations).
- **Shared-edge fraction:** `|A ∩ B| / M` when both graphs contain exactly M edges. Report this alongside Jaccard because it is easier to interpret under a matched budget.
- **Edge disagreement:** Number of A-only and B-only edges.
- **Mean node-neighbor Jaccard:** Average Jaccard of each node's scanner-specific undirected neighbor sets; specify the convention for nodes with empty unions.
- **Graph topology diagnostics:** Number of nodes, edges, density, average degree, degree distribution, isolated nodes, number of connected components, largest-component fraction, and clustering coefficient when defined.
- **Feature relationship diagnostics:** Paired cosine similarity and within-scanner nearest-neighbor overlap. Distinguish a same-node feature score from inter-node ranking agreement.
- **Optional:** Degree assortativity, path length only within connected components, and weighted edge stability after a justified shared weighting scheme.

Save one row **per independent region × scanner pair × encoder × method × parameter setting**, plus a versioned edge-list table and eligible-candidate counts. Never summarize statistical uncertainty using nodes as if they were independent specimens.

## 9. Primary statistical comparison

- **Primary planned contrast:** B3 versus B4 at a fixed jointly feasible M and preregistered r.
- **Outcome:** Paired difference in edge Jaccard across independently sampled specimens/regions.
- **Direction:** Test whether spatial restriction improves stability, but also assess null-adjusted overlap.
- **Uncertainty:** Report paired differences, 95% confidence intervals using resampling at the specimen level, and an appropriate paired test. If multiple regions belong to one specimen, cluster accordingly.
- **Multiple configurations:** Treat k/r/density sweeps as secondary sensitivity analysis; correct or qualify inference for multiple comparisons.
- **Small sample:** Prefer descriptive pilot summaries and intervals rather than strong significance claims.

## 10. Graph quality and downstream safeguards

**Topological agreement is not a sufficient success criterion.** A trivial fixed graph may be perfectly stable but omit useful tissue relationships. Evaluate:

- Whether the graph remains connected and preserves meaningful spatial/feature relations.
- Graph representation similarity from a **frozen, identical graph encoder**, when a suitable model is specified.
- Pooling/coarsening assignment similarity **separately**, after defining matched nodes and a fair alignment procedure for clusters.
- Prediction agreement and task performance only with independent, adequately labeled data.

A graph method that increases edge Jaccard but reduces meaningful downstream performance cannot automatically be called superior.

## 11. Leakage and split management

- Keep both acquisitions of each tissue location within the same split.
- Group all regions from the same original specimen or slide source together; verify identities from dataset documentation/metadata.
- Preserve the 90-node pilot as **development-only**.
- Lock the held-out test split before model/parameter selection.
- Do not derive exclusion thresholds or tune edge budgets using held-out test outcomes.

## 12. Minimum reproducible outputs

Create these files when implementation begins:

```text
research_design/
  protocols/
    05_controlled_graph_construction_baseline_protocol.md

controlled_graph_experiments/                # Future implementation, not required now
  configs/
    graph_grid.yaml
  scripts/
  outputs/
    tables/
      graph_pair_metrics.csv
      graph_edges.csv
      candidate_universe_sizes.csv
    reports/
      reproducibility_manifest.json
    figures/
      edge_stability_by_method.png
```

Persist exact graph code version, dataset version/commit, image/embedding source paths or hashes, encoder revision, candidate rules, seed, software versions, excluded regions, and all selected hyperparameters.

## 13. Protocol acceptance gates

Proceed to experimental implementation only after confirming:

- [ ] Physical meaning of tile coordinates and scanner-pair alignment is verified.
- [ ] Independent specimen identifiers and available scanner pairings are verified.
- [ ] Matching node IDs are unique, ordered consistently, and image pairs meet quality checks.
- [ ] Equal-M graph configurations are feasible for the selected regions.
- [ ] Candidate-space and random-null controls are included.
- [ ] Development versus held-out test separation is established.
- [ ] Primary contrast and primary outcome are preregistered in the study protocol.

## 14. Limitations of the current pilot

The existing pilot used one 90-node tissue region, DINOv2-Small, AT2/GT450, and differing graph edge budgets. It supports **method feasibility and the observation of edge changes**, but does not support statistical generalization, coarsening claims, classification benefit, or novelty claims. The exact causes of graph differences require independent paired samples and controls.

