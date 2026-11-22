# Phase 2, Step 2.5: Feature Extractor Selection and Comparison Protocol

**Research project:** Cross-Scanner Graph Topology Stability in Computational Histopathology  
**Status:** Proposed experimental protocol, not executed  
**Date:** 2026-11-23  
**Location in repository:** `research_design/protocols/06_feature_extractor_selection_comparison_protocol.md`

## 1. Purpose

Determine whether scanner-associated instability of histopathology graph edges is specific to a particular patch encoder or persists across general-purpose and pathology-specialized pretrained image representations. Hold image pairs and graph-construction procedures fixed within each comparison.

This stage **specifies the protocol**. It does not download model weights, generate new embeddings, or claim model superiority.

## 2. Starting evidence from Phase 1

The PLISM pilot used 90 paired 224 × 224 tissue tiles from the same GIVH staining condition, scanned with AT2 and GT450. DINOv2-Small generated 384-dimensional frozen CLS-token embeddings:

| Pilot metric | Observation |
|---|---:|
| Mean paired cosine similarity | 0.973113 |
| Minimum / maximum paired cosine similarity | 0.940725 / 0.986593 |
| Spatial graph edge Jaccard | 1.000000 |
| Feature-kNN edge Jaccard | 0.357724 |
| Spatial-feature hybrid edge Jaccard | 0.796935 |

The graph types used different edge budgets and neighborhood rules. These numbers are feasibility observations, not controlled evidence of a superior method.

## 3. Candidate encoders

| Encoder | Pretraining/domain | Representative embedding dimension | Access and role |
|---|---|---:|---|
| `facebook/dinov2-small` | General-purpose vision | 384 | Required baseline; already run |
| `owkin/phikon-v2` | Histopathology-specific | 1024 | Priority pathology comparator; check current model license and implementation |
| `MahmoodLab/UNI2-h` | Histopathology-specific | 1536 | Optional large-model comparator; Hugging Face access approval required |
| `MahmoodLab/CONCH` | Pathology vision-language | Verify from the selected checkpoint/output API | Optional different pretraining family; gated access and academic-use licensing |

**Selection plan:** Use DINOv2-Small + Phikon-v2 as the initial two-model comparison if terms and available hardware permit. Add UNI2-h or CONCH only if useful for the scientific question and feasible under access, storage, and GPU constraints. Avoid substituting models mid-experiment without documenting it.

Model feature dimensions refer to selected model representations and must be verified by a shape check at execution time. The 384-dimensional DINOv2 baseline was already measured locally.

## 4. Model access, licensing, and provenance

1. Open each official model page before use and read its current license and access terms.
2. Request gated weights through the official platform when required; do not distribute restricted weights.
3. Record exact repository name, resolved checkpoint revision/commit hash, model library version, license notice, output embedding type, and approved usage conditions.
4. Do not assume all model outputs are equivalent: distinguish CLS features, pooled features, projected embeddings, and normalized embeddings.
5. Keep downloaded checkpoints and generated embeddings local, consistent with the root `.gitignore`.

## 5. Fixed experimental material

- **Pilot material:** the same 90 paired tile IDs and exact PNG files from Phase 1.
- **Scanner conditions:** AT2 vs GT450, with matching GIVH stainer.
- **Node correspondence:** matched by `tile_id`; maintain identical node ordering across encoders and scanners.
- **Graph coordinates:** the previously inferred row/column values; validate geographic/spatial interpretation before broader claims.
- **Later main study:** independent specimens/regions selected using Step 2.3's specimen-group protocol. No patch-level split leakage.
- **Frozen encoders:** no fine-tuning in the primary analysis.

## 6. Preprocessing protocol

**Use each encoder's officially recommended preprocessing** for valid inference, and log it explicitly. Record:

- input pixel dimensions and color mode;
- resizing and crop policy;
- mean/std normalization and any color normalization;
- patch/optical magnification assumptions;
- interpolation mode and image handling;
- pretrained checkpoint and inference precision.

All paired AT2 and GT450 tiles **within a model** must undergo the same deterministic preprocessing pipeline. Different models may require different official transforms; do not force one generic transform across encoders if that would invalidate a checkpoint. Preserve native RGB images and report any color normalization separately.

The PLISM tiles are 224 × 224 in the current pilot. A model-specific 256 × 256 resize is a preprocessing step, not proof of underlying tissue magnification equivalence.

## 7. Embedding extraction

1. Sort paired locations by `(grid_row, grid_col, tile_id)` and save one node index file.
2. Evaluate each encoder in inference mode using frozen parameters.
3. Use fixed image order, deterministic transforms, and fixed numeric precision when practical.
4. Extract one explicitly named feature representation per image, yielding `N × D` arrays.
5. Check for non-finite values, zero-norm vectors, wrong dimensions, missing images, duplicate IDs, and node-order mismatch.
6. Use L2-normalized copies for cosine distance; retain raw embeddings for descriptive raw-space analysis.
7. Save scanner arrays and metadata together, with model ID, checkpoint hash, transform description, and code version.

A suggested artifact name is `outputs/features/<encoder>_paired_embeddings.npz`, where output storage stays ignored by Git. Lightweight metrics CSVs can remain tracked.

## 8. Primary comparisons

### 8.1 Paired-feature stability

For the exact matched node set, calculate:

- paired cosine similarity: mean, median, standard deviation, and empirical quantiles;
- feature-space Euclidean distance, **reported within each model only** (dimension and scale differ);
- normalized-feature Euclidean distance where useful;
- paired retrieval / identification top-1 and top-k accuracy: given an AT2 patch, retrieve its matching GT450 patch among the same region's candidates;
- scanner-wise pairwise-distance or similarity matrix agreement (e.g., correlation of upper triangles).

A high average cosine score does not imply stable nearest-neighbor rankings.

### 8.2 Graph stability

Reuse Step 2.4's controlled methods for every encoder:

1. spatial-only fixed-coordinate control;
2. feature-based global kNN pilot graph;
3. spatial-feature hybrid kNN pilot graph;
4. matched-edge-budget global similarity graph;
5. matched-edge-budget spatially constrained similarity graph.

For any comparison claiming a spatial constraint effect, align eligible edge budgets and report potential constraints on maximum allowable spatial edges. Report edge Jaccard, per-node neighbor Jaccard, number of edges, connected components, edge density, and degree statistics. Include a null/control comparator and hyperparameter sweeps.

**Important:** A fixed spatial-only graph has perfect cross-scanner edge agreement by construction; it must not be interpreted as a successful robustness-learning result.

## 9. Hyperparameter control

Prespecify feasible values before evaluating the locked test split, for example:

| Parameter | Proposed exploratory sweep |
|---|---|
| kNN neighbors | 3, 5, 8, 10 (cap at N - 1) |
| Spatial radius | 1, √2, 2 grid units |
| Matched undirected edge budget | Several prespecified feasible densities, identical across compared methods |
| Feature metric | Cosine as primary; Euclidean on normalized vectors as consistency check |
| Edge symmetrization | Union kNN as pilot; optionally mutual kNN as sensitivity analysis |

Not every spatial neighborhood can support every edge budget. Log feasibility and choose only budgets feasible for all compared graph methods on the same node set.

## 10. Statistics and independent sampling

- Experimental replicates are independent specimens/slide groups or defensibly independent tissue regions, not individual tile pairs.
- Pair outcomes by matched physical regions and scanner pair; compare encoders on the **same** matched regions.
- Report specimen-level distributions, paired effect sizes, and confidence intervals (e.g., cluster/bootstrap at specimen level).
- Limit the number of primary hypotheses; identify secondary and exploratory analyses.
- Use development material to tune settings; use a separately held-out test grouping for final conclusions.
- If independence cannot be verified from PLISM metadata, report descriptive feasibility evidence only.

## 11. Resource and engineering checks

Before broad extraction, run a **small compatibility check** with a few paired tiles per encoder:

- Does the model load in the current Python, Torch, Transformers/timm environment?
- What are required GPU memory, per-batch runtime, output shape, and preprocessing details?
- Does weight access require approval?
- Can inference run at batch size 1 if the model is too large?
- Do results contain NaNs or precision-related anomalies?

Do not start by downloading the entire PLISM collection or every candidate checkpoint. Freeze the protocol and run the two-model pilot first.

## 12. Decision criteria for moving beyond the pilot

Proceed to a multi-region comparison when:

1. two eligible encoders produce valid, aligned features for all pilot pairs;
2. preprocessing and checkpoint provenance are recorded;
3. matched-budget graph comparisons are implemented and validated;
4. independent specimen sampling feasibility has been established;
5. the primary stability metrics and split protocol are frozen.

If a model is gated or infeasible, record the issue and nominate a documented substitute rather than quietly altering the model roster.

## 13. Threats to validity

- DINOv2 and pathology encoders may be trained on markedly different image scales and tissues.
- Scanner effects can be entangled with preprocessing, registration errors, and staining.
- Raw embedding distance magnitudes are not directly comparable between architectures.
- Feature dimensions and normalization affect similarity statistics.
- Multiple nodes from one region are correlated.
- Graph stability does not imply classification accuracy, graph pooling robustness, or clinical utility.
- Model licenses can restrict redistribution and research use.

## Official model references

- DINOv2-Small: https://huggingface.co/facebook/dinov2-small
- DINOv2 model card: https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md
- Phikon-v2: https://huggingface.co/owkin/phikon-v2
- UNI2-h: https://huggingface.co/MahmoodLab/UNI2-h
- CONCH: https://huggingface.co/MahmoodLab/CONCH

