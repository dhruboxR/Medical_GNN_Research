
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd

# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path("dataset_exploration/outputs")

GRAPH_FILE = ROOT / "tables/plism_graph_stability.csv"
MOSAIC_FILE = ROOT / "tables/plism_spatial_mosaic_metrics.csv"
FEATURE_FILE = ROOT / "features/plism_dinov2_features.npz"
MANIFEST_FILE = (
    ROOT / "tables/plism_spatial_region_download_manifest.csv"
)
GRAPH_CONFIG_FILE = (
    ROOT / "reports/plism_graph_stability.json"
)

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = REPORT_DIR / "PLISM_PILOT_FINAL_REPORT.md"

print("=" * 65)
print("PLISM FINAL PILOT EXPLORATION REPORT")
print("=" * 65)

# ==================================================
# STEP 1: VERIFY INPUTS
# ==================================================

required = [
    GRAPH_FILE,
    MOSAIC_FILE,
    FEATURE_FILE,
    MANIFEST_FILE,
    GRAPH_CONFIG_FILE
]

for path in required:
    if not path.exists():
        raise SystemExit(f"Required file missing: {path}")

print("\n[1] Loading exploration results...")

graphs = pd.read_csv(GRAPH_FILE)
mosaic = pd.read_csv(MOSAIC_FILE)
manifest = pd.read_csv(MANIFEST_FILE)

with open(GRAPH_CONFIG_FILE, encoding="utf-8") as f:
    graph_config = json.load(f)

with np.load(FEATURE_FILE) as data:
    a = data["at2"].astype(np.float64)
    b = data["gt450"].astype(np.float64)
    tile_ids = data["tile_ids"].astype(str)

if a.shape != b.shape:
    raise SystemExit("Feature shapes do not match.")

if len(manifest) != len(tile_ids):
    raise SystemExit("Manifest and feature counts differ.")

if set(manifest["tile_id"]) != set(tile_ids):
    raise SystemExit("Manifest and feature IDs differ.")

# ==================================================
# STEP 2: COMPUTE SUMMARY METRICS
# ==================================================

print("\n[2] Calculating final statistics...")

norm_a = a / np.maximum(
    np.linalg.norm(a, axis=1, keepdims=True),
    1e-12
)
norm_b = b / np.maximum(
    np.linalg.norm(b, axis=1, keepdims=True),
    1e-12
)

paired_cosine = np.sum(norm_a * norm_b, axis=1)

paired_distance = np.linalg.norm(a - b, axis=1)

matched_locations = len(tile_ids)
feature_dim = a.shape[1]

pixel_mae = float(mosaic["pixel_mae"].mean())

at2_brightness = float(
    mosaic["at2_mean_brightness"].mean()
)
gt450_brightness = float(
    mosaic["gt450_mean_brightness"].mean()
)

grid_rows = manifest["grid_row"].nunique()
grid_cols = manifest["grid_col"].nunique()

grid_coverage = (
    matched_locations / (grid_rows * grid_cols)
)

# ==================================================
# STEP 3: FORMAT GRAPH RESULTS
# ==================================================

graph_lines = []

for _, row in graphs.iterrows():
    graph_lines.append(
        f"| {row['graph_type']} "
        f"| {int(row['at2_edges'])} "
        f"| {int(row['gt450_edges'])} "
        f"| {int(row['shared_edges'])} "
        f"| {row['edge_jaccard']:.4f} "
        f"| {row['mean_neighbor_jaccard']:.4f} "
        f"| {int(row['at2_components'])} "
        f"| {int(row['gt450_components'])} |"
    )

graph_table = "\n".join(graph_lines)

# ==================================================
# STEP 4: GENERATE REPORT
# ==================================================

print("\n[3] Generating Markdown report...")

report = f"""# PLISM Pilot Dataset Exploration Report

## 1. Research Context

**Research direction:** Cross-scanner graph topology
stability in computational histopathology.

**Dataset:** PLISM

**Staining condition:** GIVH

**Scanners:** AT2 and GT450

**Feature extractor:** facebook/dinov2-small

**Report generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

### Objective

Investigate whether scanner-dependent image feature
differences alter graph connections when corresponding
tissue locations are represented as graph nodes.

This is a pilot feasibility analysis, not a diagnostic
classification or GNN training experiment.

## 2. Dataset Preparation

| Property | Result |
|---|---|
| Matched tissue locations | {matched_locations} |
| Scanners | AT2, GT450 |
| Staining condition | GIVH |
| Valid paired images | {matched_locations * 2} |
| Grid dimensions | {grid_rows} x {grid_cols} |
| Grid coverage | {grid_coverage:.2%} |
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
| Mean paired pixel MAE | {pixel_mae:.4f} |
| AT2 average brightness | {at2_brightness:.4f} |
| GT450 average brightness | {gt450_brightness:.4f} |

GT450 images were visibly paler than AT2 images.

The brightness-based tissue coverage estimate is
sensitive to scanner appearance and must not be
interpreted as a reliable segmentation measurement.

Pixel differences do not directly establish graph
topology instability.

## 4. Feature Extraction

**Model:** DINOv2-Small

**Embedding dimensionality:** {feature_dim}

**Matched feature pairs:** {matched_locations}

| Metric | Result |
|---|---:|
| Mean paired cosine similarity | {paired_cosine.mean():.6f} |
| Minimum paired cosine similarity | {paired_cosine.min():.6f} |
| Maximum paired cosine similarity | {paired_cosine.max():.6f} |
| Mean Euclidean feature distance | {paired_distance.mean():.6f} |

The paired embeddings showed high cosine similarity.
However, feature similarity alone cannot establish
stability of nearest-neighbor relationships.

## 5. Cross-Scanner Graph Comparison

Three graph constructions were examined:

1. Spatial-only graph using fixed grid adjacency.
2. Feature-based cosine k-nearest-neighbor graph.
3. Hybrid graph using feature similarity within
   spatially restricted candidate neighbors.

Feature k: {graph_config['feature_k']}

Hybrid k: {graph_config['hybrid_k']}

| Graph | AT2 edges | GT450 edges | Shared | Edge Jaccard | Neighbor Jaccard | AT2 components | GT450 components |
|---|---:|---:|---:|---:|---:|---:|---:|
{graph_table}

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
"""

OUTPUT.write_text(report, encoding="utf-8")

# ==================================================
# STEP 5: FINISH
# ==================================================

print("\n[4] Final summary:")
print(f"Matched locations: {matched_locations}")
print(f"Feature dimension: {feature_dim}")
print(f"Mean cosine similarity: {paired_cosine.mean():.6f}")

for _, row in graphs.iterrows():
    print(
        f"{row['graph_type']}: "
        f"Edge Jaccard = {row['edge_jaccard']:.6f}"
    )

print("\n" + "=" * 65)
print("PILOT EXPLORATION REPORT GENERATED")
print(f"Saved: {OUTPUT}")
print("=" * 65)
