
"""
PHASE 3B.28 - FINAL CONSOLIDATION

Verify completed Phase 3B experiments and produce
a research evidence summary and closure report.

No new experiments.
No modifications to previous results.
No trained GNN.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "multi_region_research"

TABLES = BASE / "outputs/tables"
REPORTS = BASE / "outputs/reports"
VALIDATED = BASE / "outputs/validated"

MANIFEST = VALIDATED / "phase3b_frozen_tile_manifest.csv"
FEATURES = BASE / "outputs/features/plism_phase3b_dinov2_features.npz"

EXPECTED_HASH = (
    "69d11e1a0dd1107f2e6534632727e2c2291ff83d6b17"
    "cf953ed5056bea9020f9"
)

INPUTS = {
    20: TABLES / "plism_phase3b_graph_stability_per_region.csv",
    21: TABLES / "plism_phase3b_connected_graph_per_region.csv",
    22: TABLES / "plism_phase3b_node_neighborhood_per_region.csv",
    23: TABLES / "plism_phase3b_propagation_per_region.csv",
    24: TABLES / "plism_phase3b_oversmoothing_paired.csv",
    25: TABLES / "plism_phase3b_geometry_conditioned_paired.csv",
    26: TABLES / "plism_phase3b_smoothing_matched_per_region.csv",
    27: TABLES / "plism_phase3b_equal_strength_paired.csv",
}

EXPECTED_ROWS = {
    20: 80,
    21: 60,
    22: 60,
    23: 300,
    24: 150,
    25: 30,
    26: 30,
    27: 120,
}

REPORT_PATH = REPORTS / "28_phase3b_final_evidence.json"
MARKDOWN_PATH = REPORTS / "PHASE_3B_FINAL_RESEARCH_SUMMARY.md"

for path in (REPORT_PATH, MARKDOWN_PATH):
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite: {path}")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_columns(df, names, step):
    missing = set(names) - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Step {step} missing columns: {sorted(missing)}"
        )


print("=" * 70)
print("PHASE 3B.28 - FINAL RESEARCH CONSOLIDATION")
print("=" * 70)

print("\n[1] Verifying frozen research inputs...")

if file_hash(MANIFEST) != EXPECTED_HASH:
    raise RuntimeError("Frozen manifest hash mismatch.")

manifest = pd.read_csv(MANIFEST)

if len(manifest) != 360 or manifest["region_id"].nunique() != 10:
    raise RuntimeError("Frozen manifest cohort mismatch.")

if not (manifest.groupby("region_id").size() == 36).all():
    raise RuntimeError("Invalid region sizes.")

with np.load(FEATURES, allow_pickle=False) as archive:
    if archive["at2"].shape != (360, 384):
        raise RuntimeError("AT2 feature mismatch.")
    if archive["gt450"].shape != (360, 384):
        raise RuntimeError("GT450 feature mismatch.")

print("Frozen regions: 10")
print("Paired tiles: 360")
print("Feature dimensions: 384")

print("\n[2] Verifying experiment outputs...")

data = {}
inventory = {}

for step, path in INPUTS.items():
    if not path.is_file():
        raise RuntimeError(f"Missing Step 3B.{step}: {path}")

    df = pd.read_csv(path)

    if len(df) != EXPECTED_ROWS[step]:
        raise RuntimeError(
            f"Step 3B.{step}: expected {EXPECTED_ROWS[step]} "
            f"rows, found {len(df)}"
        )

    assert_columns(df, ["region_id", "edge_budget"], step)

    if df["region_id"].nunique() != 10:
        raise RuntimeError(
            f"Step 3B.{step}: unexpected region count"
        )

    if df.isna().all(axis=1).any():
        raise RuntimeError(
            f"Step 3B.{step}: completely empty row"
        )

    data[step] = df

    inventory[str(step)] = {
        "path": str(path.relative_to(ROOT)),
        "rows": len(df),
        "sha256": file_hash(path),
    }

    print(f"  Step 3B.{step}: {len(df)} records verified")


print("\n[3] Extracting final evidence...")

# Connected graph agreement at 72 edges.
connected = data[21]
assert_columns(
    connected,
    ["graph_type", "residual_edge_jaccard",
     "residual_chance_adjusted"],
    21,
)

connected72 = connected[connected.edge_budget == 72]

connected_summary = (
    connected72.groupby("graph_type")[
        ["residual_edge_jaccard", "residual_chance_adjusted"]
    ].mean()
)

# Equal-strength primary diagnostic.
equal = data[27]
assert_columns(
    equal,
    ["alpha", "global_topology_l2",
     "spatial_topology_l2",
     "global_minus_spatial_topology_l2",
     "global_retention", "spatial_retention"],
    27,
)

primary = equal[
    (equal.edge_budget == 72)
    & np.isclose(equal.alpha, 0.50)
]

if len(primary) != 10:
    raise RuntimeError("Primary comparison incomplete.")

primary_global = float(primary.global_topology_l2.mean())
primary_spatial = float(primary.spatial_topology_l2.mean())

primary_reduction = (
    (primary_global - primary_spatial) / primary_global
)

# Geometry-conditioned results.
geometry = data[25]
assert_columns(
    geometry,
    ["global_conditional_adjusted",
     "spatial_conditional_adjusted"],
    25,
)

geometry72 = geometry[geometry.edge_budget == 72]

if len(geometry72) != 10:
    raise RuntimeError("Geometry comparison incomplete.")

geometry_differences = (
    geometry72.spatial_conditional_adjusted
    - geometry72.global_conditional_adjusted
)

# Smoothing-matched results.
matched = data[26]
assert_columns(
    matched,
    ["retention_absolute_error",
     "matched_topology_advantage"],
    26,
)

# Oversmoothing results.
smoothing = data[24]
assert_columns(
    smoothing,
    ["depth", "spatial_minus_global_retention"],
    24,
)

smoothing72 = smoothing[
    (smoothing.edge_budget == 72)
    & (smoothing.depth == 2)
]

if len(smoothing72) != 10:
    raise RuntimeError("Smoothing audit incomplete.")

evidence = {
    "primary_edge_budget": 72,
    "primary_alpha": 0.50,
    "primary_depth": 2,
    "primary_mean_global_topology_l2": primary_global,
    "primary_mean_spatial_topology_l2": primary_spatial,
    "primary_relative_l2_reduction": float(primary_reduction),
    "primary_positive_regions": int(
        (primary.global_minus_spatial_topology_l2 > 0).sum()
    ),
    "connected_global_mean_residual_jaccard": float(
        connected_summary.loc["global", "residual_edge_jaccard"]
    ),
    "connected_spatial_mean_residual_jaccard": float(
        connected_summary.loc["spatial", "residual_edge_jaccard"]
    ),
    "geometry_adjusted_spatial_positive_regions": int(
        (geometry_differences > 0).sum()
    ),
    "geometry_adjusted_mean_difference": float(
        geometry_differences.mean()
    ),
    "spatial_lower_variance_retention_regions": int(
        (smoothing72.spatial_minus_global_retention < 0).sum()
    ),
    "variance_matched_acceptable_comparisons": int(
        (matched.retention_absolute_error <= 0.005).sum()
    ),
    "variance_matched_total_comparisons": len(matched),
}

print("Primary L2 reduction:", f"{primary_reduction:.2%}")
print(
    "Primary spatial-positive regions:",
    f"{evidence['primary_positive_regions']}/10"
)
print(
    "Geometry-adjusted spatial-positive regions:",
    f"{evidence['geometry_adjusted_spatial_positive_regions']}/10"
)

print("\n[4] Writing closure documents...")

REPORTS.mkdir(parents=True, exist_ok=True)

report = {
    "phase": "3B",
    "status": "CLOSED",
    "closure_step": "3B.28",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "manifest_sha256": EXPECTED_HASH,
    "feature_archive_sha256": file_hash(FEATURES),
    "dataset": "Owkin PLISM original Parquet PNG images",
    "scanner_pair": ["AT2", "GT450"],
    "regions": 10,
    "paired_tile_positions": 360,
    "original_images": 720,
    "model": "facebook/dinov2-small",
    "feature_dimension": 384,
    "experiment_inventory": inventory,
    "key_evidence": evidence,
    "main_conclusion": (
        "Spatial graph construction showed improved observed "
        "cross-scanner topology and deterministic representation "
        "consistency in the selected regions, with smoothing "
        "and geometry-related trade-offs."
    ),
    "unresolved_limitations": [
        "One registered scanner acquisition pair",
        "Biological specimen independence unverified",
        "Manual registration review not confirmed",
        "Spatial graph advantage not universal after geometry adjustment",
        "Variance preservation is not downstream utility",
        "No trained GNN or clinical task evaluated",
        "Novelty requires current literature review",
    ],
    "next_phase": "Phase 4 - main research design and validation",
    "frozen_inputs_modified": False,
}

REPORT_PATH.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

markdown = f"""# Phase 3B: Final Research Summary

**Status:** CLOSED  
**Dataset:** Owkin PLISM  
**Scanner pair:** AT2 / GT450  
**Selected regions:** 10  
**Matched tile positions:** 360  
**Original PNG images:** 720  
**Encoder:** DINOv2-Small (384-dimensional CLS embeddings)

## Research objective

Investigate cross-scanner graph topology instability and
representation consistency under controlled graph construction.

## Confirmed findings

1. Global and spatial feature-similarity graphs select different
   edge sets between scanner acquisitions.
2. Spatial graph construction generally improves raw edge and
   node-neighborhood stability at matched edge budgets.
3. Connected spatial graphs preserve a residual-edge stability
   advantage in the selected cohort.
4. Geometry-conditioned adjustment reduces the size and
   consistency of the spatial advantage.
5. Deterministic spatial propagation exhibits smaller
   topology-induced representation differences.
6. Spatial propagation also leads to greater node-variance loss.
7. Equal-strength propagation retains the observed topology
   sensitivity advantage.

## Reference comparison

At **72 edges**, depth **2**, alpha **0.50**:

| Metric | Result |
|---|---:|
| Global topology L2 | {primary_global:.6f} |
| Spatial topology L2 | {primary_spatial:.6f} |
| Relative L2 reduction | {primary_reduction:.2%} |
| Regions favoring spatial | {evidence["primary_positive_regions"]}/10 |
| Geometry-adjusted regions favoring spatial | {evidence["geometry_adjusted_spatial_positive_regions"]}/10 |
| Regions with lower spatial variance retention | {evidence["spatial_lower_variance_retention_regions"]}/10 |

These are descriptive within-acquisition results.

## Limitations

- Only one registered acquisition pair was investigated.
- Ten spatial regions are not verified independent specimens.
- Manual registration quality review remains unconfirmed.
- Graph-construction and smoothing effects remain interrelated.
- No downstream clinical labels or trained GNN have been evaluated.
- Novelty and broader generalization have not been established.

## Phase 4 direction

Develop and assess a stability-aware graph construction approach
that balances cross-scanner robustness with representation diversity.

Before making downstream performance claims, establish a defensible
training objective, non-collapse controls, baseline comparisons,
and independent evaluation where possible.

## Research preservation

- Frozen manifest SHA256: `{EXPECTED_HASH}`
- Phase 3B feature archive and experiment tables are unchanged.
- Previous research results remain historical records.
- Phase 3B is closed after Step 3B.28.
"""

MARKDOWN_PATH.write_text(
    markdown,
    encoding="utf-8",
)

print("\n" + "=" * 70)
print("PHASE 3B - FINAL CONSOLIDATION COMPLETED")
print("=" * 70)

print("Experiments audited:", len(INPUTS))
print("Primary regions:", 10)
print("Phase status: CLOSED")
print("Previous outputs: UNCHANGED")
print("\nSaved:")
print(REPORT_PATH)
print(MARKDOWN_PATH)

print("\nNEXT: PHASE 4 - MAIN RESEARCH")
