
"""
PHASE 3B.24
Multiregion Stability vs Oversmoothing Audit

Uses saved Phase 3B.23 propagation results.

Measures:
- AT2 and GT450 variance retention
- Cross-node cosine similarity changes
- Spatial versus global topology sensitivity
- Paired region-level differences

No models are run.
No images or feature archives are modified.
No biological independence is assumed.
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

INPUT = (
    TABLES / "plism_phase3b_propagation_per_region.csv"
)

MANIFEST = (
    BASE / "outputs/validated/"
    "phase3b_frozen_tile_manifest.csv"
)

OUT_DETAILS = (
    TABLES / "plism_phase3b_oversmoothing_details.csv"
)

OUT_SUMMARY = (
    TABLES / "plism_phase3b_oversmoothing_summary.csv"
)

OUT_PAIRED = (
    TABLES / "plism_phase3b_oversmoothing_paired.csv"
)

OUT_REPORT = (
    REPORTS / "24_multiregion_oversmoothing_audit.json"
)

EXPECTED_MANIFEST_HASH = (
    "69d11e1a0dd1107f2e6534632727e2c2291ff83d6b17"
    "cf953ed5056bea9020f9"
)

BUDGETS = [54, 72, 90]
DEPTHS = [0, 1, 2, 4, 8]
METHODS = ["global", "spatial"]


def sha256_file(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


print("=" * 70, flush=True)
print("PHASE 3B.24 - OVERSMOOTHING AUDIT", flush=True)
print("=" * 70, flush=True)


# ============================================================
# STEP 1: VERIFY INPUTS
# ============================================================

print("\n[1] Verifying previous experiment...", flush=True)

if sha256_file(MANIFEST) != EXPECTED_MANIFEST_HASH:
    raise RuntimeError("Frozen manifest hash mismatch.")

if not INPUT.exists():
    raise RuntimeError("Phase 3B.23 results missing.")

for path in (
    OUT_DETAILS,
    OUT_SUMMARY,
    OUT_PAIRED,
    OUT_REPORT,
):
    if path.exists():
        raise RuntimeError(
            f"Refusing to overwrite: {path}"
        )

df = pd.read_csv(INPUT)

required_columns = {
    "region_id",
    "graph_type",
    "edge_budget",
    "depth",
    "topology_only_l2",
    "natural_l2",
    "at2_node_variance",
    "gt450_node_variance",
    "at2_offdiag_cosine",
    "gt450_offdiag_cosine",
}

missing = required_columns - set(df.columns)

if missing:
    raise RuntimeError(
        f"Missing columns: {sorted(missing)}"
    )

if len(df) != 300:
    raise RuntimeError("Expected 300 experiment rows.")

if df.duplicated(
    ["region_id", "graph_type", "edge_budget", "depth"]
).any():
    raise RuntimeError("Duplicate experiment keys.")

regions = sorted(df["region_id"].unique())

if len(regions) != 10:
    raise RuntimeError("Expected ten regions.")

for region_id in regions:
    for method in METHODS:
        for budget in BUDGETS:
            subset = df[
                (df.region_id == region_id)
                & (df.graph_type == method)
                & (df.edge_budget == budget)
            ]

            if sorted(subset.depth.tolist()) != DEPTHS:
                raise RuntimeError(
                    "Incomplete depth series: "
                    f"{region_id}, {method}, {budget}"
                )

numeric_columns = [
    "topology_only_l2",
    "natural_l2",
    "at2_node_variance",
    "gt450_node_variance",
    "at2_offdiag_cosine",
    "gt450_offdiag_cosine",
]

if not np.isfinite(
    df[numeric_columns].to_numpy(dtype=float)
).all():
    raise RuntimeError("Non-finite measurements.")

if (
    df[["at2_node_variance", "gt450_node_variance"]]
    .to_numpy() < 0
).any():
    raise RuntimeError("Negative variance.")

print("Previous experiment: VERIFIED")
print("Regions:", len(regions))
print("Experiment records:", len(df))


# ============================================================
# STEP 2: COMPUTE RETENTION
# ============================================================

print("\n[2] Computing variance retention...", flush=True)

records = []

for (region, method, budget), group in df.groupby(
    ["region_id", "graph_type", "edge_budget"]
):

    group = group.sort_values("depth")

    baseline = group[group.depth == 0].iloc[0]

    initial_at2_variance = float(
        baseline.at2_node_variance
    )

    initial_gt450_variance = float(
        baseline.gt450_node_variance
    )

    if (
        initial_at2_variance <= 1e-12
        or initial_gt450_variance <= 1e-12
    ):
        raise RuntimeError(
            f"Invalid baseline variance: {region}"
        )

    if abs(float(baseline.topology_only_l2)) > 1e-8:
        raise RuntimeError(
            "Topology effect must be zero at depth 0."
        )

    for row in group.itertuples(index=False):

        retention_a = (
            float(row.at2_node_variance)
            / initial_at2_variance
        )

        retention_b = (
            float(row.gt450_node_variance)
            / initial_gt450_variance
        )

        cosine_change_a = (
            float(row.at2_offdiag_cosine)
            - float(baseline.at2_offdiag_cosine)
        )

        cosine_change_b = (
            float(row.gt450_offdiag_cosine)
            - float(baseline.gt450_offdiag_cosine)
        )

        records.append({
            "region_id": region,
            "graph_type": method,
            "edge_budget": int(budget),
            "depth": int(row.depth),
            "at2_variance_retention": retention_a,
            "gt450_variance_retention": retention_b,
            "mean_variance_retention": (
                retention_a + retention_b
            ) / 2,
            "at2_offdiag_cosine": float(
                row.at2_offdiag_cosine
            ),
            "gt450_offdiag_cosine": float(
                row.gt450_offdiag_cosine
            ),
            "at2_offdiag_cosine_change": cosine_change_a,
            "gt450_offdiag_cosine_change": cosine_change_b,
            "mean_offdiag_cosine_change": (
                cosine_change_a + cosine_change_b
            ) / 2,
            "topology_only_l2": float(
                row.topology_only_l2
            ),
            "natural_l2": float(row.natural_l2),
        })

details = pd.DataFrame(records)

if len(details) != 300:
    raise RuntimeError("Detail count mismatch.")


# ============================================================
# STEP 3: AGGREGATE RESULTS
# ============================================================

print("\n[3] Comparing stability and smoothing...", flush=True)

summary = (
    details.groupby(
        ["graph_type", "edge_budget", "depth"],
        as_index=False,
    )
    .agg(
        regions=("region_id", "nunique"),
        mean_variance_retention=(
            "mean_variance_retention", "mean"
        ),
        std_variance_retention=(
            "mean_variance_retention", "std"
        ),
        mean_cosine_change=(
            "mean_offdiag_cosine_change", "mean"
        ),
        mean_topology_l2=(
            "topology_only_l2", "mean"
        ),
        mean_natural_l2=("natural_l2", "mean"),
    )
)

pairs = []

for (region, budget, depth), group in details.groupby(
    ["region_id", "edge_budget", "depth"]
):

    g = group.set_index("graph_type")

    global_row = g.loc["global"]
    spatial_row = g.loc["spatial"]

    pairs.append({
        "region_id": region,
        "edge_budget": budget,
        "depth": depth,

        # Positive means spatial topology is
        # less sensitive to graph swapping.
        "topology_l2_advantage": (
            global_row.topology_only_l2
            - spatial_row.topology_only_l2
        ),

        # Negative means spatial graph retains
        # less node variance than global graph.
        "spatial_minus_global_retention": (
            spatial_row.mean_variance_retention
            - global_row.mean_variance_retention
        ),

        # Positive means spatial propagation
        # increases cross-node cosine more.
        "spatial_minus_global_cosine_change": (
            spatial_row.mean_offdiag_cosine_change
            - global_row.mean_offdiag_cosine_change
        ),

        "global_retention": (
            global_row.mean_variance_retention
        ),
        "spatial_retention": (
            spatial_row.mean_variance_retention
        ),
    })

paired = pd.DataFrame(pairs)

if len(paired) != 150:
    raise RuntimeError("Paired comparison count mismatch.")


# ============================================================
# STEP 4: PRINT KEY RESULTS
# ============================================================

print("\nDepth 2 results:", flush=True)

depth2 = summary[summary.depth == 2]

print(
    depth2[
        [
            "graph_type",
            "edge_budget",
            "mean_variance_retention",
            "mean_cosine_change",
            "mean_topology_l2",
            "mean_natural_l2",
        ]
    ].to_string(index=False)
)

print("\nPaired spatial-minus-global trade-offs at depth 2:")

for budget in BUDGETS:
    subset = paired[
        (paired.edge_budget == budget)
        & (paired.depth == 2)
    ]

    print(f"\nBudget {budget}:")
    print(
        "  Regions with lower spatial topology L2:",
        int((subset.topology_l2_advantage > 0).sum()),
        "/10",
    )
    print(
        "  Regions with lower spatial variance retention:",
        int(
            (subset.spatial_minus_global_retention < 0).sum()
        ),
        "/10",
    )
    print(
        "  Mean topology L2 advantage:",
        f"{subset.topology_l2_advantage.mean():.6f}",
    )
    print(
        "  Mean retention difference:",
        f"{subset.spatial_minus_global_retention.mean():.6f}",
    )
    print(
        "  Mean cosine-change difference:",
        f"{subset.spatial_minus_global_cosine_change.mean():.6f}",
    )

print("\nDepth progression at budget 72:")

print(
    summary[summary.edge_budget == 72][
        [
            "graph_type",
            "depth",
            "mean_variance_retention",
            "mean_cosine_change",
            "mean_topology_l2",
        ]
    ].to_string(index=False)
)


# ============================================================
# STEP 5: SAVE OUTPUTS
# ============================================================

print("\n[4] Saving research outputs...", flush=True)

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

details.to_csv(OUT_DETAILS, index=False)
summary.to_csv(OUT_SUMMARY, index=False)
paired.to_csv(OUT_PAIRED, index=False)

report = {
    "phase": "3B",
    "step": "3B.24",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "manifest_sha256": sha256_file(MANIFEST),
    "source_step23_csv_sha256": sha256_file(INPUT),
    "regions": len(regions),
    "edge_budgets": BUDGETS,
    "depths": DEPTHS,
    "detail_records": len(details),
    "paired_records": len(paired),
    "variance_retention": "V(depth)/V(0)",
    "offdiag_cosine_change":
        "Mean off-diagonal cosine(depth) minus baseline",
    "topology_advantage":
        "Global topology L2 minus spatial topology L2",
    "trained_gnn_used": False,
    "frozen_data_modified": False,
    "biological_independence_verified": False,
    "limitations": [
        "Variance retention is an oversmoothing proxy",
        "Cross-node cosine is not an accuracy metric",
        "Node variance can change due to normalization",
        "The ten regions come from one acquisition pair",
        "Spatial and global candidate pools differ",
        "No downstream task performance has been evaluated",
    ],
}

OUT_REPORT.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\n" + "=" * 70)
print("PHASE 3B.24 - OVERSMOOTHING AUDIT COMPLETED")
print("=" * 70)

print("Regions:", len(regions))
print("Detailed records:", len(details))
print("Paired comparison records:", len(paired))

print("\nSaved:")
print(OUT_DETAILS)
print(OUT_SUMMARY)
print(OUT_PAIRED)
print(OUT_REPORT)

print("\nOriginal data: UNCHANGED")
print("Trained GNN: NOT USED")
print("Biological independence: UNVERIFIED")
print("\nSTEP 3B.24 COMPLETED")
