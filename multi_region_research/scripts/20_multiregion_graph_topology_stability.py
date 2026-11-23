
"""
PHASE 3B.20
MULTI-REGION CROSS-SCANNER GRAPH TOPOLOGY STABILITY

Primary comparison:
    Global feature similarity graph
    Spatially constrained feature similarity graph

Controls:
    Same nodes, edge count, feature encoder,
    scanner pairing and graph budget.

Regions:
    10 regions, 36 nodes per region

Edge budgets:
    36, 54, 72, 90

Measures:
    Edge Jaccard
    Observed shared edges
    Expected random shared edges
    Chance-adjusted edge agreement
    Connected components
    Isolated nodes
    Largest component size

No trained GNN.
No connectivity-preserving backbone yet.
No modifications to frozen data.
"""

from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "multi_region_research"

FEATURE_FILE = (
    BASE / "outputs/features/"
    "plism_phase3b_dinov2_features.npz"
)

MANIFEST_FILE = (
    BASE / "outputs/validated/"
    "phase3b_frozen_tile_manifest.csv"
)

TABLES = BASE / "outputs/tables"
REPORTS = BASE / "outputs/reports"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

DETAIL_PATH = (
    TABLES / "plism_phase3b_graph_stability_per_region.csv"
)

SUMMARY_PATH = (
    TABLES / "plism_phase3b_graph_stability_summary.csv"
)

REPORT_PATH = (
    REPORTS / "20_multiregion_graph_stability.json"
)

EXPECTED_MANIFEST_SHA256 = (
    "69d11e1a0dd1107f2e6534632727e2c2291ff83d6b17"
    "cf953ed5056bea9020f9"
)

EDGE_BUDGETS = [36, 54, 72, 90]
GRAPH_TYPES = ("global", "spatial")
SCANNERS = ("AT2", "GT450")

EXPECTED_REGIONS = 10
NODES_PER_REGION = 36

# Never overwrite previous experiment results.
for path in (DETAIL_PATH, SUMMARY_PATH, REPORT_PATH):
    if path.exists():
        raise RuntimeError(
            f"Output already exists: {path}"
        )


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def normalize_rows(features):
    norms = np.linalg.norm(
        features,
        axis=1,
        keepdims=True,
    )

    if np.any(norms < 1e-12):
        raise RuntimeError(
            "Zero-norm feature vector detected."
        )

    return features / norms


def build_candidates(rows, cols, graph_type):
    edges = []

    n = len(rows)

    for i, j in combinations(range(n), 2):

        if graph_type == "spatial":

            dx = abs(int(rows[i]) - int(rows[j]))
            dy = abs(int(cols[i]) - int(cols[j]))

            if not (
                max(dx, dy) == 1
                and dx <= 1
                and dy <= 1
            ):
                continue

        edges.append((i, j))

    return edges


def select_top_edges(features, candidates, budget):
    """
    Deterministic Top-M graph.

    Similarities are computed on L2-normalized
    features. Ties are broken by node indices.
    """

    if budget > len(candidates):
        raise RuntimeError(
            "Edge budget exceeds candidate pool."
        )

    normalized = normalize_rows(
        features.astype(np.float64)
    )

    similarity = normalized @ normalized.T

    ranked = sorted(
        candidates,
        key=lambda edge: (
            -float(similarity[edge[0], edge[1]]),
            edge[0],
            edge[1],
        )
    )

    return set(ranked[:budget])


def graph_connectivity(node_count, edges):
    adjacency = [
        set() for _ in range(node_count)
    ]

    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    isolated = sum(
        len(neighbors) == 0
        for neighbors in adjacency
    )

    visited = set()
    component_sizes = []

    for node in range(node_count):

        if node in visited:
            continue

        stack = [node]
        visited.add(node)

        size = 0

        while stack:
            current = stack.pop()
            size += 1

            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        component_sizes.append(size)

    return {
        "components": len(component_sizes),
        "isolated_nodes": isolated,
        "largest_component": max(component_sizes),
    }


def stability_statistics(edges_a, edges_b, pool_size, budget):
    intersection = len(edges_a & edges_b)
    union = len(edges_a | edges_b)

    jaccard = (
        intersection / union
        if union else 1.0
    )

    # Under two independent uniform samples
    # of 'budget' edges from the same pool.
    expected_intersection = (
        budget * budget / pool_size
    )

    observed_agreement = intersection / budget
    random_agreement = budget / pool_size

    denominator = 1.0 - random_agreement

    adjusted = (
        (observed_agreement - random_agreement)
        / denominator
        if denominator > 0 else float("nan")
    )

    return {
        "shared_edges": intersection,
        "union_edges": union,
        "edge_jaccard": jaccard,
        "observed_edge_agreement": observed_agreement,
        "expected_random_shared_edges":
            expected_intersection,
        "chance_adjusted_agreement": adjusted,
    }


# ============================================================
# STEP 1: VERIFY INPUTS
# ============================================================

print("=" * 70, flush=True)
print("PHASE 3B.20 - MULTI-REGION GRAPH STABILITY", flush=True)
print("=" * 70, flush=True)

print("\n[1] Verifying frozen inputs...", flush=True)

if not MANIFEST_FILE.is_file():
    raise RuntimeError("Frozen manifest missing.")

if not FEATURE_FILE.is_file():
    raise RuntimeError("Feature archive missing.")

manifest_hash = sha256_file(MANIFEST_FILE)

if manifest_hash != EXPECTED_MANIFEST_SHA256:
    raise RuntimeError(
        "Frozen manifest fingerprint mismatch."
    )

manifest = pd.read_csv(
    MANIFEST_FILE,
    dtype={
        "region_id": str,
        "tile_id": str,
    }
)

manifest = manifest.sort_values(
    ["region_id", "level", "x", "y", "tile_id"]
).reset_index(drop=True)

with np.load(
    FEATURE_FILE,
    allow_pickle=False,
) as data:

    required = {
        "at2", "gt450", "tile_ids",
        "grid_rows", "grid_cols", "region_ids",
    }

    if not required.issubset(set(data.files)):
        raise RuntimeError(
            "Feature archive missing required arrays."
        )

    at2 = data["at2"].copy()
    gt450 = data["gt450"].copy()

    tile_ids = data["tile_ids"].astype(str)
    region_ids = data["region_ids"].astype(str)

    rows = data["grid_rows"].astype(np.int64)
    cols = data["grid_cols"].astype(np.int64)

if at2.shape != (360, 384):
    raise RuntimeError("Unexpected AT2 features.")

if gt450.shape != (360, 384):
    raise RuntimeError("Unexpected GT450 features.")

if not np.isfinite(at2).all():
    raise RuntimeError("Non-finite AT2 values.")

if not np.isfinite(gt450).all():
    raise RuntimeError("Non-finite GT450 values.")

if not np.array_equal(
    tile_ids, manifest["tile_id"].to_numpy()
):
    raise RuntimeError("Tile ordering mismatch.")

if not np.array_equal(
    region_ids, manifest["region_id"].to_numpy()
):
    raise RuntimeError("Region ordering mismatch.")

if not np.array_equal(
    rows, manifest["x"].to_numpy()
):
    raise RuntimeError("Grid row mismatch.")

if not np.array_equal(
    cols, manifest["y"].to_numpy()
):
    raise RuntimeError("Grid column mismatch.")

unique_regions = sorted(set(region_ids))

if len(unique_regions) != EXPECTED_REGIONS:
    raise RuntimeError(
        "Expected exactly ten regions."
    )

print("Frozen manifest: VERIFIED")
print("Feature archive: VERIFIED")
print("Matched positions: 360")
print("Regions:", len(unique_regions))


# ============================================================
# STEP 2: REGION GRAPH EXPERIMENTS
# ============================================================

print("\n[2] Running controlled graph comparisons...", flush=True)

records = []

for region_id in unique_regions:

    indices = np.where(
        region_ids == region_id
    )[0]

    if len(indices) != NODES_PER_REGION:
        raise RuntimeError(
            f"Unexpected node count: {region_id}"
        )

    region_rows = rows[indices]
    region_cols = cols[indices]

    if len(set(zip(region_rows, region_cols))) != 36:
        raise RuntimeError(
            f"Duplicate grid coordinates: {region_id}"
        )

    # Validate complete 6x6 geometry.
    unique_x = np.unique(region_rows)
    unique_y = np.unique(region_cols)

    if (
        len(unique_x) != 6
        or len(unique_y) != 6
        or not np.array_equal(
            unique_x,
            np.arange(unique_x.min(), unique_x.min() + 6)
        )
        or not np.array_equal(
            unique_y,
            np.arange(unique_y.min(), unique_y.min() + 6)
        )
    ):
        raise RuntimeError(
            f"Invalid grid geometry: {region_id}"
        )

    all_positions = {
        (int(x), int(y))
        for x, y in zip(region_rows, region_cols)
    }

    if all_positions != {
        (int(x), int(y))
        for x in unique_x
        for y in unique_y
    }:
        raise RuntimeError(
            f"Incomplete 6x6 grid: {region_id}"
        )

    region_features = {
        "AT2": at2[indices],
        "GT450": gt450[indices],
    }

    a_normalized = normalize_rows(
        region_features["AT2"].astype(np.float64)
    )

    b_normalized = normalize_rows(
        region_features["GT450"].astype(np.float64)
    )

    paired_cosine = np.sum(
        a_normalized * b_normalized,
        axis=1
    )

    print(
        f"\n{region_id}: "
        f"mean paired cosine={paired_cosine.mean():.6f}",
        flush=True
    )

    for graph_type in GRAPH_TYPES:

        candidates = build_candidates(
            region_rows, region_cols, graph_type
        )

        pool_size = len(candidates)

        print(
            f"  {graph_type}: "
            f"{pool_size} candidate edges",
            flush=True
        )

        for budget in EDGE_BUDGETS:

            edges_a = select_top_edges(
                region_features["AT2"],
                candidates,
                budget,
            )

            edges_b = select_top_edges(
                region_features["GT450"],
                candidates,
                budget,
            )

            statistics = stability_statistics(
                edges_a,
                edges_b,
                pool_size,
                budget,
            )

            topology_a = graph_connectivity(
                NODES_PER_REGION, edges_a
            )

            topology_b = graph_connectivity(
                NODES_PER_REGION, edges_b
            )

            records.append({
                "region_id": region_id,
                "graph_type": graph_type,
                "edge_budget": budget,
                "node_count": NODES_PER_REGION,
                "candidate_edge_count": pool_size,
                "mean_paired_feature_cosine":
                    float(paired_cosine.mean()),
                **statistics,
                "at2_components":
                    topology_a["components"],
                "gt450_components":
                    topology_b["components"],
                "at2_isolated_nodes":
                    topology_a["isolated_nodes"],
                "gt450_isolated_nodes":
                    topology_b["isolated_nodes"],
                "at2_largest_component":
                    topology_a["largest_component"],
                "gt450_largest_component":
                    topology_b["largest_component"],
            })


# ============================================================
# STEP 3: SUMMARY ACROSS REGIONS
# ============================================================

print("\n[3] Aggregating region-level results...", flush=True)

details = pd.DataFrame(records)

if len(details) != 10 * 2 * 4:
    raise RuntimeError(
        "Expected exactly 80 experiment rows."
    )

summary = (
    details.groupby(
        ["graph_type", "edge_budget"],
        as_index=False
    )
    .agg(
        region_count=("region_id", "nunique"),
        mean_jaccard=("edge_jaccard", "mean"),
        std_jaccard=("edge_jaccard", "std"),
        min_jaccard=("edge_jaccard", "min"),
        max_jaccard=("edge_jaccard", "max"),
        mean_chance_adjusted=(
            "chance_adjusted_agreement", "mean"
        ),
        mean_at2_components=(
            "at2_components", "mean"
        ),
        mean_gt450_components=(
            "gt450_components", "mean"
        ),
        mean_at2_isolated=(
            "at2_isolated_nodes", "mean"
        ),
        mean_gt450_isolated=(
            "gt450_isolated_nodes", "mean"
        ),
    )
)

# Paired comparison within each region and budget.
wide = details.pivot(
    index=["region_id", "edge_budget"],
    columns="graph_type",
    values="edge_jaccard"
)

wide["spatial_minus_global"] = (
    wide["spatial"] - wide["global"]
)

differences = (
    wide.reset_index()
    .groupby("edge_budget")["spatial_minus_global"]
    .agg(["mean", "std", "min", "max"])
)

print("\nSpatial minus global Jaccard by budget:")
print(differences.to_string())

print("\nGraph stability summary:")
print(
    summary[
        [
            "graph_type",
            "edge_budget",
            "region_count",
            "mean_jaccard",
            "std_jaccard",
            "mean_chance_adjusted",
            "mean_at2_components",
            "mean_gt450_components",
        ]
    ].to_string(index=False)
)


# ============================================================
# STEP 4: SAVE RESULTS
# ============================================================

print("\n[4] Saving experimental outputs...", flush=True)

details.to_csv(DETAIL_PATH, index=False)
summary.to_csv(SUMMARY_PATH, index=False)

report = {
    "phase": "3B",
    "step": "3B.20",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "manifest_sha256": manifest_hash,
    "feature_archive_sha256":
        sha256_file(FEATURE_FILE),
    "encoder": "facebook/dinov2-small",
    "embedding": "raw CLS token",
    "normalization_for_graph_similarity":
        "L2 normalization",
    "regions": len(unique_regions),
    "nodes_per_region": NODES_PER_REGION,
    "graph_types": list(GRAPH_TYPES),
    "edge_budgets": EDGE_BUDGETS,
    "experiment_rows": len(details),
    "chance_adjustment": (
        "Observed common-edge fraction adjusted "
        "against independent uniform edge selection "
        "from the same candidate pool."
    ),
    "connectivity_enforced": False,
    "trained_gnn_used": False,
    "images_modified": False,
    "frozen_manifest_modified": False,
    "biological_independence_verified": False,
    "manual_visual_review_completed": False,
    "limitations": [
        "Ten regions originate from one acquisition pair",
        "Spatial candidate pools are structurally restricted",
        "Chance adjustment assumes uniform random edge sets",
        "Top-M selection may create disconnected graphs",
        "Descriptive statistics do not establish independence",
        "No trained GNN performance has been evaluated",
    ],
}

REPORT_PATH.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3B.20 - GRAPH STABILITY COMPLETED")
print("=" * 70)

print("Regions:", len(unique_regions))
print("Nodes per region:", NODES_PER_REGION)
print("Graph methods:", len(GRAPH_TYPES))
print("Edge budgets:", EDGE_BUDGETS)
print("Experiment records:", len(details))

print("\nSaved:")
print(DETAIL_PATH)
print(SUMMARY_PATH)
print(REPORT_PATH)

print("\nTrained GNN: NOT USED")
print("Connectivity enforced: NO")
print("Frozen manifest: UNCHANGED")
print("Biological independence: UNVERIFIED")
print("\nSTEP 3B.20 COMPLETED")
