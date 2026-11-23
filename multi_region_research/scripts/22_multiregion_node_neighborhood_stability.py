
"""
PHASE 3B.22
MULTI-REGION NODE NEIGHBORHOOD STABILITY

Graph construction matches Phase 3B.21:
    - Shared coordinate-only spatial MST
    - Scanner-specific cosine-ranked residual edges
    - Budgets: 54, 72, 90

Node metrics:
    - Residual-neighborhood Jaccard
    - Both-empty residual neighborhoods
    - Changed neighborhood count
    - Residual degree difference
    - Full graph degree difference

Graph metrics:
    - Mean residual edge distance
    - Fraction of long-range residual edges

No trained GNN.
No file modifications to previous experiments.
"""

from pathlib import Path
from itertools import combinations
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "multi_region_research"

MANIFEST = (
    BASE / "outputs/validated/"
    "phase3b_frozen_tile_manifest.csv"
)

FEATURES = (
    BASE / "outputs/features/"
    "plism_phase3b_dinov2_features.npz"
)

TABLES = BASE / "outputs/tables"
REPORTS = BASE / "outputs/reports"

NODE_PATH = (
    TABLES / "plism_phase3b_node_neighborhood_details.csv"
)

REGION_PATH = (
    TABLES / "plism_phase3b_node_neighborhood_per_region.csv"
)

SUMMARY_PATH = (
    TABLES / "plism_phase3b_node_neighborhood_summary.csv"
)

REPORT_PATH = (
    REPORTS / "22_multiregion_node_neighborhood_stability.json"
)

EXPECTED_HASH = (
    "69d11e1a0dd1107f2e6534632727e2c2291ff83d6b17"
    "cf953ed5056bea9020f9"
)

BUDGETS = (54, 72, 90)
METHODS = ("global", "spatial")

NODES = 36


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_similarity(features):
    features = np.asarray(features, dtype=np.float64)

    norms = np.linalg.norm(
        features, axis=1, keepdims=True
    )

    if np.any(norms <= 1e-12):
        raise RuntimeError("Zero-norm embedding.")

    normalized = features / norms

    return normalized @ normalized.T


def candidate_edges(rows, cols, method):
    edges = []

    for a, b in combinations(range(len(rows)), 2):

        dx = abs(int(rows[a]) - int(rows[b]))
        dy = abs(int(cols[a]) - int(cols[b]))

        if method == "spatial" and max(dx, dy) != 1:
            continue

        edges.append((a, b))

    return edges


def spatial_mst(rows, cols):
    """
    Same deterministic Kruskal MST as Step 3B.21.
    """

    edges = candidate_edges(rows, cols, "spatial")

    edges.sort(
        key=lambda e: (
            (int(rows[e[0]]) - int(rows[e[1]])) ** 2
            + (int(cols[e[0]]) - int(cols[e[1]])) ** 2,
            e[0],
            e[1],
        )
    )

    parent = list(range(len(rows)))
    sizes = [1] * len(rows)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a = find(a)
        b = find(b)

        if a == b:
            return False

        if sizes[a] < sizes[b]:
            a, b = b, a

        parent[b] = a
        sizes[a] += sizes[b]
        return True

    tree = set()

    for a, b in edges:
        if union(a, b):
            tree.add((a, b))

        if len(tree) == len(rows) - 1:
            break

    if len(tree) != len(rows) - 1:
        raise RuntimeError("MST construction failed.")

    return tree


def select_residual(similarity, candidates, backbone, count):
    pool = [
        edge
        for edge in candidates
        if edge not in backbone
    ]

    if count > len(pool):
        raise RuntimeError("Insufficient residual edges.")

    ranked = sorted(
        pool,
        key=lambda e: (
            -float(similarity[e[0], e[1]]),
            e[0],
            e[1],
        )
    )

    return set(ranked[:count])


def neighborhoods(edges, n):
    result = [set() for _ in range(n)]

    for a, b in edges:
        result[a].add(b)
        result[b].add(a)

    return result


def edge_distances(edges, rows, cols):
    return [
        float(np.hypot(
            int(rows[a]) - int(rows[b]),
            int(cols[a]) - int(cols[b]),
        ))
        for a, b in edges
    ]


print("=" * 70, flush=True)
print("PHASE 3B.22 - NODE NEIGHBORHOOD STABILITY", flush=True)
print("=" * 70, flush=True)

# ------------------------------------------------------------
# STEP 1: VALIDATE INPUTS
# ------------------------------------------------------------

print("\n[1] Validating research inputs...", flush=True)

if not MANIFEST.is_file() or not FEATURES.is_file():
    raise RuntimeError("Required input missing.")

if sha256_file(MANIFEST) != EXPECTED_HASH:
    raise RuntimeError("Frozen manifest hash mismatch.")

for path in (
    NODE_PATH, REGION_PATH, SUMMARY_PATH, REPORT_PATH
):
    if path.exists():
        raise RuntimeError(
            f"Refusing to overwrite previous output: {path}"
        )

manifest = pd.read_csv(
    MANIFEST,
    dtype={"tile_id": str, "region_id": str},
).sort_values(
    ["region_id", "level", "x", "y", "tile_id"]
).reset_index(drop=True)

with np.load(FEATURES, allow_pickle=False) as data:
    a = data["at2"].copy()
    b = data["gt450"].copy()
    tile_ids = data["tile_ids"].astype(str)
    region_ids = data["region_ids"].astype(str)
    rows = data["grid_rows"].astype(int)
    cols = data["grid_cols"].astype(int)

if a.shape != (360, 384) or b.shape != (360, 384):
    raise RuntimeError("Feature shape mismatch.")

if not np.isfinite(a).all() or not np.isfinite(b).all():
    raise RuntimeError("Non-finite features.")

for actual, expected in (
    (tile_ids, manifest["tile_id"].to_numpy(dtype=str)),
    (region_ids, manifest["region_id"].to_numpy(dtype=str)),
    (rows, manifest["x"].to_numpy(dtype=int)),
    (cols, manifest["y"].to_numpy(dtype=int)),
):
    if not np.array_equal(actual, expected):
        raise RuntimeError("Feature ordering mismatch.")

regions = sorted(set(region_ids))

if len(regions) != 10:
    raise RuntimeError("Expected ten regions.")

print("Frozen manifest: VERIFIED")
print("Feature archive: VERIFIED")
print("Regions:", len(regions))

# ------------------------------------------------------------
# STEP 2: CONSTRUCT GRAPHS AND MEASURE NODES
# ------------------------------------------------------------

print("\n[2] Measuring node neighborhoods...", flush=True)

node_records = []
region_records = []

for region_id in regions:

    indices = np.where(region_ids == region_id)[0]

    if len(indices) != NODES:
        raise RuntimeError("Unexpected region size.")

    r = rows[indices]
    c = cols[indices]
    ids = tile_ids[indices]

    expected_coords = {
        (x, y)
        for x in range(int(r.min()), int(r.min()) + 6)
        for y in range(int(c.min()), int(c.min()) + 6)
    }

    if set(zip(r.tolist(), c.tolist())) != expected_coords:
        raise RuntimeError(f"Invalid grid: {region_id}")

    backbone = spatial_mst(r, c)

    sim_a = normalized_similarity(a[indices])
    sim_b = normalized_similarity(b[indices])

    for method in METHODS:

        candidates = candidate_edges(r, c, method)

        for budget in BUDGETS:

            residual_count = budget - len(backbone)

            residual_a = select_residual(
                sim_a, candidates, backbone, residual_count
            )

            residual_b = select_residual(
                sim_b, candidates, backbone, residual_count
            )

            full_a = backbone | residual_a
            full_b = backbone | residual_b

            if len(full_a) != budget or len(full_b) != budget:
                raise RuntimeError("Invalid edge budget.")

            neighbors_a = neighborhoods(residual_a, NODES)
            neighbors_b = neighborhoods(residual_b, NODES)

            full_neighbors_a = neighborhoods(full_a, NODES)
            full_neighbors_b = neighborhoods(full_b, NODES)

            valid_node_jaccards = []
            both_empty_count = 0
            changed_count = 0
            degree_differences = []
            full_degree_differences = []

            for node in range(NODES):

                na = neighbors_a[node]
                nb = neighbors_b[node]

                union = na | nb
                shared = na & nb

                both_empty = len(union) == 0

                # Both-empty nodes are excluded from the
                # primary neighborhood Jaccard average.
                if both_empty:
                    score = np.nan
                    both_empty_count += 1
                else:
                    score = len(shared) / len(union)
                    valid_node_jaccards.append(score)

                changed = na != nb

                if changed:
                    changed_count += 1

                degree_difference = abs(len(na) - len(nb))

                full_degree_difference = abs(
                    len(full_neighbors_a[node])
                    - len(full_neighbors_b[node])
                )

                degree_differences.append(degree_difference)
                full_degree_differences.append(
                    full_degree_difference
                )

                node_records.append({
                    "region_id": region_id,
                    "graph_type": method,
                    "edge_budget": budget,
                    "tile_id": str(ids[node]),
                    "node_index": node,
                    "at2_residual_degree": len(na),
                    "gt450_residual_degree": len(nb),
                    "at2_full_degree": len(
                        full_neighbors_a[node]
                    ),
                    "gt450_full_degree": len(
                        full_neighbors_b[node]
                    ),
                    "residual_neighbors_shared": len(shared),
                    "residual_neighbors_union": len(union),
                    "residual_neighborhood_jaccard": score,
                    "both_residual_empty": both_empty,
                    "residual_neighborhood_changed": changed,
                    "residual_degree_absolute_difference":
                        degree_difference,
                    "full_degree_absolute_difference":
                        full_degree_difference,
                })

            distances_a = edge_distances(
                residual_a, r, c
            )

            distances_b = edge_distances(
                residual_b, r, c
            )

            all_distances = distances_a + distances_b

            # Long range is defined relative to the
            # immediate 8-neighbor grid, not physical units.
            long_fraction = np.mean(
                np.asarray(all_distances) > np.sqrt(2) + 1e-9
            )

            region_records.append({
                "region_id": region_id,
                "graph_type": method,
                "edge_budget": budget,
                "nodes": NODES,
                "residual_edges_per_graph": residual_count,
                "mean_residual_node_jaccard": (
                    float(np.mean(valid_node_jaccards))
                    if valid_node_jaccards
                    else np.nan
                ),
                "nodes_in_primary_jaccard_average":
                    len(valid_node_jaccards),
                "both_empty_nodes": both_empty_count,
                "changed_neighborhood_nodes": changed_count,
                "changed_neighborhood_fraction":
                    changed_count / NODES,
                "mean_residual_degree_difference":
                    float(np.mean(degree_differences)),
                "mean_full_degree_difference":
                    float(np.mean(full_degree_differences)),
                "mean_residual_edge_distance":
                    float(np.mean(all_distances)),
                "long_range_residual_edge_fraction":
                    float(long_fraction),
            })

    print(f"  {region_id}: completed", flush=True)

# ------------------------------------------------------------
# STEP 3: AGGREGATE RESULTS
# ------------------------------------------------------------

print("\n[3] Summarizing results...", flush=True)

nodes_df = pd.DataFrame(node_records)
regions_df = pd.DataFrame(region_records)

if len(nodes_df) != 10 * 2 * 3 * 36:
    raise RuntimeError("Unexpected node record count.")

if len(regions_df) != 10 * 2 * 3:
    raise RuntimeError("Unexpected region record count.")

summary = (
    regions_df.groupby(
        ["graph_type", "edge_budget"],
        as_index=False
    )
    .agg(
        regions=("region_id", "nunique"),
        mean_node_jaccard=(
            "mean_residual_node_jaccard", "mean"
        ),
        std_node_jaccard=(
            "mean_residual_node_jaccard", "std"
        ),
        mean_both_empty_nodes=("both_empty_nodes", "mean"),
        mean_changed_fraction=(
            "changed_neighborhood_fraction", "mean"
        ),
        mean_residual_degree_difference=(
            "mean_residual_degree_difference", "mean"
        ),
        mean_edge_distance=(
            "mean_residual_edge_distance", "mean"
        ),
        mean_long_range_fraction=(
            "long_range_residual_edge_fraction", "mean"
        ),
    )
)

print("\nNeighborhood stability summary:")
print(summary.to_string(index=False))

for budget in BUDGETS:

    subset = regions_df[
        regions_df["edge_budget"] == budget
    ]

    pivot = subset.pivot(
        index="region_id",
        columns="graph_type",
        values="mean_residual_node_jaccard",
    )

    differences = pivot["spatial"] - pivot["global"]

    print(
        f"\nBudget {budget}: "
        f"spatial-minus-global node Jaccard"
    )
    print(f"  Mean: {differences.mean():.6f}")
    print(f"  Minimum: {differences.min():.6f}")
    print(f"  Maximum: {differences.max():.6f}")
    print(f"  Positive regions: {(differences > 0).sum()}/10")

# ------------------------------------------------------------
# STEP 4: SAVE
# ------------------------------------------------------------

print("\n[4] Saving results...", flush=True)

nodes_df.to_csv(NODE_PATH, index=False)
regions_df.to_csv(REGION_PATH, index=False)
summary.to_csv(SUMMARY_PATH, index=False)

report = {
    "phase": "3B",
    "step": "3B.22",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "manifest_sha256": sha256_file(MANIFEST),
    "feature_sha256": sha256_file(FEATURES),
    "regions": 10,
    "nodes_per_region": NODES,
    "graph_types": list(METHODS),
    "edge_budgets": list(BUDGETS),
    "node_records": len(nodes_df),
    "region_records": len(regions_df),
    "shared_backbone_edges": 35,
    "both_empty_nodes_excluded_from_primary_jaccard": True,
    "long_range_threshold_grid_units":
        float(np.sqrt(2)),
    "trained_gnn_used": False,
    "images_modified": False,
    "frozen_manifest_modified": False,
    "biological_independence_verified": False,
    "limitations": [
        "A shared backbone is used in both scanner graphs",
        "Residual neighborhood Jaccard excludes both-empty nodes",
        "Global and spatial candidate pools have different sizes",
        "Spatial graph residual edges are local by definition",
        "Ten regions do not establish biological independence",
        "Node observations within a region are not independent",
        "No learned GNN representation has been evaluated",
    ],
}

REPORT_PATH.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("PHASE 3B.22 - NODE STABILITY COMPLETED")
print("=" * 70)

print("Regions analyzed:", 10)
print("Graph configurations:", 60)
print("Node records:", len(nodes_df))

print("\nSaved:")
print(NODE_PATH)
print(REGION_PATH)
print(SUMMARY_PATH)
print(REPORT_PATH)

print("\nOriginal images: UNCHANGED")
print("Frozen manifest: UNCHANGED")
print("Trained GNN: NOT USED")
print("Biological independence: UNVERIFIED")

