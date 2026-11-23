
"""
PHASE 3B.21
MULTI-REGION CONNECTIVITY-PRESERVING GRAPH STABILITY

Reuses frozen Phase 3B DINOv2 features.

For each region:
    - 36 nodes
    - Shared deterministic spatial MST (35 edges)
    - Global or spatial residual Top-M edges
    - Total edge budgets: 54, 72, 90

Metrics:
    Full edge Jaccard
    Residual edge Jaccard
    Residual chance-adjusted agreement
    Node connectivity and isolated nodes
    Paired spatial-minus-global differences

All selected graphs must be connected.

No trained GNN.
No original data modifications.
"""

from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations
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

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

DETAIL_PATH = (
    TABLES / "plism_phase3b_connected_graph_per_region.csv"
)
SUMMARY_PATH = (
    TABLES / "plism_phase3b_connected_graph_summary.csv"
)
DIFFERENCE_PATH = (
    TABLES / "plism_phase3b_connected_graph_paired_differences.csv"
)
REPORT_PATH = (
    REPORTS / "21_multiregion_connected_graph_stability.json"
)

EXPECTED_HASH = (
    "69d11e1a0dd1107f2e6534632727e2c2291ff83d6b17"
    "cf953ed5056bea9020f9"
)

BUDGETS = (54, 72, 90)
GRAPH_TYPES = ("global", "spatial")

NODES = 36
BACKBONE_EDGES = NODES - 1

for output in (
    DETAIL_PATH,
    SUMMARY_PATH,
    DIFFERENCE_PATH,
    REPORT_PATH,
):
    if output.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing output: {output}"
        )


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_similarity(x):
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)

    if np.any(norms <= 1e-12):
        raise RuntimeError("Zero-norm embedding.")

    x = x / norms
    return x @ x.T


def candidate_edges(rows, cols, method):
    result = []

    for a, b in combinations(range(len(rows)), 2):
        dx = abs(int(rows[a]) - int(rows[b]))
        dy = abs(int(cols[a]) - int(cols[b]))

        if method == "spatial" and max(dx, dy) != 1:
            continue

        result.append((a, b))

    return result


def spatial_mst(rows, cols):
    """
    Deterministic Kruskal spanning tree.

    Uses grid distances only, never scanner features.
    Identical backbone for AT2 and GT450.
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
    size = [1] * len(rows)

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

        if size[a] < size[b]:
            a, b = b, a

        parent[b] = a
        size[a] += size[b]
        return True

    tree = set()

    for a, b in edges:
        if union(a, b):
            tree.add((a, b))

        if len(tree) == len(rows) - 1:
            break

    if len(tree) != len(rows) - 1:
        raise RuntimeError("Spatial MST disconnected.")

    return tree


def select_residual(similarity, candidates, backbone, count):
    pool = [e for e in candidates if e not in backbone]

    if count > len(pool):
        raise RuntimeError("Residual budget exceeds pool.")

    ranked = sorted(
        pool,
        key=lambda e: (
            -float(similarity[e[0], e[1]]),
            e[0],
            e[1],
        )
    )

    return set(ranked[:count]), len(pool)


def graph_connectivity(n, edges):
    adjacency = [set() for _ in range(n)]

    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    visited = set()
    components = 0

    for start in range(n):
        if start in visited:
            continue

        components += 1
        stack = [start]
        visited.add(start)

        while stack:
            node = stack.pop()

            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

    isolated = sum(not x for x in adjacency)

    return components, isolated


def jaccard(a, b):
    union = len(a | b)
    return len(a & b) / union if union else 1.0


print("=" * 70, flush=True)
print("PHASE 3B.21 - CONNECTED GRAPH STABILITY", flush=True)
print("=" * 70, flush=True)

print("\n[1] Validating frozen inputs...", flush=True)

if not MANIFEST.is_file() or not FEATURES.is_file():
    raise RuntimeError("Missing frozen inputs.")

if sha256_file(MANIFEST) != EXPECTED_HASH:
    raise RuntimeError("Manifest SHA256 mismatch.")

manifest = pd.read_csv(MANIFEST).sort_values(
    ["region_id", "level", "x", "y", "tile_id"]
).reset_index(drop=True)

with np.load(FEATURES, allow_pickle=False) as data:
    at2 = data["at2"].copy()
    gt450 = data["gt450"].copy()
    tile_ids = data["tile_ids"].astype(str)
    region_ids = data["region_ids"].astype(str)
    rows = data["grid_rows"].astype(int)
    cols = data["grid_cols"].astype(int)

if at2.shape != (360, 384) or gt450.shape != (360, 384):
    raise RuntimeError("Unexpected feature dimensions.")

if not np.isfinite(at2).all() or not np.isfinite(gt450).all():
    raise RuntimeError("Non-finite feature values.")

for saved, expected in (
    (tile_ids, manifest["tile_id"].to_numpy(dtype=str)),
    (region_ids, manifest["region_id"].to_numpy(dtype=str)),
    (rows, manifest["x"].to_numpy(dtype=int)),
    (cols, manifest["y"].to_numpy(dtype=int)),
):
    if not np.array_equal(saved, expected):
        raise RuntimeError("Feature/manifest ordering mismatch.")

regions = sorted(set(region_ids))

if len(regions) != 10:
    raise RuntimeError("Expected ten regions.")

print("Manifest: VERIFIED")
print("Feature ordering: VERIFIED")
print("Regions:", len(regions))

print("\n[2] Constructing connected graphs...", flush=True)

records = []

for region_id in regions:
    idx = np.where(region_ids == region_id)[0]

    if len(idx) != NODES:
        raise RuntimeError("Unexpected node count.")

    r = rows[idx]
    c = cols[idx]

    coords = set(zip(r.tolist(), c.tolist()))

    if (
        len(coords) != 36
        or len(set(r)) != 6
        or len(set(c)) != 6
        or coords != {
            (x, y)
            for x in range(min(r), min(r) + 6)
            for y in range(min(c), min(c) + 6)
        }
    ):
        raise RuntimeError(f"Invalid grid: {region_id}")

    backbone = spatial_mst(r, c)

    if len(backbone) != BACKBONE_EDGES:
        raise RuntimeError("Invalid backbone.")

    similarities = {
        "AT2": normalized_similarity(at2[idx]),
        "GT450": normalized_similarity(gt450[idx]),
    }

    print(
        f"\n{region_id}: backbone={len(backbone)}",
        flush=True,
    )

    for method in GRAPH_TYPES:
        candidates = candidate_edges(r, c, method)

        for budget in BUDGETS:
            residual_count = budget - BACKBONE_EDGES

            residual_a, pool_a = select_residual(
                similarities["AT2"],
                candidates,
                backbone,
                residual_count,
            )

            residual_b, pool_b = select_residual(
                similarities["GT450"],
                candidates,
                backbone,
                residual_count,
            )

            if pool_a != pool_b:
                raise RuntimeError("Candidate pool mismatch.")

            full_a = backbone | residual_a
            full_b = backbone | residual_b

            if len(full_a) != budget or len(full_b) != budget:
                raise RuntimeError("Edge budget mismatch.")

            components_a, isolated_a = graph_connectivity(
                NODES, full_a
            )

            components_b, isolated_b = graph_connectivity(
                NODES, full_b
            )

            if (
                components_a != 1
                or components_b != 1
                or isolated_a != 0
                or isolated_b != 0
            ):
                raise RuntimeError("Graph is not connected.")

            shared_residual = len(residual_a & residual_b)

            # Random-overlap reference for independently
            # selected residual sets from the same pool.
            random_fraction = residual_count / pool_a
            observed_fraction = shared_residual / residual_count

            adjusted = (
                (observed_fraction - random_fraction)
                / (1 - random_fraction)
            )

            records.append({
                "region_id": region_id,
                "graph_type": method,
                "edge_budget": budget,
                "nodes": NODES,
                "backbone_edges": len(backbone),
                "residual_edges": residual_count,
                "residual_candidate_pool": pool_a,
                "full_edge_jaccard": jaccard(full_a, full_b),
                "residual_edge_jaccard": jaccard(
                    residual_a, residual_b
                ),
                "shared_residual_edges": shared_residual,
                "expected_random_shared_residual":
                    residual_count ** 2 / pool_a,
                "residual_chance_adjusted": adjusted,
                "at2_components": components_a,
                "gt450_components": components_b,
                "at2_isolated_nodes": isolated_a,
                "gt450_isolated_nodes": isolated_b,
            })

    print("  Connected graph comparisons: PASSED")


print("\n[3] Aggregating results...", flush=True)

details = pd.DataFrame(records)

if len(details) != 10 * 2 * 3:
    raise RuntimeError("Unexpected experiment count.")

summary = (
    details.groupby(["graph_type", "edge_budget"], as_index=False)
    .agg(
        regions=("region_id", "nunique"),
        mean_full_jaccard=("full_edge_jaccard", "mean"),
        mean_residual_jaccard=("residual_edge_jaccard", "mean"),
        std_residual_jaccard=("residual_edge_jaccard", "std"),
        mean_chance_adjusted=(
            "residual_chance_adjusted", "mean"
        ),
        mean_shared_residual=("shared_residual_edges", "mean"),
        mean_at2_components=("at2_components", "mean"),
        mean_gt450_components=("gt450_components", "mean"),
    )
)

difference_rows = []

for budget in BUDGETS:
    subset = details[details["edge_budget"] == budget]

    pivot = subset.pivot(
        index="region_id",
        columns="graph_type",
        values=["residual_edge_jaccard", "residual_chance_adjusted"],
    )

    for region_id in pivot.index:
        difference_rows.append({
            "region_id": region_id,
            "edge_budget": budget,
            "spatial_minus_global_residual_jaccard": (
                pivot.loc[
                    region_id,
                    ("residual_edge_jaccard", "spatial")
                ]
                - pivot.loc[
                    region_id,
                    ("residual_edge_jaccard", "global")
                ]
            ),
            "spatial_minus_global_chance_adjusted": (
                pivot.loc[
                    region_id,
                    ("residual_chance_adjusted", "spatial")
                ]
                - pivot.loc[
                    region_id,
                    ("residual_chance_adjusted", "global")
                ]
            ),
        })

differences = pd.DataFrame(difference_rows)

print("\nConnected graph summary:")
print(summary.to_string(index=False))

print("\nSpatial minus global residual differences:")
print(
    differences.groupby("edge_budget")[
        [
            "spatial_minus_global_residual_jaccard",
            "spatial_minus_global_chance_adjusted",
        ]
    ].agg(["mean", "min", "max"]).to_string()
)

print("\n[4] Saving outputs...", flush=True)

details.to_csv(DETAIL_PATH, index=False)
summary.to_csv(SUMMARY_PATH, index=False)
differences.to_csv(DIFFERENCE_PATH, index=False)

report = {
    "phase": "3B",
    "step": "3B.21",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "manifest_sha256": sha256_file(MANIFEST),
    "feature_archive_sha256": sha256_file(FEATURES),
    "regions": len(regions),
    "nodes_per_region": NODES,
    "shared_spatial_backbone_edges": BACKBONE_EDGES,
    "edge_budgets": list(BUDGETS),
    "comparison_records": len(details),
    "all_graphs_connected": True,
    "spatial_residual_pool_size": 110 - BACKBONE_EDGES,
    "global_residual_pool_size": 630 - BACKBONE_EDGES,
    "method": (
        "Shared coordinate-only spatial MST followed by "
        "scanner-specific top-cosine residual edge selection."
    ),
    "chance_adjustment": (
        "Residual edge agreement adjusted against independent "
        "uniform selection from each residual candidate pool."
    ),
    "trained_gnn_used": False,
    "biological_independence_verified": False,
    "manual_registration_review_completed": False,
    "limitations": [
        "Shared backbone artificially raises full-edge stability",
        "Residual-edge comparisons are therefore primary",
        "Global and spatial residual pools differ in size",
        "Uniform random overlap is not a geometry-matched null model",
        "Ten regions are not verified independent specimens",
        "Connectivity alone does not establish representation robustness",
    ],
}

REPORT_PATH.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("PHASE 3B.21 - CONNECTED GRAPH ANALYSIS COMPLETED")
print("=" * 70)

print("Regions:", len(regions))
print("Experiment records:", len(details))
print("Shared backbone edges:", BACKBONE_EDGES)
print("All graphs connected: YES")
print("All graphs without isolated nodes: YES")

print("\nSaved:")
print(DETAIL_PATH)
print(SUMMARY_PATH)
print(DIFFERENCE_PATH)
print(REPORT_PATH)

print("\nTrained GNN: NOT USED")
print("Frozen data: UNCHANGED")
print("Biological independence: UNVERIFIED")
