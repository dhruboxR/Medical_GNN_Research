
"""
PHASE 3B.23
Cross-scanner graph propagation sensitivity

Reproduces the connected graph protocol of Phase 3B.21:
  - 36 nodes per region
  - Coordinate-only shared MST of 35 edges
  - Scanner-specific top-cosine residual edges
  - Global and spatial graph construction
  - Total edge budgets 54, 72, 90

Tests:
  - Natural cross-scanner representation change
  - Topology-only change with fixed features
  - Feature-only change with fixed topology
  - Oversmoothing diagnostics

Propagation:
  H(k) = S^k X
  S = D^-1/2 (A + I) D^-1/2

No learned GNN or labels.
No existing results are overwritten.
"""

from pathlib import Path
from itertools import combinations
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

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
    TABLES /
    "plism_phase3b_propagation_per_region.csv"
)

SUMMARY_PATH = (
    TABLES /
    "plism_phase3b_propagation_summary.csv"
)

DIFFERENCE_PATH = (
    TABLES /
    "plism_phase3b_propagation_paired_differences.csv"
)

REPORT_PATH = (
    REPORTS /
    "23_multiregion_graph_propagation.json"
)

EXPECTED_MANIFEST_HASH = (
    "69d11e1a0dd1107f2e6534632727e2c2291ff83d6b17"
    "cf953ed5056bea9020f9"
)

BUDGETS = (54, 72, 90)
DEPTHS = (0, 1, 2, 4, 8)
METHODS = ("global", "spatial")
NODES = 36

for output in (
    DETAIL_PATH,
    SUMMARY_PATH,
    DIFFERENCE_PATH,
    REPORT_PATH,
):
    if output.exists():
        raise RuntimeError(
            f"Refusing to overwrite: {output}"
        )


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def file_hash(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def cosine_matrix(features):
    x = np.asarray(features, dtype=np.float64)

    norms = np.linalg.norm(
        x, axis=1, keepdims=True
    )

    if np.any(norms <= 1e-12):
        raise RuntimeError("Zero-norm feature.")

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
    edges = candidate_edges(rows, cols, "spatial")

    # Identical ordering and tie-breaking to Phase 3B.21.
    edges.sort(
        key=lambda e: (
            (int(rows[e[0]]) - int(rows[e[1]])) ** 2
            + (int(cols[e[0]]) - int(cols[e[1]])) ** 2,
            e[0], e[1],
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
        edge for edge in candidates
        if edge not in backbone
    ]

    if count > len(pool):
        raise RuntimeError("Insufficient candidate edges.")

    ranked = sorted(
        pool,
        key=lambda e: (
            -float(similarity[e[0], e[1]]),
            e[0], e[1],
        )
    )

    return set(ranked[:count])


def normalized_adjacency(n, edges):
    adjacency = np.zeros(
        (n, n), dtype=np.float64
    )

    for a, b in edges:
        adjacency[a, b] = 1.0
        adjacency[b, a] = 1.0

    adjacency += np.eye(n)

    degree = adjacency.sum(axis=1)

    inv_sqrt = 1.0 / np.sqrt(degree)

    return (
        inv_sqrt[:, None]
        * adjacency
        * inv_sqrt[None, :]
    )


# ============================================================
# REPRESENTATION METRICS
# ============================================================

def mean_node_l2(x, y):
    return float(
        np.linalg.norm(x - y, axis=1).mean()
    )


def mean_node_cosine(x, y):
    dot = np.sum(x * y, axis=1)
    norms = (
        np.linalg.norm(x, axis=1)
        * np.linalg.norm(y, axis=1)
    )

    cosine = dot / np.maximum(norms, 1e-12)

    return float(np.clip(cosine, -1, 1).mean())


def mean_node_variance(x):
    """
    Mean squared deviation from the node-wise
    feature centroid.

    Lower values indicate loss of node diversity,
    but are not sufficient alone to establish
    harmful oversmoothing.
    """
    centered = x - x.mean(axis=0, keepdims=True)

    return float(
        np.mean(np.sum(centered ** 2, axis=1))
    )


def mean_offdiagonal_cosine(x):
    sim = cosine_matrix(x)
    n = len(x)

    return float(
        (sim.sum() - np.trace(sim)) / (n * (n - 1))
    )


def propagate_all(S, x, depths):
    states = {0: x.copy()}
    current = x.copy()

    for depth in range(1, max(depths) + 1):
        current = S @ current

        if depth in depths:
            states[depth] = current.copy()

    return states


# ============================================================
# STEP 1: VALIDATION
# ============================================================

print("=" * 70, flush=True)
print("PHASE 3B.23 - GRAPH PROPAGATION SENSITIVITY", flush=True)
print("=" * 70, flush=True)

print("\n[1] Validating frozen research inputs...", flush=True)

if not MANIFEST.is_file() or not FEATURES.is_file():
    raise RuntimeError("Missing required input.")

manifest_hash = file_hash(MANIFEST)

if manifest_hash != EXPECTED_MANIFEST_HASH:
    raise RuntimeError("Frozen manifest hash mismatch.")

manifest = pd.read_csv(
    MANIFEST,
    dtype={"tile_id": str, "region_id": str},
).sort_values(
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
    raise RuntimeError("Feature shape mismatch.")

if not np.isfinite(at2).all() or not np.isfinite(gt450).all():
    raise RuntimeError("Non-finite feature values.")

for actual, expected in (
    (tile_ids, manifest["tile_id"].to_numpy(dtype=str)),
    (region_ids, manifest["region_id"].to_numpy(dtype=str)),
    (rows, manifest["x"].to_numpy(dtype=int)),
    (cols, manifest["y"].to_numpy(dtype=int)),
):
    if not np.array_equal(actual, expected):
        raise RuntimeError(
            "Feature/manifest alignment mismatch."
        )

regions = sorted(set(region_ids))

if len(regions) != 10:
    raise RuntimeError("Expected ten regions.")

print("Frozen manifest: VERIFIED", flush=True)
print("Feature archive: VERIFIED", flush=True)
print("Regions:", len(regions), flush=True)


# ============================================================
# STEP 2: CROSS-SCANNER PROPAGATION
# ============================================================

print("\n[2] Running representation experiments...", flush=True)

records = []

for region_id in regions:
    idx = np.where(region_ids == region_id)[0]

    if len(idx) != NODES:
        raise RuntimeError("Unexpected region size.")

    r = rows[idx]
    c = cols[idx]

    expected_grid = {
        (x, y)
        for x in range(int(r.min()), int(r.min()) + 6)
        for y in range(int(c.min()), int(c.min()) + 6)
    }

    if set(zip(r.tolist(), c.tolist())) != expected_grid:
        raise RuntimeError(f"Invalid grid: {region_id}")

    # Raw unnormalized CLS features, as in Phase 1.
    x_a = at2[idx].astype(np.float64)
    x_b = gt450[idx].astype(np.float64)

    sim_a = cosine_matrix(x_a)
    sim_b = cosine_matrix(x_b)

    backbone = spatial_mst(r, c)

    for method in METHODS:
        candidates = candidate_edges(r, c, method)

        for budget in BUDGETS:
            count = budget - len(backbone)

            residual_a = select_residual(
                sim_a, candidates, backbone, count
            )

            residual_b = select_residual(
                sim_b, candidates, backbone, count
            )

            edges_a = backbone | residual_a
            edges_b = backbone | residual_b

            if (
                len(edges_a) != budget
                or len(edges_b) != budget
            ):
                raise RuntimeError("Invalid edge budget.")

            S_a = normalized_adjacency(NODES, edges_a)
            S_b = normalized_adjacency(NODES, edges_b)

            # Four crossed conditions:
            # first letter = feature source
            # second letter = graph source
            states = {
                "AA": propagate_all(S_a, x_a, DEPTHS),
                "AB": propagate_all(S_b, x_a, DEPTHS),
                "BA": propagate_all(S_a, x_b, DEPTHS),
                "BB": propagate_all(S_b, x_b, DEPTHS),
            }

            for depth in DEPTHS:
                aa = states["AA"][depth]
                ab = states["AB"][depth]
                ba = states["BA"][depth]
                bb = states["BB"][depth]

                # Natural scanner-to-scanner effect.
                natural_l2 = mean_node_l2(aa, bb)
                natural_cos = mean_node_cosine(aa, bb)

                # Same features, different graphs.
                topology_a_l2 = mean_node_l2(aa, ab)
                topology_b_l2 = mean_node_l2(ba, bb)

                topology_a_cos = mean_node_cosine(aa, ab)
                topology_b_cos = mean_node_cosine(ba, bb)

                # Different features, same graph.
                feature_a_l2 = mean_node_l2(aa, ba)
                feature_b_l2 = mean_node_l2(ab, bb)

                feature_a_cos = mean_node_cosine(aa, ba)
                feature_b_cos = mean_node_cosine(ab, bb)

                variance_a = mean_node_variance(aa)
                variance_b = mean_node_variance(bb)

                cosine_a = mean_offdiagonal_cosine(aa)
                cosine_b = mean_offdiagonal_cosine(bb)

                records.append({
                    "region_id": region_id,
                    "graph_type": method,
                    "edge_budget": budget,
                    "depth": depth,
                    "natural_l2": natural_l2,
                    "natural_cosine": natural_cos,
                    "topology_only_l2": (
                        topology_a_l2 + topology_b_l2
                    ) / 2,
                    "topology_only_cosine": (
                        topology_a_cos + topology_b_cos
                    ) / 2,
                    "feature_only_l2": (
                        feature_a_l2 + feature_b_l2
                    ) / 2,
                    "feature_only_cosine": (
                        feature_a_cos + feature_b_cos
                    ) / 2,
                    "at2_node_variance": variance_a,
                    "gt450_node_variance": variance_b,
                    "at2_offdiag_cosine": cosine_a,
                    "gt450_offdiag_cosine": cosine_b,
                })

    print(
        f"  {region_id}: "
        "all crossed conditions completed",
        flush=True,
    )


# ============================================================
# STEP 3: SUMMARIZE
# ============================================================

print("\n[3] Summarizing propagation outcomes...", flush=True)

details = pd.DataFrame(records)

expected_records = (
    len(regions) * len(METHODS)
    * len(BUDGETS) * len(DEPTHS)
)

if len(details) != expected_records:
    raise RuntimeError(
        f"Expected {expected_records} rows, "
        f"found {len(details)}."
    )

summary = (
    details.groupby(
        ["graph_type", "edge_budget", "depth"],
        as_index=False,
    )
    .agg(
        regions=("region_id", "nunique"),
        mean_natural_l2=("natural_l2", "mean"),
        mean_natural_cosine=("natural_cosine", "mean"),
        mean_topology_l2=("topology_only_l2", "mean"),
        std_topology_l2=("topology_only_l2", "std"),
        mean_topology_cosine=(
            "topology_only_cosine", "mean"
        ),
        mean_feature_l2=("feature_only_l2", "mean"),
        mean_feature_cosine=(
            "feature_only_cosine", "mean"
        ),
        mean_at2_variance=(
            "at2_node_variance", "mean"
        ),
        mean_gt450_variance=(
            "gt450_node_variance", "mean"
        ),
        mean_at2_offdiag_cosine=(
            "at2_offdiag_cosine", "mean"
        ),
        mean_gt450_offdiag_cosine=(
            "gt450_offdiag_cosine", "mean"
        ),
    )
)

difference_rows = []

for (region_id, budget, depth), group in details.groupby(
    ["region_id", "edge_budget", "depth"]
):
    if set(group["graph_type"]) != set(METHODS):
        raise RuntimeError(
            "Missing matched graph method."
        )

    g = group.set_index("graph_type")

    difference_rows.append({
        "region_id": region_id,
        "edge_budget": budget,
        "depth": depth,
        "global_topology_l2": float(
            g.loc["global", "topology_only_l2"]
        ),
        "spatial_topology_l2": float(
            g.loc["spatial", "topology_only_l2"]
        ),
        "global_minus_spatial_topology_l2": float(
            g.loc["global", "topology_only_l2"]
            - g.loc["spatial", "topology_only_l2"]
        ),
        "global_minus_spatial_natural_l2": float(
            g.loc["global", "natural_l2"]
            - g.loc["spatial", "natural_l2"]
        ),
    })

differences = pd.DataFrame(difference_rows)

print("\nDepth 2 topology-only L2:")
print(
    summary[summary["depth"] == 2][
        [
            "graph_type",
            "edge_budget",
            "mean_topology_l2",
            "std_topology_l2",
            "mean_natural_l2",
            "mean_natural_cosine",
        ]
    ].to_string(index=False)
)

print("\nGlobal minus spatial topology L2 at depth 2:")
depth2 = differences[differences["depth"] == 2]

print(
    depth2.groupby("edge_budget")[
        "global_minus_spatial_topology_l2"
    ].agg(["mean", "min", "max"]).to_string()
)

print("\nDepth progression at budget 72:")
print(
    summary[summary["edge_budget"] == 72][
        [
            "graph_type", "depth",
            "mean_topology_l2",
            "mean_natural_l2",
            "mean_at2_variance",
            "mean_gt450_variance",
        ]
    ].to_string(index=False)
)


# ============================================================
# STEP 4: SAVE
# ============================================================

print("\n[4] Saving results...", flush=True)

details.to_csv(DETAIL_PATH, index=False)
summary.to_csv(SUMMARY_PATH, index=False)
differences.to_csv(DIFFERENCE_PATH, index=False)

report = {
    "phase": "3B",
    "step": "3B.23",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "manifest_sha256": manifest_hash,
    "feature_archive_sha256": file_hash(FEATURES),
    "regions": len(regions),
    "nodes_per_region": NODES,
    "edge_budgets": list(BUDGETS),
    "depths": list(DEPTHS),
    "graph_types": list(METHODS),
    "detail_records": len(details),
    "paired_difference_records": len(differences),
    "propagation_operator": (
        "D^-1/2 (A+I) D^-1/2"
    ),
    "initial_features": (
        "raw DINOv2-Small CLS embeddings"
    ),
    "topology_effect": (
        "Average of AT2 and GT450 feature matrices "
        "propagated through alternate scanner graphs"
    ),
    "feature_effect": (
        "Average difference between AT2 and GT450 "
        "features propagated through a fixed graph"
    ),
    "trained_gnn_used": False,
    "weights_learned": False,
    "images_modified": False,
    "features_modified": False,
    "frozen_manifest_modified": False,
    "manual_registration_review_completed": False,
    "biological_independence_verified": False,
    "limitations": [
        "Deterministic propagation is not trained GNN evaluation",
        "No nonlinearities, trainable weights or classification loss",
        "Graph and feature effects are not additive",
        "Smaller propagation differences do not imply better accuracy",
        "Node variance is a descriptive smoothing diagnostic",
        "All ten regions originate from one acquisition pair",
        "Spatial and global candidate pools remain different",
    ],
}

REPORT_PATH.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3B.23 - PROPAGATION ANALYSIS COMPLETED")
print("=" * 70)

print("Regions:", len(regions))
print("Graph configurations:", len(regions) * 2 * 3)
print("Propagation depths:", list(DEPTHS))
print("Detailed experiment records:", len(details))
print("Paired comparison records:", len(differences))

print("\nSaved:")
print(DETAIL_PATH)
print(SUMMARY_PATH)
print(DIFFERENCE_PATH)
print(REPORT_PATH)

print("\nTrained GNN: NO")
print("Frozen manifest: UNCHANGED")
print("Original features: UNCHANGED")
print("Biological independence: UNVERIFIED")
print("\nSTEP 3B.23 COMPLETED")
