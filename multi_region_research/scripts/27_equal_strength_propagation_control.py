
"""
PHASE 3B.27
Equal-Strength Cross-Scanner Propagation Control

Purpose:
Compare global and spatial graphs using identical
propagation strength (alpha), depth, nodes and edge budget.

Reconstructs the exact Phase 3B.21 graphs and validates
the alpha=1 results against Phase 3B.23.

Experiments:
    Regions: 10
    Methods: global, spatial
    Budgets: 54, 72, 90
    Alpha: 0.25, 0.50, 0.75, 1.00
    Depth: 2

Outputs:
    240 graph-condition rows
    120 paired method comparisons
    12 aggregate summary rows

All original research inputs remain unchanged.
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

MANIFEST = BASE / (
    "outputs/validated/"
    "phase3b_frozen_tile_manifest.csv"
)

FEATURES = BASE / (
    "outputs/features/"
    "plism_phase3b_dinov2_features.npz"
)

CONNECTED = BASE / (
    "outputs/tables/"
    "plism_phase3b_connected_graph_per_region.csv"
)

PROPAGATION = BASE / (
    "outputs/tables/"
    "plism_phase3b_propagation_per_region.csv"
)

TABLES = BASE / "outputs/tables"
REPORTS = BASE / "outputs/reports"

DETAIL = TABLES / (
    "plism_phase3b_equal_strength_details.csv"
)

PAIRED = TABLES / (
    "plism_phase3b_equal_strength_paired.csv"
)

SUMMARY = TABLES / (
    "plism_phase3b_equal_strength_summary.csv"
)

REPORT = REPORTS / (
    "27_equal_strength_propagation_control.json"
)

EXPECTED_HASH = (
    "69d11e1a0dd1107f2e6534632727e2c2291ff83d6b17"
    "cf953ed5056bea9020f9"
)

BUDGETS = (54, 72, 90)
ALPHAS = (0.25, 0.50, 0.75, 1.00)
METHODS = ("global", "spatial")
DEPTH = 2
NODES = 36


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    edges = candidate_edges(rows, cols, "spatial")

    # Identical sorting to Phase 3B.21.
    edges.sort(
        key=lambda e: (
            (int(rows[e[0]]) - int(rows[e[1]])) ** 2
            + (int(cols[e[0]]) - int(cols[e[1]])) ** 2,
            e[0],
            e[1],
        )
    )

    n = len(rows)
    parent = list(range(n))
    sizes = [1] * n

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

    for edge in edges:
        if union(*edge):
            tree.add(edge)

        if len(tree) == n - 1:
            break

    if len(tree) != n - 1:
        raise RuntimeError("MST construction failed.")

    return tree


def cosine_matrix(features):
    x = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)

    if np.any(norms <= 1e-12):
        raise RuntimeError("Zero-norm feature.")

    z = x / norms
    return z @ z.T


def construct_graph(features, rows, cols,
                    method, budget, backbone):

    similarity = cosine_matrix(features)

    pool = [
        edge
        for edge in candidate_edges(rows, cols, method)
        if edge not in backbone
    ]

    pool.sort(
        key=lambda edge: (
            -float(similarity[edge[0], edge[1]]),
            edge[0],
            edge[1],
        )
    )

    count = budget - len(backbone)

    if count > len(pool):
        raise RuntimeError("Insufficient candidate edges.")

    residual = set(pool[:count])
    graph = backbone | residual

    if len(graph) != budget:
        raise RuntimeError("Graph edge count mismatch.")

    return graph, residual


def normalized_adjacency(edges, n):
    A = np.eye(n, dtype=np.float64)

    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0

    degrees = A.sum(axis=1)
    inv = degrees ** -0.5

    return inv[:, None] * A * inv[None, :]


def propagate(S, x, alpha):
    operator = (
        (1.0 - alpha) * np.eye(len(x))
        + alpha * S
    )

    state = x.copy()

    for _ in range(DEPTH):
        state = operator @ state

    return state


def mean_node_l2(x, y):
    return float(
        np.linalg.norm(x - y, axis=1).mean()
    )


def node_variance(x):
    centered = x - x.mean(axis=0, keepdims=True)

    return float(
        np.mean(np.sum(centered ** 2, axis=1))
    )


def evaluate(Sa, Sb, xa, xb, alpha):
    aa = propagate(Sa, xa, alpha)
    ab = propagate(Sb, xa, alpha)
    ba = propagate(Sa, xb, alpha)
    bb = propagate(Sb, xb, alpha)

    topology_l2 = (
        mean_node_l2(aa, ab)
        + mean_node_l2(ba, bb)
    ) / 2.0

    natural_l2 = mean_node_l2(aa, bb)

    retention_a = node_variance(aa) / node_variance(xa)
    retention_b = node_variance(bb) / node_variance(xb)

    return {
        "topology_l2": topology_l2,
        "natural_l2": natural_l2,
        "variance_retention": (
            retention_a + retention_b
        ) / 2.0,
    }


# ============================================================
# STEP 1: VERIFY INPUTS
# ============================================================

print("=" * 70, flush=True)
print(
    "PHASE 3B.27 - EQUAL-STRENGTH PROPAGATION CONTROL",
    flush=True,
)
print("=" * 70, flush=True)

print("\n[1] Verifying frozen inputs...", flush=True)

for path in (
    MANIFEST, FEATURES, CONNECTED, PROPAGATION
):
    if not path.is_file():
        raise RuntimeError(f"Missing input: {path}")

for path in (DETAIL, PAIRED, SUMMARY, REPORT):
    if path.exists():
        raise RuntimeError(
            f"Refusing to overwrite result: {path}"
        )

if sha256_file(MANIFEST) != EXPECTED_HASH:
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

if not (np.isfinite(at2).all()
        and np.isfinite(gt450).all()):
    raise RuntimeError("Non-finite features.")

for actual, expected in (
    (tile_ids, manifest["tile_id"].to_numpy(dtype=str)),
    (region_ids, manifest["region_id"].to_numpy(dtype=str)),
    (rows, manifest["x"].to_numpy(dtype=int)),
    (cols, manifest["y"].to_numpy(dtype=int)),
):
    if not np.array_equal(actual, expected):
        raise RuntimeError("Feature ordering mismatch.")

connected = pd.read_csv(CONNECTED)
previous = pd.read_csv(PROPAGATION)
previous = previous[previous["depth"] == DEPTH].copy()

keys = ["region_id", "graph_type", "edge_budget"]

if connected.duplicated(keys).any():
    raise RuntimeError("Duplicate connected graph records.")

if previous.duplicated(keys).any():
    raise RuntimeError("Duplicate propagation records.")

if len(connected) != 60 or len(previous) != 60:
    raise RuntimeError("Incomplete prior experiments.")

connected = connected.set_index(keys)
previous = previous.set_index(keys)

regions = sorted(set(region_ids))

if len(regions) != 10:
    raise RuntimeError("Expected ten regions.")

print("Frozen manifest: VERIFIED", flush=True)
print("Feature alignment: VERIFIED", flush=True)
print("Prior graph results: AVAILABLE", flush=True)
print("Regions:", len(regions), flush=True)


# ============================================================
# STEP 2: MATCHED ALPHA EXPERIMENT
# ============================================================

print("\n[2] Running equal-alpha comparisons...", flush=True)

records = []

for region_id in regions:
    indices = np.where(region_ids == region_id)[0]

    if len(indices) != NODES:
        raise RuntimeError("Unexpected region size.")

    r = rows[indices]
    c = cols[indices]

    expected_grid = {
        (x, y)
        for x in range(int(r.min()), int(r.min()) + 6)
        for y in range(int(c.min()), int(c.min()) + 6)
    }

    if set(zip(r.tolist(), c.tolist())) != expected_grid:
        raise RuntimeError(
            f"Invalid grid: {region_id}"
        )

    xa = at2[indices].astype(np.float64)
    xb = gt450[indices].astype(np.float64)

    backbone = spatial_mst(r, c)

    for budget in BUDGETS:
        for method in METHODS:

            edges_a, residual_a = construct_graph(
                xa, r, c, method, budget, backbone
            )
            edges_b, residual_b = construct_graph(
                xb, r, c, method, budget, backbone
            )

            key = (region_id, method, budget)

            observed_shared = len(
                residual_a & residual_b
            )

            expected_shared = int(
                connected.loc[key, "shared_residual_edges"]
            )

            if observed_shared != expected_shared:
                raise RuntimeError(
                    f"Graph mismatch: {key}"
                )

            Sa = normalized_adjacency(edges_a, NODES)
            Sb = normalized_adjacency(edges_b, NODES)

            for alpha in ALPHAS:
                metrics = evaluate(Sa, Sb, xa, xb, alpha)

                if alpha == 1.0:
                    old = previous.loc[key]

                    for measured, old_field in (
                        ("topology_l2", "topology_only_l2"),
                        ("natural_l2", "natural_l2"),
                    ):
                        if not np.isclose(
                            metrics[measured],
                            float(old[old_field]),
                            atol=1e-5,
                            rtol=1e-7,
                        ):
                            raise RuntimeError(
                                f"Propagation mismatch: {key}"
                            )

                records.append({
                    "region_id": region_id,
                    "graph_type": method,
                    "edge_budget": budget,
                    "alpha": float(alpha),
                    "depth": DEPTH,
                    **metrics,
                })

    print(
        f"  {region_id}: completed",
        flush=True,
    )


# ============================================================
# STEP 3: PAIRED RESULTS
# ============================================================

print("\n[3] Comparing methods at equal alpha...", flush=True)

details = pd.DataFrame(records)

if len(details) != 240:
    raise RuntimeError("Expected 240 detailed records.")

paired_records = []

for (region_id, budget, alpha), group in details.groupby(
    ["region_id", "edge_budget", "alpha"]
):
    if set(group["graph_type"]) != set(METHODS):
        raise RuntimeError("Missing graph method.")

    g = group.set_index("graph_type")
    glob = g.loc["global"]
    spat = g.loc["spatial"]

    paired_records.append({
        "region_id": region_id,
        "edge_budget": budget,
        "alpha": alpha,
        "global_topology_l2": glob.topology_l2,
        "spatial_topology_l2": spat.topology_l2,
        "global_minus_spatial_topology_l2":
            glob.topology_l2 - spat.topology_l2,
        "global_retention": glob.variance_retention,
        "spatial_retention": spat.variance_retention,
        "spatial_minus_global_retention":
            spat.variance_retention
            - glob.variance_retention,
        "global_natural_l2": glob.natural_l2,
        "spatial_natural_l2": spat.natural_l2,
    })

paired = pd.DataFrame(paired_records)

if len(paired) != 120:
    raise RuntimeError("Expected 120 paired records.")

summary = (
    paired.groupby(
        ["edge_budget", "alpha"],
        as_index=False,
    )
    .agg(
        regions=("region_id", "nunique"),
        mean_global_topology_l2=(
            "global_topology_l2", "mean"
        ),
        mean_spatial_topology_l2=(
            "spatial_topology_l2", "mean"
        ),
        mean_topology_advantage=(
            "global_minus_spatial_topology_l2", "mean"
        ),
        min_topology_advantage=(
            "global_minus_spatial_topology_l2", "min"
        ),
        positive_regions=(
            "global_minus_spatial_topology_l2",
            lambda x: int((x > 0).sum())
        ),
        mean_global_retention=(
            "global_retention", "mean"
        ),
        mean_spatial_retention=(
            "spatial_retention", "mean"
        ),
        mean_retention_difference=(
            "spatial_minus_global_retention", "mean"
        ),
        mean_global_natural_l2=(
            "global_natural_l2", "mean"
        ),
        mean_spatial_natural_l2=(
            "spatial_natural_l2", "mean"
        ),
    )
)

if len(summary) != 12:
    raise RuntimeError("Expected 12 summary rows.")

print("\nEqual-alpha comparison summary:")
print(summary.to_string(index=False))

print("\nAlpha = 0.50:")
print(
    summary[summary["alpha"] == 0.50][
        [
            "edge_budget",
            "mean_global_topology_l2",
            "mean_spatial_topology_l2",
            "mean_topology_advantage",
            "positive_regions",
            "mean_global_retention",
            "mean_spatial_retention",
        ]
    ].to_string(index=False)
)


# ============================================================
# STEP 4: SAVE
# ============================================================

print("\n[4] Saving outputs...", flush=True)

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

details.to_csv(DETAIL, index=False)
paired.to_csv(PAIRED, index=False)
summary.to_csv(SUMMARY, index=False)

report = {
    "phase": "3B",
    "step": "3B.27",
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "manifest_sha256": sha256_file(MANIFEST),
    "feature_sha256": sha256_file(FEATURES),
    "step21_sha256": sha256_file(CONNECTED),
    "step23_sha256": sha256_file(PROPAGATION),
    "regions": 10,
    "depth": DEPTH,
    "edge_budgets": list(BUDGETS),
    "alphas": list(ALPHAS),
    "detail_rows": len(details),
    "paired_rows": len(paired),
    "summary_rows": len(summary),
    "operator": "(1-alpha)*I + alpha*S",
    "graph_reconstruction_verified": True,
    "alpha_one_reproduction_verified": True,
    "same_alpha_across_methods": True,
    "frozen_manifest_modified": False,
    "trained_gnn_used": False,
    "biological_independence_verified": False,
    "limitations": [
        "Equal alpha does not imply equal smoothing",
        "Raw topology L2 depends on propagation strength",
        "Candidate edge pools differ between graph methods",
        "Regions belong to one acquisition pair",
        "No independent specimen-level evaluation",
        "No trained GNN or downstream task",
    ],
}

REPORT.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70, flush=True)
print(
    "PHASE 3B.27 - EQUAL-STRENGTH CONTROL COMPLETED",
    flush=True,
)
print("=" * 70, flush=True)

print("Regions:", len(regions))
print("Detailed records:", len(details))
print("Paired comparisons:", len(paired))
print("Summary configurations:", len(summary))

print("\nSaved:")
print(DETAIL)
print(PAIRED)
print(SUMMARY)
print(REPORT)

print("\nFrozen manifest: UNCHANGED")
print("Original features: UNCHANGED")
print("Trained GNN: NOT USED")
print("Biological independence: UNVERIFIED")
print("\nSTEP 3B.27 COMPLETED")
