
"""
PHASE 3B.26
SMOOTHING-MATCHED REPRESENTATION STABILITY

Research objective
------------------
Determine whether spatially constrained graphs retain their
lower cross-scanner topology sensitivity when spatial and
global propagation are matched on node-variance retention.

Inputs
------
1. Frozen Phase 3B manifest
2. Phase 3B.19 DINOv2 feature archive
3. Phase 3B.21 connected graph results
4. Phase 3B.23 propagation results

Protocol
--------
- Ten frozen regions
- 36 nodes per region
- Shared deterministic 35-edge spatial MST
- Scanner-specific cosine-ranked residual edges
- Total edge budgets: 54, 72, 90
- Two propagation layers

Global graph:
    Original propagation operator, alpha = 1.0

Spatial graph:
    Calibrated propagation operator

    S_alpha = (1 - alpha) * I + alpha * S

    S = D^(-1/2) (A + I) D^(-1/2)

    H_2 = S_alpha @ S_alpha @ X

One spatial alpha is selected per region and edge
budget, using average AT2 and GT450 variance retention.

Outputs
-------
- Per-region smoothing-matched comparison CSV
- Aggregate smoothing-matched summary CSV
- JSON reproducibility report

Safety
------
- Does not overwrite previous outputs
- Does not modify original images
- Does not modify saved features
- Does not modify frozen sampling manifest

Important
---------
This is an exploratory oracle-calibrated control.
No trained GNN or biological independence is assumed.
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
    BASE
    / "outputs/validated"
    / "phase3b_frozen_tile_manifest.csv"
)

FEATURES = (
    BASE
    / "outputs/features"
    / "plism_phase3b_dinov2_features.npz"
)

PREVIOUS_PROPAGATION = (
    BASE
    / "outputs/tables"
    / "plism_phase3b_propagation_per_region.csv"
)

PREVIOUS_CONNECTED = (
    BASE
    / "outputs/tables"
    / "plism_phase3b_connected_graph_per_region.csv"
)

TABLE_DIR = BASE / "outputs/tables"
REPORT_DIR = BASE / "outputs/reports"

DETAIL_PATH = (
    TABLE_DIR
    / "plism_phase3b_smoothing_matched_per_region.csv"
)

SUMMARY_PATH = (
    TABLE_DIR
    / "plism_phase3b_smoothing_matched_summary.csv"
)

REPORT_PATH = (
    REPORT_DIR
    / "26_smoothing_matched_stability.json"
)

EXPECTED_MANIFEST_HASH = (
    "69d11e1a0dd1107f2e6534632727e2c2291ff83d6b17"
    "cf953ed5056bea9020f9"
)

EDGE_BUDGETS = (54, 72, 90)

METHODS = ("global", "spatial")

NODES_PER_REGION = 36

BACKBONE_EDGES = 35

PROPAGATION_DEPTH = 2

ALPHA_GRID = np.linspace(
    0.0, 1.0, 1001,
    dtype=np.float64,
)

RETENTION_WARNING_TOLERANCE = 0.005

REPRODUCTION_ATOL = 1e-5


# ============================================================
# GENERAL HELPERS
# ============================================================

def sha256_file(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def check_file_exists(path):
    if not path.is_file():
        raise RuntimeError(
            f"Required input file missing:\n{path}"
        )


def refuse_overwrite(paths):
    for path in paths:
        if path.exists():
            raise RuntimeError(
                "Refusing to overwrite previous result:\n"
                f"{path}"
            )


def assert_close(actual, expected, label):
    if not np.isclose(
        float(actual),
        float(expected),
        atol=REPRODUCTION_ATOL,
        rtol=1e-7,
    ):
        raise RuntimeError(
            f"Reproduction mismatch: {label}\n"
            f"Actual: {actual}\n"
            f"Expected: {expected}"
        )


# ============================================================
# GRAPH CONSTRUCTION
# Matches Phase 3B.21 and Phase 3B.23
# ============================================================

def cosine_matrix(features):

    x = np.asarray(
        features,
        dtype=np.float64,
    )

    norms = np.linalg.norm(
        x,
        axis=1,
        keepdims=True,
    )

    if np.any(norms <= 1e-12):
        raise RuntimeError(
            "Zero-norm feature vector detected."
        )

    normalized = x / norms

    return normalized @ normalized.T


def candidate_edges(rows, cols, method):

    edges = []

    n = len(rows)

    for a, b in combinations(range(n), 2):

        dx = abs(
            int(rows[a]) - int(rows[b])
        )

        dy = abs(
            int(cols[a]) - int(cols[b])
        )

        if method == "spatial":

            if max(dx, dy) != 1:
                continue

        edges.append((a, b))

    return edges


def spatial_mst(rows, cols):
    """
    Deterministic spatial MST.

    Same Kruskal ordering as Phase 3B.21:
    squared grid distance, then node indices.
    """

    edges = candidate_edges(
        rows, cols, "spatial"
    )

    edges.sort(
        key=lambda edge: (
            (
                int(rows[edge[0]])
                - int(rows[edge[1]])
            ) ** 2
            +
            (
                int(cols[edge[0]])
                - int(cols[edge[1]])
            ) ** 2,
            edge[0],
            edge[1],
        )
    )

    n = len(rows)

    parent = list(range(n))
    sizes = [1] * n

    def find(node):

        while parent[node] != node:

            parent[node] = parent[parent[node]]

            node = parent[node]

        return node

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
        raise RuntimeError(
            "Spatial MST construction failed."
        )

    return tree


def build_graph(
    features,
    rows,
    cols,
    method,
    edge_budget,
    backbone,
):
    """
    Scanner-specific residual edge selection.

    Features are L2-normalized for cosine ranking.
    Saved raw feature vectors are not modified.
    """

    similarity = cosine_matrix(features)

    candidates = candidate_edges(
        rows, cols, method
    )

    residual_pool = [
        edge
        for edge in candidates
        if edge not in backbone
    ]

    residual_count = (
        edge_budget - len(backbone)
    )

    if residual_count > len(residual_pool):
        raise RuntimeError(
            "Insufficient residual candidates."
        )

    ranked = sorted(
        residual_pool,
        key=lambda edge: (
            -float(
                similarity[
                    edge[0],
                    edge[1],
                ]
            ),
            edge[0],
            edge[1],
        ),
    )

    residual = set(
        ranked[:residual_count]
    )

    full_graph = backbone | residual

    if len(full_graph) != edge_budget:
        raise RuntimeError(
            "Graph edge budget mismatch."
        )

    return full_graph, residual


def normalized_adjacency(edges, n):
    """
    Symmetric normalized adjacency with self-loops.
    """

    adjacency = np.eye(
        n,
        dtype=np.float64,
    )

    for a, b in edges:

        adjacency[a, b] = 1.0
        adjacency[b, a] = 1.0

    degrees = adjacency.sum(axis=1)

    if np.any(degrees <= 0):
        raise RuntimeError(
            "Invalid graph degree."
        )

    inv_sqrt = (
        1.0 / np.sqrt(degrees)
    )

    return (
        inv_sqrt[:, None]
        * adjacency
        * inv_sqrt[None, :]
    )


# ============================================================
# REPRESENTATION METRICS
# ============================================================

def propagate(operator_matrix, features, alpha):

    n = len(features)

    mixed_operator = (
        (1.0 - alpha) * np.eye(n)
        +
        alpha * operator_matrix
    )

    result = features.copy()

    for _ in range(PROPAGATION_DEPTH):
        result = mixed_operator @ result

    return result


def node_variance(features):
    """
    Mean squared feature-vector distance
    from the regional centroid.
    """

    centered = (
        features
        -
        features.mean(
            axis=0,
            keepdims=True,
        )
    )

    return float(
        np.mean(
            np.sum(
                centered ** 2,
                axis=1,
            )
        )
    )


def mean_node_l2(x, y):

    return float(
        np.linalg.norm(
            x - y,
            axis=1,
        ).mean()
    )


def mean_node_cosine(x, y):

    numerator = np.sum(
        x * y,
        axis=1,
    )

    denominator = (
        np.linalg.norm(x, axis=1)
        *
        np.linalg.norm(y, axis=1)
    )

    cosine = numerator / np.maximum(
        denominator,
        1e-12,
    )

    return float(
        np.clip(
            cosine,
            -1.0,
            1.0,
        ).mean()
    )


def evaluate_condition(
    operator_at2,
    operator_gt450,
    features_at2,
    features_gt450,
    alpha,
):
    """
    Four crossed propagation conditions:

    AA: AT2 features on AT2 graph
    AB: AT2 features on GT450 graph
    BA: GT450 features on AT2 graph
    BB: GT450 features on GT450 graph
    """

    aa = propagate(
        operator_at2,
        features_at2,
        alpha,
    )

    ab = propagate(
        operator_gt450,
        features_at2,
        alpha,
    )

    ba = propagate(
        operator_at2,
        features_gt450,
        alpha,
    )

    bb = propagate(
        operator_gt450,
        features_gt450,
        alpha,
    )

    topology_l2 = (
        mean_node_l2(aa, ab)
        +
        mean_node_l2(ba, bb)
    ) / 2.0

    topology_cosine = (
        mean_node_cosine(aa, ab)
        +
        mean_node_cosine(ba, bb)
    ) / 2.0

    natural_l2 = mean_node_l2(
        aa, bb
    )

    natural_cosine = mean_node_cosine(
        aa, bb
    )

    initial_at2_variance = node_variance(
        features_at2
    )

    initial_gt450_variance = node_variance(
        features_gt450
    )

    if (
        initial_at2_variance <= 1e-12
        or initial_gt450_variance <= 1e-12
    ):
        raise RuntimeError(
            "Invalid initial node variance."
        )

    retention_at2 = (
        node_variance(aa)
        / initial_at2_variance
    )

    retention_gt450 = (
        node_variance(bb)
        / initial_gt450_variance
    )

    mean_retention = (
        retention_at2
        +
        retention_gt450
    ) / 2.0

    return {
        "topology_l2": topology_l2,
        "topology_cosine": topology_cosine,
        "natural_l2": natural_l2,
        "natural_cosine": natural_cosine,
        "variance_retention": mean_retention,
        "at2_variance_retention": retention_at2,
        "gt450_variance_retention": retention_gt450,
    }


# ============================================================
# EFFICIENT ALPHA CALIBRATION
# ============================================================

def retention_curve(
    operator_matrix,
    features,
    alphas,
):
    """
    Compute variance retention for every alpha
    without running full feature propagation
    separately for each candidate.

    At depth 2:

    ((I + alpha B)^2) X
      = X + 2 alpha B X + alpha^2 B^2 X

    where B = S - I.

    The squared norm of the centered result is
    a fourth-degree polynomial in alpha.
    """

    n = len(features)

    identity = np.eye(
        n,
        dtype=np.float64,
    )

    B = operator_matrix - identity

    term_0 = features

    term_1 = 2.0 * (B @ features)

    term_2 = B @ (B @ features)

    terms = []

    for term in (
        term_0,
        term_1,
        term_2,
    ):

        centered = (
            term
            -
            term.mean(
                axis=0,
                keepdims=True,
            )
        )

        terms.append(centered)

    initial_variance = node_variance(
        features
    )

    if initial_variance <= 1e-12:
        raise RuntimeError(
            "Invalid calibration variance."
        )

    coefficients = np.zeros(
        5,
        dtype=np.float64,
    )

    for i in range(3):
        for j in range(3):

            coefficients[i + j] += (
                np.sum(
                    terms[i] * terms[j]
                )
                / n
            )

    curve = np.polynomial.polynomial.polyval(
        alphas,
        coefficients,
    )

    curve = curve / initial_variance

    return curve


def find_matching_alpha(
    operator_at2,
    operator_gt450,
    features_at2,
    features_gt450,
    target_retention,
):
    """
    Choose a single spatial alpha per region/budget.

    Calibration uses the average natural-condition
    variance retention for both scanners.
    """

    curve_at2 = retention_curve(
        operator_at2,
        features_at2,
        ALPHA_GRID,
    )

    curve_gt450 = retention_curve(
        operator_gt450,
        features_gt450,
        ALPHA_GRID,
    )

    average_curve = (
        curve_at2 + curve_gt450
    ) / 2.0

    if not np.isfinite(
        average_curve
    ).all():
        raise RuntimeError(
            "Non-finite calibration curve."
        )

    errors = np.abs(
        average_curve - target_retention
    )

    best_index = int(
        np.argmin(errors)
    )

    best_alpha = float(
        ALPHA_GRID[best_index]
    )

    return {
        "alpha": best_alpha,
        "predicted_retention": float(
            average_curve[best_index]
        ),
        "predicted_error": float(
            errors[best_index]
        ),
        "target_within_grid_range": bool(
            average_curve.min()
            <= target_retention
            <= average_curve.max()
        ),
        "retention_curve_min": float(
            average_curve.min()
        ),
        "retention_curve_max": float(
            average_curve.max()
        ),
    }


# ============================================================
# STEP 1: VALIDATE INPUT FILES
# ============================================================

print("=" * 70, flush=True)
print(
    "PHASE 3B.26 - SMOOTHING-MATCHED STABILITY",
    flush=True,
)
print("=" * 70, flush=True)

print(
    "\n[1] Validating research inputs...",
    flush=True,
)

for input_path in (
    MANIFEST,
    FEATURES,
    PREVIOUS_PROPAGATION,
    PREVIOUS_CONNECTED,
):
    check_file_exists(input_path)

refuse_overwrite(
    (
        DETAIL_PATH,
        SUMMARY_PATH,
        REPORT_PATH,
    )
)

manifest_hash = sha256_file(
    MANIFEST
)

if manifest_hash != EXPECTED_MANIFEST_HASH:
    raise RuntimeError(
        "Frozen manifest SHA256 mismatch."
    )

manifest = pd.read_csv(
    MANIFEST,
    dtype={
        "tile_id": str,
        "region_id": str,
    },
)

manifest = manifest.sort_values(
    [
        "region_id",
        "level",
        "x",
        "y",
        "tile_id",
    ]
).reset_index(drop=True)

with np.load(
    FEATURES,
    allow_pickle=False,
) as data:

    features_at2 = data["at2"].copy()

    features_gt450 = data["gt450"].copy()

    tile_ids = data["tile_ids"].astype(str)

    region_ids = data["region_ids"].astype(str)

    grid_rows = data["grid_rows"].astype(int)

    grid_cols = data["grid_cols"].astype(int)


# ============================================================
# VERIFY FEATURE ALIGNMENT
# ============================================================

if features_at2.shape != (360, 384):
    raise RuntimeError(
        "Unexpected AT2 feature dimensions."
    )

if features_gt450.shape != (360, 384):
    raise RuntimeError(
        "Unexpected GT450 feature dimensions."
    )

if not (
    np.isfinite(features_at2).all()
    and np.isfinite(features_gt450).all()
):
    raise RuntimeError(
        "Non-finite DINOv2 features detected."
    )

checks = (
    (
        tile_ids,
        manifest["tile_id"].to_numpy(dtype=str),
        "tile IDs",
    ),
    (
        region_ids,
        manifest["region_id"].to_numpy(dtype=str),
        "region IDs",
    ),
    (
        grid_rows,
        manifest["x"].to_numpy(dtype=int),
        "grid rows",
    ),
    (
        grid_cols,
        manifest["y"].to_numpy(dtype=int),
        "grid columns",
    ),
)

for actual, expected, label in checks:

    if not np.array_equal(actual, expected):
        raise RuntimeError(
            f"Feature/manifest mismatch: {label}"
        )


# ============================================================
# LOAD PREVIOUS EXPERIMENTS
# ============================================================

previous = pd.read_csv(
    PREVIOUS_PROPAGATION
)

connected = pd.read_csv(
    PREVIOUS_CONNECTED
)

if previous.duplicated(
    [
        "region_id",
        "graph_type",
        "edge_budget",
        "depth",
    ]
).any():
    raise RuntimeError(
        "Duplicate records in Step 3B.23."
    )

previous = previous[
    previous["depth"] == PROPAGATION_DEPTH
].copy()

if len(previous) != 60:
    raise RuntimeError(
        "Expected 60 depth-2 propagation records."
    )

if connected.duplicated(
    [
        "region_id",
        "graph_type",
        "edge_budget",
    ]
).any():
    raise RuntimeError(
        "Duplicate records in Step 3B.21."
    )

if len(connected) != 60:
    raise RuntimeError(
        "Expected 60 connected graph records."
    )

previous = previous.set_index(
    [
        "region_id",
        "graph_type",
        "edge_budget",
    ]
)

connected = connected.set_index(
    [
        "region_id",
        "graph_type",
        "edge_budget",
    ]
)

regions = sorted(
    set(region_ids)
)

if len(regions) != 10:
    raise RuntimeError(
        "Expected exactly ten regions."
    )

print(
    "Frozen manifest: VERIFIED",
    flush=True,
)

print(
    "Feature alignment: VERIFIED",
    flush=True,
)

print(
    "Step 3B.21 connected graph results: AVAILABLE",
    flush=True,
)

print(
    "Step 3B.23 propagation results: AVAILABLE",
    flush=True,
)

print(
    "Regions:",
    len(regions),
    flush=True,
)


# ============================================================
# STEP 2: RECONSTRUCT GRAPHS
# ============================================================

print(
    "\n[2] Reconstructing graphs and calibrating smoothing...",
    flush=True,
)

records = []

for region_id in regions:

    indices = np.where(
        region_ids == region_id
    )[0]

    if len(indices) != NODES_PER_REGION:
        raise RuntimeError(
            f"Unexpected region size: {region_id}"
        )

    rows = grid_rows[indices]

    cols = grid_cols[indices]

    expected_grid = {
        (x, y)
        for x in range(
            int(rows.min()),
            int(rows.min()) + 6,
        )
        for y in range(
            int(cols.min()),
            int(cols.min()) + 6,
        )
    }

    actual_grid = set(
        zip(
            rows.tolist(),
            cols.tolist(),
        )
    )

    if actual_grid != expected_grid:
        raise RuntimeError(
            f"Invalid 6x6 region geometry: {region_id}"
        )

    x_at2 = features_at2[
        indices
    ].astype(np.float64)

    x_gt450 = features_gt450[
        indices
    ].astype(np.float64)

    backbone = spatial_mst(
        rows,
        cols,
    )

    if len(backbone) != BACKBONE_EDGES:
        raise RuntimeError(
            "Unexpected spanning tree size."
        )

    for budget in EDGE_BUDGETS:

        operators = {}

        for method in METHODS:

            at2_edges, at2_residual = build_graph(
                x_at2,
                rows,
                cols,
                method,
                budget,
                backbone,
            )

            gt450_edges, gt450_residual = build_graph(
                x_gt450,
                rows,
                cols,
                method,
                budget,
                backbone,
            )

            key = (
                region_id,
                method,
                budget,
            )

            if key not in connected.index:
                raise RuntimeError(
                    f"Missing Step 3B.21 record: {key}"
                )

            observed_shared = len(
                at2_residual & gt450_residual
            )

            previous_shared = int(
                connected.loc[
                    key,
                    "shared_residual_edges",
                ]
            )

            if observed_shared != previous_shared:
                raise RuntimeError(
                    "Graph reconstruction mismatch:\n"
                    f"{key}"
                )

            operators[method] = (
                normalized_adjacency(
                    at2_edges,
                    NODES_PER_REGION,
                ),
                normalized_adjacency(
                    gt450_edges,
                    NODES_PER_REGION,
                ),
            )

        global_at2, global_gt450 = (
            operators["global"]
        )

        spatial_at2, spatial_gt450 = (
            operators["spatial"]
        )


        # ====================================================
        # REPRODUCE ORIGINAL DEPTH-2 RESULTS
        # ====================================================

        global_original = evaluate_condition(
            global_at2,
            global_gt450,
            x_at2,
            x_gt450,
            alpha=1.0,
        )

        spatial_original = evaluate_condition(
            spatial_at2,
            spatial_gt450,
            x_at2,
            x_gt450,
            alpha=1.0,
        )

        for method, result in (
            ("global", global_original),
            ("spatial", spatial_original),
        ):

            key = (
                region_id,
                method,
                budget,
            )

            if key not in previous.index:
                raise RuntimeError(
                    f"Missing Step 3B.23 result: {key}"
                )

            previous_row = previous.loc[key]

            assert_close(
                result["topology_l2"],
                previous_row["topology_only_l2"],
                f"{key} topology L2",
            )

            assert_close(
                result["natural_l2"],
                previous_row["natural_l2"],
                f"{key} natural L2",
            )

            assert_close(
                result["natural_cosine"],
                previous_row["natural_cosine"],
                f"{key} natural cosine",
            )

            expected_retention = (
                (
                    float(
                        previous_row["at2_node_variance"]
                    )
                    /
                    node_variance(x_at2)
                )
                +
                (
                    float(
                        previous_row["gt450_node_variance"]
                    )
                    /
                    node_variance(x_gt450)
                )
            ) / 2.0

            assert_close(
                result["variance_retention"],
                expected_retention,
                f"{key} variance retention",
            )


        # ====================================================
        # MATCH SPATIAL SMOOTHING TO GLOBAL
        # ====================================================

        target_retention = (
            global_original["variance_retention"]
        )

        calibration = find_matching_alpha(
            spatial_at2,
            spatial_gt450,
            x_at2,
            x_gt450,
            target_retention,
        )

        selected_alpha = calibration["alpha"]

        spatial_matched = evaluate_condition(
            spatial_at2,
            spatial_gt450,
            x_at2,
            x_gt450,
            alpha=selected_alpha,
        )

        assert_close(
            spatial_matched["variance_retention"],
            calibration["predicted_retention"],
            (
                f"{region_id}, budget {budget}, "
                "calibration curve"
            ),
        )

        retention_error = abs(
            spatial_matched["variance_retention"]
            -
            target_retention
        )

        match_acceptable = (
            retention_error
            <= RETENTION_WARNING_TOLERANCE
        )

        original_advantage = (
            global_original["topology_l2"]
            -
            spatial_original["topology_l2"]
        )

        matched_advantage = (
            global_original["topology_l2"]
            -
            spatial_matched["topology_l2"]
        )

        records.append({
            "region_id": region_id,
            "edge_budget": budget,

            "global_alpha": 1.0,
            "spatial_original_alpha": 1.0,
            "spatial_matched_alpha": selected_alpha,

            "global_variance_retention":
                global_original["variance_retention"],

            "spatial_original_variance_retention":
                spatial_original["variance_retention"],

            "spatial_matched_variance_retention":
                spatial_matched["variance_retention"],

            "retention_absolute_error":
                retention_error,

            "retention_match_acceptable":
                match_acceptable,

            "target_within_grid_range":
                calibration["target_within_grid_range"],

            "global_topology_l2":
                global_original["topology_l2"],

            "spatial_original_topology_l2":
                spatial_original["topology_l2"],

            "spatial_matched_topology_l2":
                spatial_matched["topology_l2"],

            "original_topology_advantage":
                original_advantage,

            "matched_topology_advantage":
                matched_advantage,

            "global_topology_cosine":
                global_original["topology_cosine"],

            "spatial_original_topology_cosine":
                spatial_original["topology_cosine"],

            "spatial_matched_topology_cosine":
                spatial_matched["topology_cosine"],

            "global_natural_l2":
                global_original["natural_l2"],

            "spatial_original_natural_l2":
                spatial_original["natural_l2"],

            "spatial_matched_natural_l2":
                spatial_matched["natural_l2"],

            "global_natural_cosine":
                global_original["natural_cosine"],

            "spatial_original_natural_cosine":
                spatial_original["natural_cosine"],

            "spatial_matched_natural_cosine":
                spatial_matched["natural_cosine"],
        })

    print(
        f"  {region_id}: completed",
        flush=True,
    )


# ============================================================
# STEP 3: SUMMARIZE RESULTS
# ============================================================

print(
    "\n[3] Summarizing smoothing-matched results...",
    flush=True,
)

details = pd.DataFrame(records)

if len(details) != 30:
    raise RuntimeError(
        "Expected 30 region-budget comparisons."
    )

summary = (
    details.groupby(
        "edge_budget",
        as_index=False,
    )
    .agg(
        regions=(
            "region_id",
            "nunique",
        ),

        mean_spatial_alpha=(
            "spatial_matched_alpha",
            "mean",
        ),

        min_spatial_alpha=(
            "spatial_matched_alpha",
            "min",
        ),

        max_spatial_alpha=(
            "spatial_matched_alpha",
            "max",
        ),

        mean_retention_error=(
            "retention_absolute_error",
            "mean",
        ),

        max_retention_error=(
            "retention_absolute_error",
            "max",
        ),

        acceptable_matches=(
            "retention_match_acceptable",
            "sum",
        ),

        mean_global_retention=(
            "global_variance_retention",
            "mean",
        ),

        mean_matched_spatial_retention=(
            "spatial_matched_variance_retention",
            "mean",
        ),

        mean_global_topology_l2=(
            "global_topology_l2",
            "mean",
        ),

        mean_original_spatial_l2=(
            "spatial_original_topology_l2",
            "mean",
        ),

        mean_matched_spatial_l2=(
            "spatial_matched_topology_l2",
            "mean",
        ),

        mean_original_advantage=(
            "original_topology_advantage",
            "mean",
        ),

        mean_matched_advantage=(
            "matched_topology_advantage",
            "mean",
        ),

        matched_positive_regions=(
            "matched_topology_advantage",
            lambda values: int(
                (values > 0).sum()
            ),
        ),

        mean_global_natural_l2=(
            "global_natural_l2",
            "mean",
        ),

        mean_matched_spatial_natural_l2=(
            "spatial_matched_natural_l2",
            "mean",
        ),
    )
)

print(
    "\nSmoothing-matched summary:",
    flush=True,
)

print(
    summary.to_string(index=False),
    flush=True,
)

print(
    "\nPer-budget interpretation:",
    flush=True,
)

for row in summary.itertuples(index=False):

    print(
        f"\nEdge budget: {row.edge_budget}",
        flush=True,
    )

    print(
        f"  Mean calibrated alpha: "
        f"{row.mean_spatial_alpha:.4f}",
        flush=True,
    )

    print(
        f"  Mean global topology L2: "
        f"{row.mean_global_topology_l2:.6f}",
        flush=True,
    )

    print(
        f"  Mean original spatial topology L2: "
        f"{row.mean_original_spatial_l2:.6f}",
        flush=True,
    )

    print(
        f"  Mean matched spatial topology L2: "
        f"{row.mean_matched_spatial_l2:.6f}",
        flush=True,
    )

    print(
        f"  Original spatial advantage: "
        f"{row.mean_original_advantage:.6f}",
        flush=True,
    )

    print(
        f"  Matched spatial advantage: "
        f"{row.mean_matched_advantage:.6f}",
        flush=True,
    )

    print(
        f"  Regions with positive matched advantage: "
        f"{row.matched_positive_regions}/10",
        flush=True,
    )

    print(
        f"  Acceptable variance matches: "
        f"{row.acceptable_matches}/10",
        flush=True,
    )


# ============================================================
# STEP 4: SAVE OUTPUTS
# ============================================================

print(
    "\n[4] Saving research outputs...",
    flush=True,
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

refuse_overwrite(
    (
        DETAIL_PATH,
        SUMMARY_PATH,
        REPORT_PATH,
    )
)

details.to_csv(
    DETAIL_PATH,
    index=False,
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)

report = {
    "phase": "3B",
    "step": "3B.26",

    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "manifest_sha256": manifest_hash,

    "feature_archive_sha256":
        sha256_file(FEATURES),

    "phase3b21_sha256":
        sha256_file(PREVIOUS_CONNECTED),

    "phase3b23_sha256":
        sha256_file(PREVIOUS_PROPAGATION),

    "region_count": len(regions),

    "nodes_per_region": NODES_PER_REGION,

    "shared_backbone_edges": BACKBONE_EDGES,

    "edge_budgets": list(EDGE_BUDGETS),

    "propagation_depth": PROPAGATION_DEPTH,

    "alpha_range": [0.0, 1.0],

    "alpha_step": 0.001,

    "comparison_rows": len(details),

    "calibration_target": (
        "Mean AT2 and GT450 node-variance retention "
        "of the global graph at alpha=1"
    ),

    "calibration_strategy": (
        "One spatial alpha per region and edge budget; "
        "same alpha used for both scanners"
    ),

    "maximum_retention_error": float(
        details["retention_absolute_error"].max()
    ),

    "acceptable_retention_tolerance":
        RETENTION_WARNING_TOLERANCE,

    "acceptable_retention_matches": int(
        details["retention_match_acceptable"].sum()
    ),

    "graph_reconstruction_verified": True,

    "propagation_reconstruction_verified": True,

    "trained_gnn_used": False,

    "original_images_modified": False,

    "original_features_modified": False,

    "frozen_manifest_modified": False,

    "biological_independence_verified": False,

    "manual_registration_review_completed": False,

    "limitations": [
        "Calibration is performed on evaluation-region features",
        "This is an exploratory oracle-matched control",
        "Equal variance retention does not prove equal information preservation",
        "Different propagation strengths may affect topology sensitivity",
        "Global and spatial graphs have different edge candidate pools",
        "Ten regions belong to one registered acquisition pair",
        "No independently validated patient-level replication",
        "No trained GNN or downstream diagnostic task evaluated",
    ],
}

REPORT_PATH.write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 70,
    flush=True,
)

print(
    "PHASE 3B.26 - SMOOTHING MATCH AUDIT COMPLETED",
    flush=True,
)

print(
    "=" * 70,
    flush=True,
)

print(
    "Regions:",
    len(regions),
    flush=True,
)

print(
    "Paired comparisons:",
    len(details),
    flush=True,
)

print(
    "Maximum variance-retention mismatch:",
    f"{details['retention_absolute_error'].max():.6f}",
    flush=True,
)

print(
    "Acceptable variance matches:",
    f"{int(details['retention_match_acceptable'].sum())}/30",
    flush=True,
)

print("\nSaved:", flush=True)

print(DETAIL_PATH, flush=True)
print(SUMMARY_PATH, flush=True)
print(REPORT_PATH, flush=True)

print(
    "\nFrozen manifest: UNCHANGED",
    flush=True,
)

print(
    "Original features: UNCHANGED",
    flush=True,
)

print(
    "Trained GNN: NOT USED",
    flush=True,
)

print(
    "Biological independence: UNVERIFIED",
    flush=True,
)

if not details["retention_match_acceptable"].all():

    print(
        "\nWARNING: Some configurations could not "
        "match variance retention within tolerance. "
        "Inspect their calibration errors before "
        "interpreting matched comparisons.",
        flush=True,
    )

print(
    "\nSTEP 3B.26 COMPLETED",
    flush=True,
)
