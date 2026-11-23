
"""
PHASE 3A - STEP 3.14

Oversmoothing and Topology Sensitivity Analysis

Measures:
1. Within-scanner node embedding dispersion
2. Mean pairwise node cosine similarity
3. Average distance from graph-mean embedding
4. Cross-scanner node representation similarity
5. Fixed-feature topology perturbation magnitude
6. Topology perturbation relative to natural
   cross-scanner representation differences

Uses existing DINOv2 features and Step 3.11 graphs.
No model training or network access.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


# ==========================================================
# CONFIGURATION
# ==========================================================

ROOT = Path.cwd()

FEATURE_FILE = (
    ROOT / "dataset_exploration/outputs/features/"
    "plism_dinov2_features.npz"
)

EDGE_FILE = (
    ROOT / "experimental_research/outputs/tables/"
    "plism_connected_graph_edges.csv"
)

OUTPUT = ROOT / "experimental_research/outputs"

TABLE_DIR = OUTPUT / "tables"
REPORT_DIR = OUTPUT / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DEPTHS = [0, 1, 2, 3, 4, 5]
EPS = 1e-12


# ==========================================================
# FUNCTIONS
# ==========================================================

def normalize_rows(x):
    norms = np.linalg.norm(
        x, axis=1, keepdims=True
    )
    return x / np.maximum(norms, EPS)


def mean_node_cosine(a, b):
    a = normalize_rows(a)
    b = normalize_rows(b)

    return float(
        np.mean(np.sum(a * b, axis=1))
    )


def mean_node_l2(a, b):
    return float(
        np.linalg.norm(a - b, axis=1).mean()
    )


def embedding_dispersion(x):
    """
    Measures how distinct node embeddings remain.

    Higher mean pairwise cosine suggests nodes
    are becoming more similar.

    Dispersion is measured around the graph mean.
    """
    normalized = normalize_rows(x)

    n = len(normalized)

    similarity = normalized @ normalized.T

    upper = similarity[
        np.triu_indices(n, k=1)
    ]

    center = normalized.mean(axis=0)

    mean_center_distance = float(
        np.linalg.norm(
            normalized - center,
            axis=1
        ).mean()
    )

    raw_center = x.mean(axis=0)

    raw_rms_dispersion = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (x - raw_center) ** 2,
                    axis=1
                )
            )
        )
    )

    return {
        "mean_pairwise_node_cosine":
            float(upper.mean()),
        "mean_center_distance":
            mean_center_distance,
        "raw_rms_dispersion":
            raw_rms_dispersion,
    }


def graph_operator(n, edges):
    adjacency = np.eye(
        n, dtype=np.float64
    )

    for i, j in edges:
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0

    degree = adjacency.sum(axis=1)

    return adjacency / degree[:, None]


def propagate(operator, features, depth):
    result = features.copy()

    for _ in range(depth):
        result = operator @ result

    return result


def read_graph_edges(group, index):
    edges = set()
    backbone = set()

    for row in group.itertuples(index=False):
        i = index[str(row.source_tile_id)]
        j = index[str(row.target_tile_id)]

        if i == j:
            raise ValueError("Self-loop detected")

        edge = tuple(sorted((i, j)))

        if edge in edges:
            raise ValueError("Duplicate edge")

        edges.add(edge)

        if row.edge_role == "backbone":
            backbone.add(edge)

    return edges, backbone


# ==========================================================
# STEP 1: LOAD FEATURES
# ==========================================================

print("=" * 70, flush=True)
print("PHASE 3A - OVERSMOOTHING AND TOPOLOGY SENSITIVITY")
print("=" * 70, flush=True)

print("\n[1] Loading features...", flush=True)

if not FEATURE_FILE.exists():
    raise SystemExit(f"Missing: {FEATURE_FILE}")

with np.load(FEATURE_FILE) as data:
    at2 = data["at2"].astype(np.float64)
    gt450 = data["gt450"].astype(np.float64)
    tile_ids = data["tile_ids"].astype(str)

if at2.shape != gt450.shape:
    raise ValueError("Feature shape mismatch")

if len(set(tile_ids)) != len(tile_ids):
    raise ValueError("Duplicate tile IDs")

if not (
    np.isfinite(at2).all()
    and np.isfinite(gt450).all()
):
    raise ValueError("Invalid feature values")

at2 = normalize_rows(at2)
gt450 = normalize_rows(gt450)

n, dim = at2.shape

index = {
    tile_id: i
    for i, tile_id in enumerate(tile_ids)
}

print(f"Nodes: {n}")
print(f"Feature dimensions: {dim}")


# ==========================================================
# STEP 2: LOAD GRAPH CONFIGURATIONS
# ==========================================================

print("\n[2] Loading connected graphs...", flush=True)

if not EDGE_FILE.exists():
    raise SystemExit(f"Missing: {EDGE_FILE}")

df = pd.read_csv(EDGE_FILE)

graphs = {}
backbones = {}

for key, group in df.groupby(
    ["method", "total_edge_budget", "scanner"]
):
    edges, backbone = read_graph_edges(
        group, index
    )

    if len(edges) != int(key[1]):
        raise ValueError(
            f"Incorrect edge count: {key}"
        )

    if len(backbone) != n - 1:
        raise ValueError(
            f"Incorrect backbone size: {key}"
        )

    graphs[key] = edges
    backbones[key] = backbone

reference_backbone = next(
    iter(backbones.values())
)

if not all(
    backbone == reference_backbone
    for backbone in backbones.values()
):
    raise ValueError("Backbones are inconsistent")

configurations = sorted({
    (method, budget)
    for method, budget, scanner in graphs
})

for method, budget in configurations:
    for scanner in ("AT2", "GT450"):
        if (method, budget, scanner) not in graphs:
            raise ValueError("Missing scanner graph")

print(f"Graph configurations: {len(configurations)}")
print(f"Loaded graphs: {len(graphs)}")


# ==========================================================
# STEP 3: RUN PROPAGATION ANALYSIS
# ==========================================================

print("\n[3] Measuring representation dispersion...", flush=True)

dispersion_records = []
sensitivity_records = []

methods = list(configurations)

methods.append(
    ("Backbone-only", n - 1)
)

for method, budget in methods:

    if method == "Backbone-only":
        edges_a = reference_backbone
        edges_b = reference_backbone
    else:
        edges_a = graphs[
            (method, budget, "AT2")
        ]
        edges_b = graphs[
            (method, budget, "GT450")
        ]

    op_a = graph_operator(n, edges_a)
    op_b = graph_operator(n, edges_b)

    print(
        f"\n{method} | budget={budget}",
        flush=True
    )

    for depth in DEPTHS:

        # Natural scanner-specific representations
        h_at2 = propagate(
            op_a, at2, depth
        )

        h_gt450 = propagate(
            op_b, gt450, depth
        )

        # Topology-only comparison:
        # same features, different graph structures
        h_at2_on_gt_graph = propagate(
            op_b, at2, depth
        )

        h_gt_on_at2_graph = propagate(
            op_a, gt450, depth
        )

        for scanner, h in [
            ("AT2", h_at2),
            ("GT450", h_gt450)
        ]:

            stats = embedding_dispersion(h)

            dispersion_records.append({
                "method": method,
                "edge_budget": budget,
                "scanner": scanner,
                "depth": depth,
                **stats
            })

        natural_cosine = mean_node_cosine(
            h_at2, h_gt450
        )

        natural_l2 = mean_node_l2(
            h_at2, h_gt450
        )

        topology_l2_at2 = mean_node_l2(
            h_at2, h_at2_on_gt_graph
        )

        topology_l2_gt450 = mean_node_l2(
            h_gt450, h_gt_on_at2_graph
        )

        topology_cosine_at2 = mean_node_cosine(
            h_at2, h_at2_on_gt_graph
        )

        topology_cosine_gt450 = mean_node_cosine(
            h_gt450, h_gt_on_at2_graph
        )

        avg_topology_l2 = (
            topology_l2_at2
            + topology_l2_gt450
        ) / 2

        # Descriptive ratio, not causal attribution.
        ratio = (
            avg_topology_l2 / natural_l2
            if natural_l2 > EPS
            else np.nan
        )

        sensitivity_records.append({
            "method": method,
            "edge_budget": budget,
            "depth": depth,
            "natural_cross_scanner_cosine":
                natural_cosine,
            "natural_cross_scanner_l2":
                natural_l2,
            "topology_only_cosine_at2":
                topology_cosine_at2,
            "topology_only_cosine_gt450":
                topology_cosine_gt450,
            "topology_only_l2_at2":
                topology_l2_at2,
            "topology_only_l2_gt450":
                topology_l2_gt450,
            "mean_topology_only_l2":
                avg_topology_l2,
            "topology_to_natural_l2_ratio":
                ratio
        })

        if depth in (0, 2, 5):
            print(
                f"  Depth {depth}: "
                f"natural cosine={natural_cosine:.6f}, "
                f"topology L2={avg_topology_l2:.6f}, "
                f"ratio={ratio:.4f}",
                flush=True
            )


# ==========================================================
# STEP 4: DISPERSION RETENTION
# ==========================================================

print("\n[4] Computing dispersion retention...", flush=True)

dispersion_df = pd.DataFrame(
    dispersion_records
)

sensitivity_df = pd.DataFrame(
    sensitivity_records
)

baseline = dispersion_df[
    dispersion_df["depth"] == 0
][
    ["method", "edge_budget", "scanner",
     "raw_rms_dispersion"]
].rename(
    columns={
        "raw_rms_dispersion":
            "depth0_rms_dispersion"
    }
)

dispersion_df = dispersion_df.merge(
    baseline,
    on=["method", "edge_budget", "scanner"],
    validate="many_to_one"
)

dispersion_df["dispersion_retention"] = (
    dispersion_df["raw_rms_dispersion"]
    /
    dispersion_df["depth0_rms_dispersion"]
    .clip(lower=EPS)
)


# ==========================================================
# STEP 5: SAVE RESULTS
# ==========================================================

print("\n[5] Saving results...", flush=True)

dispersion_path = (
    TABLE_DIR / "plism_oversmoothing_metrics.csv"
)

sensitivity_path = (
    TABLE_DIR / "plism_topology_sensitivity.csv"
)

dispersion_df.to_csv(
    dispersion_path,
    index=False
)

sensitivity_df.to_csv(
    sensitivity_path,
    index=False
)

report = {
    "phase": "3A",
    "step": "3.14",
    "nodes": n,
    "feature_dimension": dim,
    "depths": DEPTHS,
    "representation_operator":
        "Row-normalized adjacency with self-loops",
    "oversmoothing_indicators": [
        "Increasing within-graph pairwise cosine",
        "Decreasing embedding dispersion",
        "Decreasing dispersion retention"
    ],
    "topology_isolation":
        "Compare AT2/GT450 edges with fixed features",
    "interpretation_limitations": [
        "One tissue region",
        "No GNN training",
        "No classification labels",
        "No specimen-level statistical inference",
        "Topology-to-natural ratio is descriptive",
        "Smoothing is not proof of improved robustness"
    ],
    "depth2_sensitivity": sensitivity_df[
        sensitivity_df["depth"] == 2
    ].to_dict(orient="records")
}

report_path = (
    REPORT_DIR /
    "14_oversmoothing_topology_sensitivity.json"
)

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 70)
print("OVERSMOOTHING AND TOPOLOGY ANALYSIS COMPLETED")
print("=" * 70)

print("\nDepth-2 topology sensitivity:")

print(
    sensitivity_df[
        sensitivity_df["depth"] == 2
    ][
        [
            "method",
            "edge_budget",
            "natural_cross_scanner_cosine",
            "mean_topology_only_l2",
            "topology_to_natural_l2_ratio"
        ]
    ].to_string(index=False)
)

print("\nDepth-5 dispersion retention:")

print(
    dispersion_df[
        dispersion_df["depth"] == 5
    ][
        [
            "method",
            "edge_budget",
            "scanner",
            "mean_pairwise_node_cosine",
            "dispersion_retention"
        ]
    ].to_string(index=False)
)

print("\nSaved outputs:")
print(dispersion_path)
print(sensitivity_path)
print(report_path)

print("\nSTEP 3.14 COMPLETED")
