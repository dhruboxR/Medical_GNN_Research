
"""
Phase 3A - Step 3.13
Cross-Scanner Graph Representation Consistency

Deterministic graph propagation:
    H_0 = L2_normalize(X)
    H_(l+1) = D^-1 (A + I) H_l

Experiments:
- Natural scanner comparison: AT2 features/edges
  versus GT450 features/edges
- Fixed AT2 features, changing graph topology
- Fixed GT450 features, changing graph topology
- Backbone-only structural control

No GNN training, downloads, or network access.
"""

from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd


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

DEPTHS = [0, 1, 2, 3]
EPS = 1e-12

print("=" * 70, flush=True)
print("PHASE 3A - GRAPH REPRESENTATION CONSISTENCY", flush=True)
print("=" * 70, flush=True)


def normalize_rows(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, EPS)


def cosine_rows(a, b):
    a = normalize_rows(a)
    b = normalize_rows(b)
    return np.sum(a * b, axis=1)


def cosine_vector(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    return float(
        np.dot(a, b) / max(na * nb, EPS)
    )


def adjacency_operator(n, edges):
    """
    Row-normalized adjacency with self-loops.
    """
    matrix = np.eye(n, dtype=np.float64)

    for i, j in edges:
        matrix[i, j] = 1.0
        matrix[j, i] = 1.0

    degrees = matrix.sum(axis=1)

    return matrix / degrees[:, None]


def propagate(operator, features, depth):
    h = features.copy()

    for _ in range(depth):
        h = operator @ h

    return h


def compare_representations(a, b):
    similarities = cosine_rows(a, b)

    # Same node locations in both representations.
    differences = np.linalg.norm(a - b, axis=1)

    graph_a = a.mean(axis=0)
    graph_b = b.mean(axis=0)

    return {
        "mean_node_cosine": float(similarities.mean()),
        "min_node_cosine": float(similarities.min()),
        "mean_node_l2": float(differences.mean()),
        "graph_mean_pool_cosine": cosine_vector(
            graph_a, graph_b
        ),
        "graph_mean_pool_l2": float(
            np.linalg.norm(graph_a - graph_b)
        ),
    }


# ============================================================
# STEP 1: LOAD FEATURES
# ============================================================

print("\n[1] Loading scanner features...", flush=True)

if not FEATURE_FILE.is_file():
    raise SystemExit(f"Missing features: {FEATURE_FILE}")

with np.load(FEATURE_FILE) as data:
    at2 = data["at2"].astype(np.float64)
    gt450 = data["gt450"].astype(np.float64)
    tile_ids = data["tile_ids"].astype(str)

if at2.shape != gt450.shape:
    raise ValueError("Feature shapes differ.")

if len(set(tile_ids)) != len(tile_ids):
    raise ValueError("Duplicate node IDs.")

if not (
    np.isfinite(at2).all()
    and np.isfinite(gt450).all()
):
    raise ValueError("Non-finite features.")

# Normalize feature magnitudes for fair cosine-based
# propagation comparisons.
at2 = normalize_rows(at2)
gt450 = normalize_rows(gt450)

n, d = at2.shape

node_index = {
    name: index
    for index, name in enumerate(tile_ids)
}

print(f"Nodes: {n}")
print(f"Feature dimension: {d}")


# ============================================================
# STEP 2: RECONSTRUCT GRAPHS
# ============================================================

print("\n[2] Reconstructing connected graphs...", flush=True)

if not EDGE_FILE.is_file():
    raise SystemExit(f"Missing edges: {EDGE_FILE}")

edges_df = pd.read_csv(EDGE_FILE)

required = {
    "method", "total_edge_budget", "scanner",
    "source_tile_id", "target_tile_id", "edge_role"
}

missing = required - set(edges_df.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

graphs = {}
backbones = {}

for key, group in edges_df.groupby(
    ["method", "total_edge_budget", "scanner"]
):
    method, budget, scanner = key
    budget = int(budget)

    edges = set()
    backbone = set()

    for row in group.itertuples(index=False):
        i = node_index[str(row.source_tile_id)]
        j = node_index[str(row.target_tile_id)]

        if i == j:
            raise ValueError("Self-loop in saved edges.")

        edge = tuple(sorted((i, j)))

        if edge in edges:
            raise ValueError("Duplicate graph edge.")

        edges.add(edge)

        if row.edge_role == "backbone":
            backbone.add(edge)

    if len(edges) != budget:
        raise ValueError("Incorrect edge budget.")

    if len(backbone) != n - 1:
        raise ValueError("Incorrect backbone size.")

    graphs[(method, budget, scanner)] = edges
    backbones[(method, budget, scanner)] = backbone

print(f"Graphs loaded: {len(graphs)}")

reference_backbone = next(iter(backbones.values()))

if not all(
    b == reference_backbone
    for b in backbones.values()
):
    raise ValueError("Backbones are not identical.")

backbone_operator = adjacency_operator(
    n, reference_backbone
)


# ============================================================
# STEP 3: SET UP EXPERIMENTS
# ============================================================

print("\n[3] Preparing graph propagation experiments...")

configurations = sorted({
    (method, budget)
    for method, budget, scanner in graphs
})

experiments = [
    "natural_cross_scanner",
    "fixed_at2_features_topology_change",
    "fixed_gt450_features_topology_change",
]

results = []


# ============================================================
# STEP 4: RUN DETERMINISTIC PROPAGATION
# ============================================================

print("\n[4] Evaluating representation consistency...")

for method, budget in configurations:

    operator_a = adjacency_operator(
        n, graphs[(method, budget, "AT2")]
    )

    operator_b = adjacency_operator(
        n, graphs[(method, budget, "GT450")]
    )

    print(
        f"\n{method} | Budget={budget}",
        flush=True
    )

    for depth in DEPTHS:

        # Natural cross-scanner comparison.
        natural_a = propagate(
            operator_a, at2, depth
        )
        natural_b = propagate(
            operator_b, gt450, depth
        )

        # Isolate graph topology changes with identical
        # input node features.
        topology_at2_a = propagate(
            operator_a, at2, depth
        )
        topology_at2_b = propagate(
            operator_b, at2, depth
        )

        topology_gt_a = propagate(
            operator_a, gt450, depth
        )
        topology_gt_b = propagate(
            operator_b, gt450, depth
        )

        comparisons = {
            "natural_cross_scanner": (
                natural_a, natural_b
            ),
            "fixed_at2_features_topology_change": (
                topology_at2_a, topology_at2_b
            ),
            "fixed_gt450_features_topology_change": (
                topology_gt_a, topology_gt_b
            ),
        }

        for experiment, (x, y) in comparisons.items():

            metrics = compare_representations(x, y)

            results.append({
                "method": method,
                "edge_budget": budget,
                "depth": depth,
                "experiment": experiment,
                **metrics,
            })

        natural_score = compare_representations(
            natural_a, natural_b
        )["mean_node_cosine"]

        topology_score = compare_representations(
            topology_at2_a, topology_at2_b
        )["mean_node_cosine"]

        print(
            f"  Depth {depth}: "
            f"natural={natural_score:.6f}, "
            f"fixed-feature topology={topology_score:.6f}"
        )


# ============================================================
# STEP 5: BACKBONE-ONLY CONTROL
# ============================================================

print("\n[5] Running backbone-only control...")

for depth in DEPTHS:

    h_a = propagate(
        backbone_operator, at2, depth
    )

    h_b = propagate(
        backbone_operator, gt450, depth
    )

    metrics = compare_representations(h_a, h_b)

    results.append({
        "method": "Backbone-only",
        "edge_budget": n - 1,
        "depth": depth,
        "experiment": "natural_cross_scanner",
        **metrics,
    })

    print(
        f"  Depth {depth}: "
        f"mean node cosine="
        f"{metrics['mean_node_cosine']:.6f}"
    )


# ============================================================
# STEP 6: SAVE RESULTS
# ============================================================

print("\n[6] Saving results...", flush=True)

results_df = pd.DataFrame(results)

csv_path = (
    TABLE_DIR / "plism_graph_representation_consistency.csv"
)

results_df.to_csv(csv_path, index=False)

report = {
    "phase": "3A",
    "step": "3.13",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "dataset": "PLISM",
    "encoder": "facebook/dinov2-small",
    "nodes": n,
    "feature_dimension": d,
    "propagation_depths": DEPTHS,
    "method": (
        "Deterministic row-normalized adjacency "
        "propagation with self-loops"
    ),
    "input_feature_normalization": "Row-wise L2",
    "graph_readout": "Mean pooling",
    "trained_gnn": False,
    "results": results_df.to_dict(orient="records"),
    "limitations": [
        "Single tissue region",
        "No learned graph weights",
        "Mean pooling may hide local representation changes",
        "Increasing propagation depth may oversmooth features",
        "No independent biological replication",
        "Higher representation similarity does not imply "
        "better classification performance"
    ],
}

json_path = (
    REPORT_DIR / "13_graph_representation_consistency.json"
)

json_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GRAPH REPRESENTATION EXPERIMENT COMPLETED")
print("=" * 70)

print("\nNatural cross-scanner comparison at depth 2:")

summary = results_df[
    (results_df["depth"] == 2)
    &
    (
        results_df["experiment"]
        == "natural_cross_scanner"
    )
]

print(
    summary[
        [
            "method",
            "edge_budget",
            "mean_node_cosine",
            "mean_node_l2",
            "graph_mean_pool_cosine"
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print(csv_path)
print(json_path)

print("\nSTEP 3.13 COMPLETED")
