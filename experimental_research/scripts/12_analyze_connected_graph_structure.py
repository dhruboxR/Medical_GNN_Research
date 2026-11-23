
"""
Phase 3A - Step 3.12
Connected Graph Spatial and Node-Level Diagnostics

Input:
    Step 3.11 connected graph edges
    Phase 1 matched-node feature inventory

Output:
    1. Per-graph structural diagnostics
    2. Per-node cross-scanner stability
    3. JSON summary

Focus on feature-selected residual edges.
The common spatial backbone is analyzed separately.

No downloads or GNN training.
"""

from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

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

SCANNERS = ("AT2", "GT450")
SPATIAL_NEIGHBOR_DISTANCE = np.sqrt(2) + 1e-8


# ============================================================
# GRAPH HELPERS
# ============================================================

def build_adjacency(n, edges):
    adjacency = [set() for _ in range(n)]

    for i, j in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)

    return adjacency


def graph_components(n, edges):
    adjacency = build_adjacency(n, edges)

    visited = set()
    components = []

    for node in range(n):
        if node in visited:
            continue

        stack = [node]
        visited.add(node)
        component = []

        while stack:
            current = stack.pop()
            component.append(current)

            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        components.append(component)

    return components


def node_jaccard(a, b):
    union = a | b

    if not union:
        return 1.0

    return len(a & b) / len(union)


def spatial_distance(coords, edge):
    i, j = edge

    return float(
        np.linalg.norm(coords[i] - coords[j])
    )


def calculate_graph_metrics(n, coords, full, backbone, residual):

    full_adj = build_adjacency(n, full)
    residual_adj = build_adjacency(n, residual)

    full_degree = np.array(
        [len(x) for x in full_adj],
        dtype=int
    )

    residual_degree = np.array(
        [len(x) for x in residual_adj],
        dtype=int
    )

    residual_distances = np.array(
        [
            spatial_distance(coords, edge)
            for edge in sorted(residual)
        ],
        dtype=float
    )

    nonlocal_count = int(
        np.sum(
            residual_distances > SPATIAL_NEIGHBOR_DISTANCE
        )
    )

    components = graph_components(n, full)

    return {
        "nodes": n,
        "total_edges": len(full),
        "backbone_edges": len(backbone),
        "residual_edges": len(residual),
        "connected_components": len(components),
        "isolated_full_nodes": int(
            np.sum(full_degree == 0)
        ),
        "zero_residual_degree_nodes": int(
            np.sum(residual_degree == 0)
        ),
        "mean_full_degree": float(full_degree.mean()),
        "mean_residual_degree": float(
            residual_degree.mean()
        ),
        "min_residual_degree": int(
            residual_degree.min()
        ),
        "max_residual_degree": int(
            residual_degree.max()
        ),
        "mean_residual_spatial_distance": float(
            residual_distances.mean()
        ),
        "median_residual_spatial_distance": float(
            np.median(residual_distances)
        ),
        "max_residual_spatial_distance": float(
            residual_distances.max()
        ),
        "nonlocal_residual_edges": nonlocal_count,
        "nonlocal_residual_fraction": (
            nonlocal_count / len(residual)
        ),
        "largest_component_nodes": max(
            len(component)
            for component in components
        )
    }


# ============================================================
# STEP 1: LOAD NODE INVENTORY
# ============================================================

print("=" * 70, flush=True)
print("PHASE 3A - CONNECTED GRAPH STRUCTURAL DIAGNOSTICS", flush=True)
print("=" * 70, flush=True)

print("\n[1] Loading matched node inventory...", flush=True)

if not FEATURE_FILE.is_file():
    raise SystemExit(f"Missing: {FEATURE_FILE}")

if not EDGE_FILE.is_file():
    raise SystemExit(f"Missing: {EDGE_FILE}")

with np.load(FEATURE_FILE) as data:
    tile_ids = data["tile_ids"].astype(str)
    rows = data["grid_rows"].astype(int)
    cols = data["grid_cols"].astype(int)

n = len(tile_ids)
coords = np.column_stack([rows, cols])

if len(set(tile_ids)) != n:
    raise ValueError("Duplicate tile identifiers.")

if len(np.unique(coords, axis=0)) != n:
    raise ValueError("Duplicate grid coordinates.")

tile_to_index = {
    tile_id: index
    for index, tile_id in enumerate(tile_ids)
}

print(f"Matched nodes: {n}")


# ============================================================
# STEP 2: LOAD CONNECTED GRAPH EDGES
# ============================================================

print("\n[2] Loading Step 3.11 graph edges...", flush=True)

df = pd.read_csv(EDGE_FILE)

required_columns = {
    "method",
    "total_edge_budget",
    "scanner",
    "source_tile_id",
    "target_tile_id",
    "edge_role"
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing edge columns: {sorted(missing)}"
    )

df["total_edge_budget"] = (
    df["total_edge_budget"].astype(int)
)

graphs = {}

for key, group in df.groupby(
    ["method", "total_edge_budget", "scanner"]
):

    full = set()
    backbone = set()
    residual = set()

    for record in group.itertuples(index=False):

        a = str(record.source_tile_id)
        b = str(record.target_tile_id)

        if a not in tile_to_index or b not in tile_to_index:
            raise ValueError(
                f"Unknown node in graph {key}"
            )

        i = tile_to_index[a]
        j = tile_to_index[b]

        if i == j:
            raise ValueError(
                f"Self-loop detected: {key}"
            )

        edge = tuple(sorted((i, j)))

        if edge in full:
            raise ValueError(
                f"Duplicate edge in graph {key}"
            )

        full.add(edge)

        if record.edge_role == "backbone":
            backbone.add(edge)

        elif record.edge_role == "feature_selected":
            residual.add(edge)

        else:
            raise ValueError(
                f"Unknown edge role: {record.edge_role}"
            )

    if full != backbone | residual:
        raise ValueError(
            f"Edge partition mismatch: {key}"
        )

    if len(full) != key[1]:
        raise ValueError(
            f"Incorrect edge budget: {key}"
        )

    if len(backbone) != n - 1:
        raise ValueError(
            f"Unexpected backbone size: {key}"
        )

    if len(graph_components(n, full)) != 1:
        raise ValueError(
            f"Disconnected graph: {key}"
        )

    graphs[key] = {
        "full": full,
        "backbone": backbone,
        "residual": residual
    }

print(f"Graphs loaded: {len(graphs)}")


# ============================================================
# STEP 3: VERIFY COMMON BACKBONE
# ============================================================

print("\n[3] Checking common backbone...", flush=True)

reference_backbone = next(
    iter(graphs.values())
)["backbone"]

for key, graph in graphs.items():

    if graph["backbone"] != reference_backbone:
        raise ValueError(
            f"Backbone differs in {key}"
        )

print(f"Common backbone edges: {len(reference_backbone)}")
print("Backbone consistency: PASSED")


# ============================================================
# STEP 4: PER-GRAPH STRUCTURAL METRICS
# ============================================================

print("\n[4] Computing graph structure metrics...", flush=True)

graph_records = []

for key in sorted(graphs):

    method, budget, scanner = key

    graph = graphs[key]

    metrics = calculate_graph_metrics(
        n,
        coords,
        graph["full"],
        graph["backbone"],
        graph["residual"]
    )

    record = {
        "method": method,
        "edge_budget": budget,
        "scanner": scanner,
        **metrics
    }

    graph_records.append(record)

    print(f"\n{method} | Budget={budget} | {scanner}")

    print(
        "  Zero residual-degree nodes: "
        f"{metrics['zero_residual_degree_nodes']}"
    )

    print(
        "  Mean residual edge distance: "
        f"{metrics['mean_residual_spatial_distance']:.4f}"
    )

    print(
        "  Nonlocal residual edges: "
        f"{metrics['nonlocal_residual_edges']}"
    )

graph_df = pd.DataFrame(graph_records)


# ============================================================
# STEP 5: CROSS-SCANNER NODE STABILITY
# ============================================================

print("\n[5] Comparing residual neighbors per node...", flush=True)

node_records = []
comparison_records = []

configurations = sorted(
    {
        (method, budget)
        for method, budget, scanner in graphs
    }
)

for method, budget in configurations:

    a = graphs[(method, budget, "AT2")]
    b = graphs[(method, budget, "GT450")]

    neighbors_a = build_adjacency(
        n,
        a["residual"]
    )

    neighbors_b = build_adjacency(
        n,
        b["residual"]
    )

    node_scores = []
    degree_differences = []
    both_zero = 0

    for index in range(n):

        na = neighbors_a[index]
        nb = neighbors_b[index]

        score = node_jaccard(na, nb)

        # Both empty neighbor sets are assigned 1.0
        # by convention, but are reported separately.
        if not na and not nb:
            both_zero += 1

        node_scores.append(score)

        degree_difference = abs(len(na) - len(nb))
        degree_differences.append(degree_difference)

        node_records.append({
            "method": method,
            "edge_budget": budget,
            "tile_id": tile_ids[index],
            "grid_row": int(rows[index]),
            "grid_col": int(cols[index]),
            "at2_residual_degree": len(na),
            "gt450_residual_degree": len(nb),
            "degree_difference": degree_difference,
            "shared_residual_neighbors": len(na & nb),
            "union_residual_neighbors": len(na | nb),
            "residual_neighbor_jaccard": score,
            "both_residual_neighborhoods_empty":
                not na and not nb
        })

    scores = np.array(node_scores)

    comparison_records.append({
        "method": method,
        "edge_budget": budget,
        "mean_node_residual_jaccard": float(
            scores.mean()
        ),
        "median_node_residual_jaccard": float(
            np.median(scores)
        ),
        "mean_absolute_degree_difference": float(
            np.mean(degree_differences)
        ),
        "both_empty_residual_nodes": both_zero,
        "mean_node_jaccard_nonempty_union": float(
            np.mean([
                node_scores[i]
                for i in range(n)
                if neighbors_a[i] or neighbors_b[i]
            ])
        ) if both_zero < n else None
    })

comparison_df = pd.DataFrame(comparison_records)

print("\nNode-level comparison:")

print(
    comparison_df[
        [
            "method",
            "edge_budget",
            "mean_node_residual_jaccard",
            "mean_node_jaccard_nonempty_union",
            "both_empty_residual_nodes",
            "mean_absolute_degree_difference"
        ]
    ].to_string(index=False)
)


# ============================================================
# STEP 6: SAVE OUTPUTS
# ============================================================

print("\n[6] Saving diagnostic outputs...", flush=True)

graph_path = (
    TABLE_DIR / "plism_connected_graph_structure.csv"
)

node_path = (
    TABLE_DIR / "plism_connected_graph_node_stability.csv"
)

comparison_path = (
    TABLE_DIR / "plism_connected_graph_node_summary.csv"
)

graph_df.to_csv(graph_path, index=False)

pd.DataFrame(node_records).to_csv(
    node_path,
    index=False
)

comparison_df.to_csv(
    comparison_path,
    index=False
)

report = {
    "phase": "3A",
    "step": "3.12",
    "generated_utc":
        datetime.now(timezone.utc).isoformat(),
    "nodes": n,
    "graphs_checked": len(graphs),
    "shared_backbone_edges": len(reference_backbone),
    "all_graphs_connected": True,
    "spatial_distance_units":
        "Inferred tile-grid coordinate units",
    "both_empty_neighbor_jaccard_convention": 1.0,
    "graph_structure": graph_records,
    "node_stability_summary":
        comparison_records,
    "limitations": [
        "One tissue region only",
        "Tile coordinates are provisional",
        "Shared backbone influences full graph topology",
        "Both-empty residual neighborhoods are reported separately",
        "No downstream GNN evaluation yet"
    ]
}

report_path = (
    REPORT_DIR / "12_connected_graph_structure.json"
)

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("GRAPH STRUCTURAL DIAGNOSTICS COMPLETED")
print("=" * 70)

print(f"Nodes: {n}")
print(f"Graphs analyzed: {len(graphs)}")
print("All graphs connected: True")
print(f"Shared backbone edges: {len(reference_backbone)}")

print("\nSaved outputs:")
print(graph_path)
print(node_path)
print(comparison_path)
print(report_path)

print("\nSTEP 3.12 COMPLETED")
