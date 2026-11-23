
"""
Phase 3A - Step 3.9
Controlled Cross-Scanner Graph Stability Experiment

Compare two methods at identical edge budgets:

1. Global feature-similarity graph
2. Spatially constrained feature-similarity graph

Uses existing DINOv2 embeddings from Phase 1.
No image downloads, remote requests, or GNN training.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path.cwd()

FEATURE_FILE = (
    ROOT / "dataset_exploration/outputs/features/"
    "plism_dinov2_features.npz"
)

OUTPUT = ROOT / "experimental_research/outputs"

TABLE_DIR = OUTPUT / "tables"
REPORT_DIR = OUTPUT / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EDGE_BUDGETS = [90, 135, 180, 225]

# 8-neighbor adjacency on the provisional tile grid
SPATIAL_RADIUS = np.sqrt(2) + 1e-8


# ============================================================
# GRAPH CONSTRUCTION FUNCTIONS
# ============================================================

def get_candidate_edges(coordinates, spatial=False):
    candidates = []

    n = len(coordinates)

    for i in range(n):
        for j in range(i + 1, n):

            if spatial:
                distance = np.linalg.norm(
                    coordinates[i] - coordinates[j]
                )

                if distance > SPATIAL_RADIUS:
                    continue

            candidates.append((i, j))

    return candidates


def build_top_m_graph(features, candidates, budget):
    """
    Select exactly M undirected candidate edges
    using descending cosine similarity.
    """

    if budget > len(candidates):
        raise ValueError(
            f"Budget {budget} exceeds "
            f"{len(candidates)} candidate edges."
        )

    similarity = cosine_similarity(features)

    ranked = sorted(
        candidates,
        key=lambda edge: (
            -similarity[edge[0], edge[1]],
            edge[0],
            edge[1]
        )
    )

    return set(ranked[:budget])


def edge_jaccard(a, b):
    union = a | b

    return (
        len(a & b) / len(union)
        if union else 1.0
    )


def neighbor_jaccard(n, edges_a, edges_b):
    neighbors_a = [set() for _ in range(n)]
    neighbors_b = [set() for _ in range(n)]

    for i, j in edges_a:
        neighbors_a[i].add(j)
        neighbors_a[j].add(i)

    for i, j in edges_b:
        neighbors_b[i].add(j)
        neighbors_b[j].add(i)

    scores = []

    for a, b in zip(neighbors_a, neighbors_b):
        union = a | b

        score = (
            len(a & b) / len(union)
            if union else 1.0
        )

        scores.append(score)

    return float(np.mean(scores))


def connected_components(n, edges):
    adjacency = [[] for _ in range(n)]

    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)

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

    return components


# ============================================================
# STEP 1: LOAD EMBEDDINGS
# ============================================================

print("=" * 70)
print("PHASE 3A - CONTROLLED GRAPH STABILITY EXPERIMENT")
print("=" * 70)

print("\n[1] Loading existing DINOv2 embeddings...")

if not FEATURE_FILE.exists():
    raise SystemExit(f"Missing features: {FEATURE_FILE}")

with np.load(FEATURE_FILE) as data:
    at2 = data["at2"].astype(np.float64)
    gt450 = data["gt450"].astype(np.float64)
    tile_ids = data["tile_ids"].astype(str)
    rows = data["grid_rows"].astype(int)
    cols = data["grid_cols"].astype(int)

if at2.shape != gt450.shape:
    raise ValueError("Scanner feature shapes differ.")

if len(set(tile_ids)) != len(tile_ids):
    raise ValueError("Duplicate tile IDs detected.")

coordinates = np.column_stack([rows, cols])

if len(np.unique(coordinates, axis=0)) != len(tile_ids):
    raise ValueError("Duplicate grid coordinates detected.")

n = len(tile_ids)

if not np.isfinite(at2).all() or not np.isfinite(gt450).all():
    raise ValueError("Non-finite feature values detected.")

print(f"Matched nodes: {n}")
print(f"Feature dimensions: {at2.shape[1]}")


# ============================================================
# STEP 2: CONSTRUCT CANDIDATE EDGE SETS
# ============================================================

print("\n[2] Constructing candidate edge sets...")

global_candidates = get_candidate_edges(
    coordinates,
    spatial=False
)

spatial_candidates = get_candidate_edges(
    coordinates,
    spatial=True
)

print(f"Global candidate edges: {len(global_candidates)}")
print(f"Spatial candidate edges: {len(spatial_candidates)}")

for budget in EDGE_BUDGETS:
    if budget > len(spatial_candidates):
        raise ValueError(
            f"Invalid budget {budget}: not enough "
            "spatial candidate edges."
        )


# ============================================================
# STEP 3: COMPARE MATCHED EDGE BUDGETS
# ============================================================

print("\n[3] Evaluating matched edge budgets...")

results = []
edge_records = []

methods = {
    "Global-feature": global_candidates,
    "Spatial-feature": spatial_candidates
}

for budget in EDGE_BUDGETS:

    print(f"\nEdge budget: {budget}")

    for method, candidates in methods.items():

        graph_a = build_top_m_graph(
            at2, candidates, budget
        )

        graph_b = build_top_m_graph(
            gt450, candidates, budget
        )

        assert len(graph_a) == budget
        assert len(graph_b) == budget

        shared = len(graph_a & graph_b)

        jaccard = edge_jaccard(graph_a, graph_b)

        neighbor_score = neighbor_jaccard(
            n, graph_a, graph_b
        )

        components_a = connected_components(
            n, graph_a
        )

        components_b = connected_components(
            n, graph_b
        )

        results.append({
            "method": method,
            "edge_budget": budget,
            "nodes": n,
            "at2_edges": len(graph_a),
            "gt450_edges": len(graph_b),
            "shared_edges": shared,
            "edge_jaccard": jaccard,
            "neighbor_jaccard": neighbor_score,
            "at2_components": components_a,
            "gt450_components": components_b
        })

        for scanner, graph in [
            ("AT2", graph_a),
            ("GT450", graph_b)
        ]:

            for i, j in sorted(graph):
                edge_records.append({
                    "method": method,
                    "edge_budget": budget,
                    "scanner": scanner,
                    "source_tile_id": tile_ids[i],
                    "target_tile_id": tile_ids[j]
                })

        print(
            f"  {method}: "
            f"shared={shared}, "
            f"Jaccard={jaccard:.6f}, "
            f"neighbor={neighbor_score:.6f}, "
            f"components={components_a}/{components_b}"
        )


# ============================================================
# STEP 4: SAVE RESULTS
# ============================================================

print("\n[4] Saving experimental results...")

results_df = pd.DataFrame(results)
edge_df = pd.DataFrame(edge_records)

results_path = (
    TABLE_DIR / "plism_controlled_graph_stability.csv"
)

edges_path = (
    TABLE_DIR / "plism_controlled_graph_edges.csv"
)

results_df.to_csv(results_path, index=False)
edge_df.to_csv(edges_path, index=False)


# ============================================================
# STEP 5: FINAL REPORT
# ============================================================

print("\n[5] Generating experiment report...")

report = {
    "phase": "3A",
    "step": "3.9",
    "dataset": "PLISM",
    "scanner_pair": ["AT2", "GT450"],
    "stainer": "GIVH",
    "feature_encoder": "facebook/dinov2-small",
    "matched_nodes": n,
    "edge_budgets": EDGE_BUDGETS,
    "global_candidate_edges": len(global_candidates),
    "spatial_candidate_edges": len(spatial_candidates),
    "graph_construction": (
        "Top-M undirected edges ranked by cosine similarity"
    ),
    "edge_budget_matched": True,
    "spatial_coordinates_verified_physically": False,
    "independent_specimens_verified": False,
    "results": results
}

report_path = (
    REPORT_DIR / "09_controlled_graph_stability.json"
)

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("CONTROLLED GRAPH EXPERIMENT COMPLETED")
print("=" * 70)

print("\nFinal comparison:")

print(
    results_df[
        [
            "method",
            "edge_budget",
            "shared_edges",
            "edge_jaccard",
            "neighbor_jaccard"
        ]
    ].to_string(index=False))

print("\nSaved outputs:")
print(results_path)
print(edges_path)
print(report_path)

print("\nSTEP 3.9 COMPLETED")
