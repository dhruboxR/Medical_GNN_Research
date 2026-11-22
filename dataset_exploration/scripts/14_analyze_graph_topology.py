
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ==================================================
# CONFIGURATION
# ==================================================

FEATURE_FILE = Path(
    "dataset_exploration/outputs/features/"
    "plism_dinov2_features.npz"
)

OUTPUT_ROOT = Path("dataset_exploration/outputs")
TABLE_DIR = OUTPUT_ROOT / "tables"
REPORT_DIR = OUTPUT_ROOT / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_K = 5
HYBRID_K = 4
SPATIAL_RADIUS = np.sqrt(2) + 1e-8

print("=" * 70)
print("PLISM CROSS-SCANNER GRAPH TOPOLOGY ANALYSIS")
print("=" * 70)

# ==================================================
# STEP 1: LOAD MATCHED FEATURES
# ==================================================

print("\n[1] Loading feature embeddings...")

if not FEATURE_FILE.exists():
    raise SystemExit(f"Missing: {FEATURE_FILE}")

with np.load(FEATURE_FILE) as data:
    at2 = data["at2"].astype(np.float64)
    gt450 = data["gt450"].astype(np.float64)
    tile_ids = data["tile_ids"].astype(str)
    rows = data["grid_rows"].astype(int)
    cols = data["grid_cols"].astype(int)

assert at2.shape == gt450.shape
assert len(tile_ids) == len(set(tile_ids))
assert len(rows) == len(tile_ids)
assert len(cols) == len(tile_ids)

coordinates = np.column_stack([rows, cols])
assert len(np.unique(coordinates, axis=0)) == len(tile_ids)

N = len(tile_ids)

print(f"Matched nodes: {N}")
print(f"Feature dimensions: {at2.shape[1]}")

# ==================================================
# STEP 2: GRAPH FUNCTIONS
# ==================================================

def spatial_graph(coords):
    """Undirected 8-neighbor grid graph."""
    edges = set()

    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            distance = np.linalg.norm(
                coords[i] - coords[j]
            )

            if 0 < distance <= SPATIAL_RADIUS:
                edges.add((i, j))

    return edges


def knn_graph(features, k):
    """Undirected cosine kNN graph."""
    similarity = cosine_similarity(features)
    np.fill_diagonal(similarity, -np.inf)

    edges = set()

    for i in range(len(features)):
        neighbors = np.argsort(
            -similarity[i],
            kind="stable"
        )[:k]

        for j in neighbors:
            edges.add(tuple(sorted((i, int(j)))))

    return edges


def hybrid_graph(features, spatial_edges, k):
    """Select top-k feature neighbors within spatial edges."""
    similarity = cosine_similarity(features)

    candidate_neighbors = {
        i: set() for i in range(len(features))
    }

    for i, j in spatial_edges:
        candidate_neighbors[i].add(j)
        candidate_neighbors[j].add(i)

    edges = set()

    for i, neighbors in candidate_neighbors.items():
        ranked = sorted(
            neighbors,
            key=lambda j: (-similarity[i, j], j)
        )

        for j in ranked[:k]:
            edges.add(tuple(sorted((i, j))))

    return edges


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


def graph_stats(n, edges):
    return {
        "edges": len(edges),
        "average_degree": 2 * len(edges) / n,
        "components": connected_components(n, edges)
    }


def edge_jaccard(a, b):
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def neighbor_overlap(n, edges_a, edges_b):
    """Mean per-node neighbor Jaccard similarity."""
    neighbors_a = [set() for _ in range(n)]
    neighbors_b = [set() for _ in range(n)]

    for i, j in edges_a:
        neighbors_a[i].add(j)
        neighbors_a[j].add(i)

    for i, j in edges_b:
        neighbors_b[i].add(j)
        neighbors_b[j].add(i)

    values = []

    for a, b in zip(neighbors_a, neighbors_b):
        union = a | b
        score = len(a & b) / len(union) if union else 1.0
        values.append(score)

    return float(np.mean(values))


# ==================================================
# STEP 3: CONSTRUCT GRAPHS
# ==================================================

print("\n[2] Constructing graph variants...")

spatial_edges = spatial_graph(coordinates)

graphs = {
    "Spatial-only": {
        "AT2": spatial_edges,
        "GT450": set(spatial_edges)
    },
    "Feature-kNN": {
        "AT2": knn_graph(at2, FEATURE_K),
        "GT450": knn_graph(gt450, FEATURE_K)
    },
    "Hybrid": {
        "AT2": hybrid_graph(
            at2, spatial_edges, HYBRID_K
        ),
        "GT450": hybrid_graph(
            gt450, spatial_edges, HYBRID_K
        )
    }
}

# ==================================================
# STEP 4: COMPARE GRAPH STABILITY
# ==================================================

print("\n[3] Comparing AT2 and GT450 graphs...")

results = []

for name, scanner_graphs in graphs.items():
    a = scanner_graphs["AT2"]
    b = scanner_graphs["GT450"]

    stats_a = graph_stats(N, a)
    stats_b = graph_stats(N, b)

    results.append({
        "graph_type": name,
        "nodes": N,
        "at2_edges": stats_a["edges"],
        "gt450_edges": stats_b["edges"],
        "shared_edges": len(a & b),
        "at2_unique_edges": len(a - b),
        "gt450_unique_edges": len(b - a),
        "edge_jaccard": edge_jaccard(a, b),
        "mean_neighbor_jaccard": neighbor_overlap(
            N, a, b
        ),
        "at2_avg_degree": stats_a["average_degree"],
        "gt450_avg_degree": stats_b["average_degree"],
        "at2_components": stats_a["components"],
        "gt450_components": stats_b["components"]
    })

    print(f"\n{name}")
    print(f"  AT2 edges: {len(a)}")
    print(f"  GT450 edges: {len(b)}")
    print(f"  Shared edges: {len(a & b)}")
    print(
        f"  Edge Jaccard: "
        f"{edge_jaccard(a, b):.6f}"
    )
    print(
        f"  Mean neighbor Jaccard: "
        f"{neighbor_overlap(N, a, b):.6f}"
    )
    print(
        f"  Components: "
        f"AT2={stats_a['components']}, "
        f"GT450={stats_b['components']}"
    )

# ==================================================
# STEP 5: SAVE DATA
# ==================================================

print("\n[4] Saving graph comparison results...")

results_df = pd.DataFrame(results)

csv_path = TABLE_DIR / "plism_graph_stability.csv"
results_df.to_csv(csv_path, index=False)

edge_records = []

for name, scanner_graphs in graphs.items():
    for scanner, edges in scanner_graphs.items():
        for i, j in sorted(edges):
            edge_records.append({
                "graph_type": name,
                "scanner": scanner,
                "source_tile": tile_ids[i],
                "target_tile": tile_ids[j],
                "source_index": i,
                "target_index": j
            })

edge_df = pd.DataFrame(edge_records)
edge_path = TABLE_DIR / "plism_graph_edges.csv"
edge_df.to_csv(edge_path, index=False)

report = {
    "dataset": "PLISM",
    "stainer": "GIVH",
    "scanners": ["AT2", "GT450"],
    "feature_model": "facebook/dinov2-small",
    "matched_nodes": N,
    "feature_k": FEATURE_K,
    "hybrid_k": HYBRID_K,
    "spatial_neighborhood": "8-neighbor grid",
    "results": results
}

report_path = REPORT_DIR / "plism_graph_stability.json"

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(f"Comparison: {csv_path}")
print(f"Graph edges: {edge_path}")
print(f"Report: {report_path}")

print("\n" + "=" * 70)
print("GRAPH TOPOLOGY ANALYSIS COMPLETED")
print("=" * 70)
