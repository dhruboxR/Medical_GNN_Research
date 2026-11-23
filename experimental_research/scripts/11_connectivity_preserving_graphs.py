
"""
Phase 3A - Step 3.11
Connectivity-Preserving Graph Stability

Methods:
1. Shared spatial spanning tree + global feature edges
2. Shared spatial spanning tree + spatial feature edges

Controls:
- Same 90 nodes
- Same total edge budgets
- Same fixed spanning-tree backbone
- Separate full-graph and residual-edge Jaccard
- Random residual-edge reference for each candidate pool

No image downloads, network requests or GNN training.
"""

from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path.cwd()

FEATURE_FILE = (
    ROOT / "dataset_exploration/outputs/features/"
    "plism_dinov2_features.npz"
)

OUTPUT_ROOT = ROOT / "experimental_research/outputs"

TABLE_DIR = OUTPUT_ROOT / "tables"
REPORT_DIR = OUTPUT_ROOT / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EDGE_BUDGETS = [135, 180, 225]

RANDOM_TRIALS = 1000
SEED = 42
SPATIAL_RADIUS = np.sqrt(2) + 1e-8

rng = np.random.default_rng(SEED)


# ============================================================
# 2. UNION-FIND FOR SPANNING TREE
# ============================================================

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a = self.find(a)
        b = self.find(b)

        if a == b:
            return False

        if self.rank[a] < self.rank[b]:
            a, b = b, a

        self.parent[b] = a

        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1

        self.count -= 1
        return True


def build_spatial_backbone(coords, candidates):
    """
    Deterministic minimum spanning tree using
    grid distance followed by node indices.

    The backbone is independent of image features.
    """
    ranked = sorted(
        candidates,
        key=lambda edge: (
            float(np.linalg.norm(
                coords[edge[0]] - coords[edge[1]]
            )),
            edge[0],
            edge[1],
        )
    )

    uf = UnionFind(len(coords))
    tree = set()

    for i, j in ranked:
        if uf.union(i, j):
            tree.add((i, j))

        if len(tree) == len(coords) - 1:
            break

    if len(tree) != len(coords) - 1:
        raise RuntimeError(
            "Spatial candidate graph is disconnected. "
            "A spatial spanning tree cannot be built."
        )

    return tree


# ============================================================
# 3. GRAPH HELPERS
# ============================================================

def build_feature_edges(
    features,
    candidates,
    backbone,
    count
):
    similarity = cosine_similarity(features)

    available = [
        edge for edge in candidates
        if edge not in backbone
    ]

    if count > len(available):
        raise ValueError(
            "Insufficient candidate edges for budget."
        )

    ranked = sorted(
        available,
        key=lambda edge: (
            -float(similarity[edge[0], edge[1]]),
            edge[0],
            edge[1]
        )
    )

    return set(ranked[:count])


def jaccard(a, b):
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def connectivity(n, edges):
    uf = UnionFind(n)
    degree = np.zeros(n, dtype=int)

    for i, j in edges:
        uf.union(i, j)
        degree[i] += 1
        degree[j] += 1

    return {
        "components": uf.count,
        "isolated_nodes": int(np.sum(degree == 0)),
        "mean_degree": float(np.mean(degree))
    }


def random_residual_reference(pool_size, count):
    """
    Empirical expected residual Jaccard from two
    independent uniform draws from the same pool.
    """
    scores = []

    for _ in range(RANDOM_TRIALS):
        a = set(
            rng.choice(
                pool_size,
                size=count,
                replace=False
            ).tolist()
        )

        b = set(
            rng.choice(
                pool_size,
                size=count,
                replace=False
            ).tolist()
        )

        scores.append(jaccard(a, b))

    return float(np.mean(scores))


# ============================================================
# 4. LOAD FEATURE DATA
# ============================================================

print("=" * 70, flush=True)
print("PHASE 3A - CONNECTIVITY-PRESERVING GRAPHS", flush=True)
print("=" * 70, flush=True)

print("\n[1] Loading embeddings...", flush=True)

if not FEATURE_FILE.is_file():
    raise SystemExit(f"Missing: {FEATURE_FILE}")

with np.load(FEATURE_FILE) as data:
    at2 = data["at2"].astype(np.float64)
    gt450 = data["gt450"].astype(np.float64)
    tile_ids = data["tile_ids"].astype(str)
    rows = data["grid_rows"].astype(int)
    cols = data["grid_cols"].astype(int)

if at2.shape != gt450.shape:
    raise ValueError("Feature shapes differ.")

n = len(tile_ids)

if len(set(tile_ids)) != n:
    raise ValueError("Duplicate tile IDs.")

if not np.isfinite(at2).all() or not np.isfinite(gt450).all():
    raise ValueError("Non-finite embeddings.")

coords = np.column_stack([rows, cols])

if len(np.unique(coords, axis=0)) != n:
    raise ValueError("Duplicate grid coordinates.")

print(f"Nodes: {n}")
print(f"Feature dimension: {at2.shape[1]}")


# ============================================================
# 5. CONSTRUCT CANDIDATE POOLS
# ============================================================

print("\n[2] Building candidate pools...", flush=True)

global_pool = []
spatial_pool = []

for i in range(n):
    for j in range(i + 1, n):
        edge = (i, j)
        global_pool.append(edge)

        distance = np.linalg.norm(
            coords[i] - coords[j]
        )

        if 0 < distance <= SPATIAL_RADIUS:
            spatial_pool.append(edge)

print(f"Global candidates: {len(global_pool)}")
print(f"Spatial candidates: {len(spatial_pool)}")


# ============================================================
# 6. CONSTRUCT COMMON SPATIAL BACKBONE
# ============================================================

print("\n[3] Constructing fixed spatial backbone...", flush=True)

backbone = build_spatial_backbone(
    coords,
    spatial_pool
)

backbone_stats = connectivity(n, backbone)

print(f"Backbone edges: {len(backbone)}")
print(f"Components: {backbone_stats['components']}")
print(f"Isolated nodes: {backbone_stats['isolated_nodes']}")

assert len(backbone) == n - 1
assert backbone_stats["components"] == 1


# ============================================================
# 7. MATCHED-BUDGET EXPERIMENT
# ============================================================

print("\n[4] Running graph comparisons...", flush=True)

methods = {
    "Global-feature+MST": global_pool,
    "Spatial-feature+MST": spatial_pool
}

results = []
edge_records = []

for budget in EDGE_BUDGETS:

    residual_budget = budget - len(backbone)

    if residual_budget <= 0:
        raise ValueError(
            "Budget must exceed backbone edge count."
        )

    print(f"\nTotal edge budget: {budget}")
    print(f"Feature-selected edges: {residual_budget}")

    for method, candidates in methods.items():

        residual_a = build_feature_edges(
            at2,
            candidates,
            backbone,
            residual_budget
        )

        residual_b = build_feature_edges(
            gt450,
            candidates,
            backbone,
            residual_budget
        )

        graph_a = backbone | residual_a
        graph_b = backbone | residual_b

        assert len(graph_a) == budget
        assert len(graph_b) == budget

        stats_a = connectivity(n, graph_a)
        stats_b = connectivity(n, graph_b)

        assert stats_a["components"] == 1
        assert stats_b["components"] == 1
        assert stats_a["isolated_nodes"] == 0
        assert stats_b["isolated_nodes"] == 0

        full_score = jaccard(graph_a, graph_b)
        residual_score = jaccard(
            residual_a,
            residual_b
        )

        remaining_pool_size = (
            len(candidates) - len(backbone)
        )

        random_mean = random_residual_reference(
            remaining_pool_size,
            residual_budget
        )

        result = {
            "method": method,
            "total_edge_budget": budget,
            "nodes": n,
            "backbone_edges": len(backbone),
            "feature_selected_edges": residual_budget,
            "remaining_candidate_pool":
                remaining_pool_size,
            "at2_edges": len(graph_a),
            "gt450_edges": len(graph_b),
            "shared_total_edges": len(graph_a & graph_b),
            "shared_residual_edges":
                len(residual_a & residual_b),
            "full_edge_jaccard": full_score,
            "residual_edge_jaccard": residual_score,
            "random_residual_jaccard_mean": random_mean,
            "residual_excess_over_random":
                residual_score - random_mean,
            "at2_components": stats_a["components"],
            "gt450_components": stats_b["components"],
            "at2_isolated_nodes": stats_a["isolated_nodes"],
            "gt450_isolated_nodes": stats_b["isolated_nodes"]
        }

        results.append(result)

        for scanner, graph, residual in [
            ("AT2", graph_a, residual_a),
            ("GT450", graph_b, residual_b)
        ]:
            for i, j in sorted(graph):
                edge_records.append({
                    "method": method,
                    "total_edge_budget": budget,
                    "scanner": scanner,
                    "source_tile_id": tile_ids[i],
                    "target_tile_id": tile_ids[j],
                    "edge_role": (
                        "backbone"
                        if (i, j) in backbone
                        else "feature_selected"
                    )
                })

        print(f"\n  {method}")
        print(f"    Full Jaccard: {full_score:.6f}")
        print(f"    Residual Jaccard: {residual_score:.6f}")
        print(f"    Random residual mean: {random_mean:.6f}")
        print(
            "    Residual excess above random: "
            f"{residual_score - random_mean:.6f}"
        )
        print(
            "    Connected components: "
            f"{stats_a['components']}/{stats_b['components']}"
        )


# ============================================================
# 8. SAVE RESULTS
# ============================================================

print("\n[5] Saving experiment results...", flush=True)

results_df = pd.DataFrame(results)
edges_df = pd.DataFrame(edge_records)

results_path = (
    TABLE_DIR / "plism_connected_graph_stability.csv"
)

edges_path = (
    TABLE_DIR / "plism_connected_graph_edges.csv"
)

results_df.to_csv(results_path, index=False)
edges_df.to_csv(edges_path, index=False)

report = {
    "phase": "3A",
    "step": "3.11",
    "generated_utc":
        datetime.now(timezone.utc).isoformat(),
    "dataset": "PLISM",
    "encoder": "facebook/dinov2-small",
    "nodes": n,
    "edge_budgets": EDGE_BUDGETS,
    "backbone_type":
        "Deterministic spatial minimum spanning tree",
    "backbone_edges": len(backbone),
    "backbone_shared_across_scanners": True,
    "random_trials": RANDOM_TRIALS,
    "random_seed": SEED,
    "results": results,
    "limitations": [
        "Only one pilot tissue region",
        "Shared backbone increases full-graph overlap",
        "Different residual candidate pools",
        "Residual overlap is descriptive, not inferential",
        "Independent biological specimens not verified",
        "Physical coordinate semantics not fully verified"
    ]
}

report_path = (
    REPORT_DIR / "11_connected_graph_stability.json"
)

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)


# ============================================================
# 9. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("CONNECTIVITY-PRESERVING EXPERIMENT COMPLETED")
print("=" * 70)

print(
    results_df[
        [
            "method",
            "total_edge_budget",
            "full_edge_jaccard",
            "residual_edge_jaccard",
            "random_residual_jaccard_mean",
            "residual_excess_over_random",
            "at2_components",
            "gt450_components"
        ]
    ].to_string(index=False)
)

print("\nSaved files:")
print(results_path)
print(edges_path)
print(report_path)

print("\nSTEP 3.11 COMPLETED")
