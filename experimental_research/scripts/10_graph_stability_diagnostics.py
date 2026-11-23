
"""
Phase 3A - Step 3.10
Graph Connectivity and Random-Baseline Diagnostics

Uses graphs produced by Step 3.9.

Checks:
- Graph edge count and connectivity
- Isolated nodes
- Largest connected component
- Empirical random edge-Jaccard baseline
- Observed improvement above random baseline

No remote requests, model loading, or image downloads.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path.cwd()

PHASE1 = ROOT / "dataset_exploration" / "outputs"
PHASE3 = ROOT / "experimental_research" / "outputs"

FEATURE_FILE = (
    PHASE1 / "features/plism_dinov2_features.npz"
)

EDGE_FILE = (
    PHASE3 / "tables/plism_controlled_graph_edges.csv"
)

RESULT_FILE = (
    PHASE3 / "tables/plism_controlled_graph_stability.csv"
)

TABLE_DIR = PHASE3 / "tables"
REPORT_DIR = PHASE3 / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SCANNERS = ["AT2", "GT450"]

RANDOM_TRIALS = 1000
SEED = 42

rng = np.random.default_rng(SEED)


# ============================================================
# GRAPH HELPERS
# ============================================================

def connectivity_statistics(n, edges):

    adjacency = [set() for _ in range(n)]

    for i, j in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)

    degrees = np.array([
        len(neighbors)
        for neighbors in adjacency
    ])

    visited = set()
    sizes = []

    for start in range(n):

        if start in visited:
            continue

        stack = [start]
        visited.add(start)
        size = 0

        while stack:

            node = stack.pop()
            size += 1

            for neighbor in adjacency[node]:

                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        sizes.append(size)

    return {
        "components": len(sizes),
        "isolated_nodes": int(np.sum(degrees == 0)),
        "node_coverage": float(np.mean(degrees > 0)),
        "largest_component_nodes": max(sizes),
        "largest_component_fraction": max(sizes) / n,
        "minimum_degree": int(degrees.min()),
        "maximum_degree": int(degrees.max()),
        "mean_degree": float(degrees.mean()),
    }


def edge_jaccard(a, b):
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def random_reference(candidate_edges, budget):

    candidate_count = len(candidate_edges)

    if budget > candidate_count:
        raise ValueError("Budget exceeds candidate pool.")

    scores = np.empty(RANDOM_TRIALS, dtype=float)

    for trial in range(RANDOM_TRIALS):

        a = set(
            rng.choice(
                candidate_count,
                size=budget,
                replace=False
            ).tolist()
        )

        b = set(
            rng.choice(
                candidate_count,
                size=budget,
                replace=False
            ).tolist()
        )

        scores[trial] = edge_jaccard(a, b)

    return {
        "random_mean": float(scores.mean()),
        "random_std": float(scores.std(ddof=1)),
        "random_p025": float(np.percentile(scores, 2.5)),
        "random_p975": float(np.percentile(scores, 97.5)),
        "random_max": float(scores.max()),
    }


# ============================================================
# STEP 1: LOAD VERIFIED NODE INVENTORY
# ============================================================

print("=" * 70)
print("PHASE 3A - GRAPH STABILITY DIAGNOSTICS")
print("=" * 70)

print("\n[1] Loading node and graph inventories...")

for path in [FEATURE_FILE, EDGE_FILE, RESULT_FILE]:
    if not path.is_file():
        raise SystemExit(f"Missing file: {path}")

with np.load(FEATURE_FILE) as data:
    tile_ids = data["tile_ids"].astype(str)
    rows = data["grid_rows"].astype(int)
    cols = data["grid_cols"].astype(int)

n = len(tile_ids)

if len(set(tile_ids)) != n:
    raise ValueError("Duplicate node identifiers.")

node_index = {
    tile_id: i
    for i, tile_id in enumerate(tile_ids)
}

coords = np.column_stack([rows, cols])

edge_df = pd.read_csv(EDGE_FILE, dtype={
    "source_tile_id": str,
    "target_tile_id": str,
    "method": str,
    "scanner": str,
})

results_df = pd.read_csv(RESULT_FILE)

print(f"Nodes: {n}")
print(f"Saved graph edges: {len(edge_df)}")


# ============================================================
# STEP 2: DEFINE CANDIDATE POOLS
# ============================================================

print("\n[2] Constructing candidate pools...")

global_pool = []
spatial_pool = []

for i in range(n):
    for j in range(i + 1, n):

        edge = (i, j)
        global_pool.append(edge)

        distance = np.linalg.norm(
            coords[i] - coords[j]
        )

        if 0 < distance <= np.sqrt(2) + 1e-8:
            spatial_pool.append(edge)

candidate_pools = {
    "Global-feature": set(global_pool),
    "Spatial-feature": set(spatial_pool)
}

for method, pool in candidate_pools.items():
    print(f"{method}: {len(pool)} candidate edges")


# ============================================================
# STEP 3: RECONSTRUCT SAVED GRAPHS
# ============================================================

print("\n[3] Reconstructing saved graphs...")

graphs = {}

for (method, budget, scanner), group in edge_df.groupby(
    ["method", "edge_budget", "scanner"]
):

    edges = set()

    for row in group.itertuples(index=False):

        if (
            row.source_tile_id not in node_index
            or row.target_tile_id not in node_index
        ):
            raise ValueError("Unknown node in saved edges.")

        i = node_index[row.source_tile_id]
        j = node_index[row.target_tile_id]

        if i == j:
            raise ValueError("Self-loop detected.")

        edges.add(tuple(sorted((i, j))))

    budget = int(budget)

    if len(edges) != budget:
        raise ValueError(
            f"Unexpected edge count for "
            f"{method}, {budget}, {scanner}"
        )

    if not edges.issubset(candidate_pools[method]):
        raise ValueError(
            f"Graph contains invalid candidate edge: {method}"
        )

    graphs[(method, budget, scanner)] = edges

print(f"Graphs reconstructed: {len(graphs)}")


# ============================================================
# STEP 4: RANDOM BASELINE AND CONNECTIVITY
# ============================================================

print("\n[4] Running random and connectivity diagnostics...")

records = []

for row in results_df.itertuples(index=False):

    method = row.method
    budget = int(row.edge_budget)

    edges_a = graphs[(method, budget, "AT2")]
    edges_b = graphs[(method, budget, "GT450")]

    observed = edge_jaccard(edges_a, edges_b)

    # Cross-check the previous experiment.
    if not np.isclose(
        observed,
        float(row.edge_jaccard),
        atol=1e-6
    ):
        raise ValueError(
            f"Step 3.9 Jaccard mismatch: {method}, {budget}"
        )

    stats_a = connectivity_statistics(n, edges_a)
    stats_b = connectivity_statistics(n, edges_b)

    pool = candidate_pools[method]

    null = random_reference(
        list(pool),
        budget
    )

    excess = observed - null["random_mean"]

    # Descriptive normalization only, not a
    # standard chance-corrected statistical test.
    normalized_excess = (
        excess / (1.0 - null["random_mean"])
        if null["random_mean"] < 1
        else 0.0
    )

    result = {
        "method": method,
        "edge_budget": budget,
        "candidate_pool_size": len(pool),
        "candidate_pool_fraction": budget / len(pool),
        "observed_jaccard": observed,
        **null,
        "excess_over_random": excess,
        "normalized_excess_descriptive": normalized_excess,
        "at2_components": stats_a["components"],
        "gt450_components": stats_b["components"],
        "at2_isolated_nodes": stats_a["isolated_nodes"],
        "gt450_isolated_nodes": stats_b["isolated_nodes"],
        "at2_node_coverage": stats_a["node_coverage"],
        "gt450_node_coverage": stats_b["node_coverage"],
        "at2_largest_component_fraction":
            stats_a["largest_component_fraction"],
        "gt450_largest_component_fraction":
            stats_b["largest_component_fraction"],
    }

    records.append(result)

    print(f"\n{method} | budget={budget}")
    print(f"  Observed Jaccard: {observed:.6f}")
    print(f"  Random mean: {null['random_mean']:.6f}")
    print(f"  Excess above random: {excess:.6f}")
    print(
        f"  Connected components: "
        f"{stats_a['components']}/{stats_b['components']}"
    )
    print(
        f"  Isolated nodes: "
        f"{stats_a['isolated_nodes']}/{stats_b['isolated_nodes']}"
    )


# ============================================================
# STEP 5: SAVE RESULTS
# ============================================================

print("\n[5] Saving diagnostic results...")

diagnostics = pd.DataFrame(records)

csv_path = (
    TABLE_DIR / "plism_graph_stability_diagnostics.csv"
)

diagnostics.to_csv(csv_path, index=False)

report = {
    "phase": "3A",
    "step": "3.10",
    "dataset": "PLISM",
    "nodes": n,
    "random_trials_per_configuration": RANDOM_TRIALS,
    "random_seed": SEED,
    "random_baseline": (
        "Independent uniformly sampled edge sets "
        "from each method's eligible candidate pool"
    ),
    "interpretation_limitations": [
        "One tissue region; no independent specimen replication",
        "Different candidate-pool sizes across methods",
        "Connectivity not enforced",
        "Random references are not inferential p-values",
        "Normalized excess is descriptive, not a validated effect size",
    ],
    "results": records,
}

report_path = (
    REPORT_DIR / "10_graph_stability_diagnostics.json"
)

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("GRAPH STABILITY DIAGNOSTICS COMPLETED")
print("=" * 70)

print(
    diagnostics[
        [
            "method",
            "edge_budget",
            "observed_jaccard",
            "random_mean",
            "excess_over_random",
            "at2_isolated_nodes",
            "gt450_isolated_nodes",
        ]
    ].to_string(index=False)
)

print("\nSaved outputs:")
print(csv_path)
print(report_path)

print("\nSTEP 3.10 COMPLETED")
