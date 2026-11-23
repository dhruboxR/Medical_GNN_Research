
"""
PHASE 3B - FINAL README FIGURES

Generate publication-style descriptive figures from
completed and frozen Phase 3B experiment tables.

Does not run any new experiments.
Does not modify research data or frozen inputs.
Does not overwrite existing figures.

Run from the MEDICAL_GNN_RESEARCH project root.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "multi_region_research"
TABLES = BASE / "outputs/tables"
FIGURES = BASE / "outputs/figures"

FIGURES.mkdir(parents=True, exist_ok=True)

FILES = {
    "features": TABLES / "plism_phase3b_dinov2_region_summary.csv",
    "connected": TABLES / "plism_phase3b_connected_graph_summary.csv",
    "geometry": TABLES / "plism_phase3b_geometry_conditioned_summary.csv",
    "propagation": TABLES / "plism_phase3b_propagation_summary.csv",
    "smoothing": TABLES / "plism_phase3b_oversmoothing_summary.csv",
    "equal": TABLES / "plism_phase3b_equal_strength_summary.csv",
}

for name, path in FILES.items():
    if not path.is_file():
        raise RuntimeError(
            f"Missing {name} input table: {path}"
        )

OUTPUTS = [
    FIGURES / "phase3b_feature_similarity_by_region.png",
    FIGURES / "phase3b_connected_graph_stability.png",
    FIGURES / "phase3b_geometry_conditioned_stability.png",
    FIGURES / "phase3b_stability_smoothing_tradeoff.png",
    FIGURES / "phase3b_equal_strength_control.png",
]

for path in OUTPUTS:
    if path.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing figure: {path}"
        )

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 120,
    "savefig.dpi": 220,
})

GLOBAL_COLOR = "#3165A0"
SPATIAL_COLOR = "#D07A35"

def save(fig, name):
    path = FIGURES / name
    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print("Saved:", path, flush=True)


print("=" * 65)
print("PHASE 3B - FINAL README FIGURE GENERATION")
print("=" * 65)


# ============================================================
# FIGURE 1: FEATURE SIMILARITY BY REGION
# ============================================================

features = pd.read_csv(FILES["features"])
features = features.sort_values("region_id")

if len(features) != 10:
    raise RuntimeError("Expected ten region summaries.")

labels = [
    "R" + str(i).zfill(2)
    for i in range(1, 11)
]

values = features[
    "mean_cosine_similarity"
].to_numpy(dtype=float)

fig, ax = plt.subplots(figsize=(10, 4.7))
ax.bar(labels, values, color=GLOBAL_COLOR, width=0.68)
ax.axhline(
    values.mean(),
    color="black",
    linestyle="--",
    linewidth=1,
    label=f"Mean of region means: {values.mean():.4f}",
)

ax.set_ylim(0.90, 1.00)
ax.set_ylabel("Paired feature cosine similarity")
ax.set_xlabel("Frozen spatial region")
ax.set_title("Cross-scanner DINOv2 feature similarity")
ax.grid(axis="y", alpha=0.2)
ax.legend(frameon=False)
save(fig, "phase3b_feature_similarity_by_region.png")


# ============================================================
# FIGURE 2: CONNECTED GRAPH RESIDUAL JACCARD
# ============================================================

connected = pd.read_csv(FILES["connected"])
budgets = [54, 72, 90]

fig, ax = plt.subplots(figsize=(8.4, 5))

for method, color, label in (
    ("global", GLOBAL_COLOR, "Global graph"),
    ("spatial", SPATIAL_COLOR, "Spatial graph"),
):
    subset = (
        connected[connected["graph_type"] == method]
        .set_index("edge_budget")
        .loc[budgets]
    )

    ax.plot(
        budgets,
        subset["mean_residual_jaccard"],
        marker="o",
        markersize=8,
        linewidth=2.5,
        color=color,
        label=label,
    )

ax.set_ylim(0, 1)
ax.set_xticks(budgets)
ax.set_xlabel("Total edges per connected graph")
ax.set_ylabel("Mean residual-edge Jaccard")
ax.set_title("Cross-scanner connected-graph stability")
ax.grid(alpha=0.22)
ax.legend(frameon=False)
save(fig, "phase3b_connected_graph_stability.png")


# ============================================================
# FIGURE 3: GEOMETRY-CONDITIONED AGREEMENT
# ============================================================

geometry = pd.read_csv(FILES["geometry"])

fig, ax = plt.subplots(figsize=(8.4, 5))
x = np.arange(len(budgets))
width = 0.36

for method, color, offset, label in (
    ("global", GLOBAL_COLOR, -width / 2, "Global graph"),
    ("spatial", SPATIAL_COLOR, width / 2, "Spatial graph"),
):
    subset = (
        geometry[geometry["graph_type"] == method]
        .set_index("edge_budget")
        .loc[budgets]
    )

    ax.bar(
        x + offset,
        subset["mean_conditional_adjusted"],
        width,
        color=color,
        label=label,
    )

ax.set_xticks(x, budgets)
ax.set_ylim(0, 0.8)
ax.set_xlabel("Total edges per connected graph")
ax.set_ylabel("Mean geometry-conditioned adjusted agreement")
ax.set_title("Graph stability after distance conditioning")
ax.grid(axis="y", alpha=0.22)
ax.legend(frameon=False)
save(fig, "phase3b_geometry_conditioned_stability.png")


# ============================================================
# FIGURE 4: STABILITY VS SMOOTHING AT 72 EDGES
# ============================================================

propagation = pd.read_csv(FILES["propagation"])
smoothing = pd.read_csv(FILES["smoothing"])

depths = [0, 1, 2, 4, 8]
budget = 72

fig, axes = plt.subplots(
    1, 2, figsize=(12.2, 4.8)
)

for method, color, label in (
    ("global", GLOBAL_COLOR, "Global graph"),
    ("spatial", SPATIAL_COLOR, "Spatial graph"),
):
    p = (
        propagation[
            (propagation["graph_type"] == method)
            & (propagation["edge_budget"] == budget)
        ]
        .set_index("depth")
        .loc[depths]
    )

    s = (
        smoothing[
            (smoothing["graph_type"] == method)
            & (smoothing["edge_budget"] == budget)
        ]
        .set_index("depth")
        .loc[depths]
    )

    axes[0].plot(
        depths,
        p["mean_topology_l2"],
        marker="o",
        linewidth=2.2,
        color=color,
        label=label,
    )

    axes[1].plot(
        depths,
        s["mean_variance_retention"],
        marker="o",
        linewidth=2.2,
        color=color,
        label=label,
    )

axes[0].set_title("Topology-induced representation change")
axes[0].set_ylabel("Mean node L2 difference")
axes[0].set_xlabel("Propagation depth")
axes[0].set_xticks(depths)
axes[0].grid(alpha=0.2)

axes[1].set_title("Node-variance retention")
axes[1].set_ylabel("Variance at depth / initial variance")
axes[1].set_xlabel("Propagation depth")
axes[1].set_xticks(depths)
axes[1].grid(alpha=0.2)

fig.suptitle(
    "Stability and smoothing trade-off at 72 edges",
    fontsize=14,
)
fig.legend(
    loc="lower center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
)
fig.tight_layout(rect=[0, 0.055, 1, 0.94])

save(fig, "phase3b_stability_smoothing_tradeoff.png")


# ============================================================
# FIGURE 5: EQUAL PROPAGATION STRENGTH
# ============================================================

equal = pd.read_csv(FILES["equal"])

fig, ax = plt.subplots(figsize=(9, 5.4))

for budget, marker in (
    (54, "o"),
    (72, "s"),
    (90, "^"),
):
    subset = (
        equal[equal["edge_budget"] == budget]
        .sort_values("alpha")
    )

    ax.plot(
        subset["alpha"],
        subset["mean_topology_advantage"],
        marker=marker,
        markersize=7,
        linewidth=2.3,
        label=f"{budget} edges",
    )

ax.axhline(0, color="black", linewidth=1)
ax.set_xticks([0.25, 0.50, 0.75, 1.00])
ax.set_xlabel("Shared propagation strength (alpha)")
ax.set_ylabel("Global minus spatial topology L2")
ax.set_title("Topology sensitivity advantage at equal strength")
ax.grid(alpha=0.22)
ax.legend(frameon=False)

save(fig, "phase3b_equal_strength_control.png")


print("\n" + "=" * 65)
print("FINAL FIGURE GENERATION COMPLETED")
print("=" * 65)
print("Generated research figures:", len(OUTPUTS))
print("Original experiment tables: UNCHANGED")
print("Frozen manifest: UNCHANGED")
