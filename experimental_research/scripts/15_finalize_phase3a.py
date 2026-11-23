
"""
MEDICAL GNN RESEARCH
Phase 2 + Phase 3A Finalization

Generates:
- 8 final research figures (PNG + PDF)
- Phase 2 README.md
- Phase 3A README.md
- Final figure inventory

Uses existing experimental CSV files.
Does not modify original experimental results.
Does not download any datasets or models.
"""

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"

PHASE2 = ROOT / "research_design"
PHASE3 = ROOT / "experimental_research"

TABLES = PHASE3 / "outputs" / "tables"
FIGURES = PHASE3 / "outputs" / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)

COLORS = {
    "global": "#2463A6",
    "spatial": "#D77B2B",
    "backbone": "#666666",
    "random": "#9A9A9A",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.20,
})

created = []


def load_csv(filename, required):
    path = TABLES / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing required results: {path}"
        )

    df = pd.read_csv(path)

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"{filename} missing columns: "
            f"{sorted(missing)}"
        )

    return df


def save_figure(fig, filename, description):
    png = FIGURES / f"{filename}.png"
    pdf = FIGURES / f"{filename}.pdf"

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    created.append({
        "figure": filename,
        "description": description,
        "png": png.name,
        "pdf": pdf.name,
    })

    print(f"  Saved: {filename} (PNG + PDF)")


def line_plot(
    ax, df, x_col, y_col,
    method_col="method",
    method_filter=None
):
    if method_filter is not None:
        df = df[
            df[method_col].isin(method_filter)
        ]

    for method, group in df.groupby(method_col):
        group = group.sort_values(x_col)

        color = (
            COLORS["spatial"]
            if "Spatial" in method
            else COLORS["global"]
        )

        ax.plot(
            group[x_col],
            group[y_col],
            marker="o",
            linewidth=2,
            markersize=6,
            label=method,
            color=color,
        )


def style_axes(ax, xlabel, ylabel, title):
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)


# ==========================================================
# 2. LOAD VERIFIED EXPERIMENTAL TABLES
# ==========================================================

print("=" * 70)
print("PHASE 2 + PHASE 3A FINALIZATION")
print("=" * 70)

print("\n[1] Loading experimental results...")

stability = load_csv(
    "plism_controlled_graph_stability.csv",
    ["method", "edge_budget", "edge_jaccard"],
)

diagnostics = load_csv(
    "plism_graph_stability_diagnostics.csv",
    [
        "method", "edge_budget",
        "observed_jaccard", "random_mean",
        "excess_over_random",
        "at2_isolated_nodes",
        "gt450_isolated_nodes",
    ],
)

connected = load_csv(
    "plism_connected_graph_stability.csv",
    [
        "method", "total_edge_budget",
        "residual_edge_jaccard",
        "random_residual_jaccard_mean",
    ],
)

nodes = load_csv(
    "plism_connected_graph_node_summary.csv",
    [
        "method", "edge_budget",
        "mean_node_jaccard_nonempty_union",
    ],
)

sensitivity = load_csv(
    "plism_topology_sensitivity.csv",
    [
        "method", "edge_budget", "depth",
        "mean_topology_only_l2",
    ],
)

representations = load_csv(
    "plism_graph_representation_consistency.csv",
    [
        "method", "edge_budget",
        "depth", "experiment",
        "mean_node_cosine",
    ],
)

smoothing = load_csv(
    "plism_oversmoothing_metrics.csv",
    [
        "method", "edge_budget",
        "scanner", "depth",
        "dispersion_retention",
    ],
)

print("All 7 required result tables loaded.")


# ==========================================================
# FIGURE 01: MATCHED EDGE BUDGET STABILITY
# ==========================================================

print("\n[2] Generating final figures...")

fig, ax = plt.subplots(figsize=(8, 5))

line_plot(
    ax, stability,
    "edge_budget", "edge_jaccard"
)

style_axes(
    ax,
    "Edges per graph",
    "Cross-scanner edge Jaccard",
    "Matched-Budget Graph Topology Stability",
)

ax.set_ylim(0, 1)
ax.set_xticks(sorted(stability["edge_budget"].unique()))

save_figure(
    fig,
    "figure_01_matched_budget_stability",
    "Global versus spatial graph stability at equal edge budgets",
)


# ==========================================================
# FIGURE 02: OBSERVED VERSUS RANDOM BASELINE
# ==========================================================

fig, ax = plt.subplots(figsize=(8, 5))

line_plot(
    ax,
    diagnostics,
    "edge_budget",
    "excess_over_random",
)

style_axes(
    ax,
    "Edges per graph",
    "Observed Jaccard minus random mean",
    "Graph Stability Above Random Reference",
)

ax.axhline(
    0,
    color=COLORS["random"],
    linewidth=1,
    linestyle="--",
)

ax.set_xticks(
    sorted(diagnostics["edge_budget"].unique())
)

save_figure(
    fig,
    "figure_02_chance_adjusted_stability",
    "Descriptive excess edge agreement over random selection",
)


# ==========================================================
# FIGURE 03: GRAPH FRAGMENTATION
# ==========================================================

fig, ax = plt.subplots(figsize=(8, 5))

for method, group in diagnostics.groupby("method"):

    group = group.sort_values("edge_budget")

    mean_isolated = (
        group["at2_isolated_nodes"].to_numpy()
        + group["gt450_isolated_nodes"].to_numpy()
    ) / 2

    color = (
        COLORS["spatial"]
        if "Spatial" in method
        else COLORS["global"]
    )

    ax.plot(
        group["edge_budget"],
        mean_isolated,
        marker="o",
        linewidth=2,
        label=method,
        color=color,
    )

style_axes(
    ax,
    "Edges per graph",
    "Mean isolated nodes across scanners",
    "Fragmentation Before Connectivity Enforcement",
)

ax.set_ylim(bottom=0)
ax.set_xticks(
    sorted(diagnostics["edge_budget"].unique())
)

save_figure(
    fig,
    "figure_03_graph_fragmentation",
    "Isolated-node counts in unconstrained top-M graphs",
)


# ==========================================================
# FIGURE 04: CONNECTIVITY-PRESERVING STABILITY
# ==========================================================

fig, ax = plt.subplots(figsize=(8, 5))

line_plot(
    ax,
    connected,
    "total_edge_budget",
    "residual_edge_jaccard",
)

style_axes(
    ax,
    "Total edges per connected graph",
    "Residual edge Jaccard",
    "Feature-Selected Edge Stability with Shared MST",
)

ax.set_ylim(0, 1)
ax.set_xticks(
    sorted(connected["total_edge_budget"].unique())
)

save_figure(
    fig,
    "figure_04_connected_residual_stability",
    "Residual edge agreement excluding the fixed 89-edge backbone",
)


# ==========================================================
# FIGURE 05: NODE-LEVEL STABILITY
# ==========================================================

fig, ax = plt.subplots(figsize=(8, 5))

line_plot(
    ax,
    nodes,
    "edge_budget",
    "mean_node_jaccard_nonempty_union",
)

style_axes(
    ax,
    "Total edges per connected graph",
    "Mean node-neighbor Jaccard",
    "Cross-Scanner Residual Neighborhood Consistency",
)

ax.set_ylim(0, 1)
ax.set_xticks(
    sorted(nodes["edge_budget"].unique())
)

save_figure(
    fig,
    "figure_05_node_neighborhood_stability",
    "Node-level stability excluding both-empty residual neighborhoods",
)


# ==========================================================
# FIGURE 06: TOPOLOGY-ONLY SENSITIVITY
# ==========================================================

depth2 = sensitivity[
    (sensitivity["depth"] == 2)
    & (sensitivity["method"] != "Backbone-only")
].copy()

fig, ax = plt.subplots(figsize=(8, 5))

line_plot(
    ax,
    depth2,
    "edge_budget",
    "mean_topology_only_l2",
)

style_axes(
    ax,
    "Total edges per graph",
    "Mean topology-only node L2",
    "Topology Sensitivity with Fixed Input Features",
)

ax.set_ylim(bottom=0)
ax.set_xticks(sorted(depth2["edge_budget"].unique()))

save_figure(
    fig,
    "figure_06_topology_sensitivity",
    "Depth-2 topology perturbation magnitude under fixed features",
)


# ==========================================================
# FIGURE 07: REPRESENTATION CONSISTENCY
# ==========================================================

natural = representations[
    (
        representations["experiment"]
        == "natural_cross_scanner"
    )
    & (representations["depth"] == 2)
    & (representations["method"] != "Backbone-only")
].copy()

fig, ax = plt.subplots(figsize=(8, 5))

line_plot(
    ax,
    natural,
    "edge_budget",
    "mean_node_cosine",
)

style_axes(
    ax,
    "Total edges per graph",
    "Mean paired-node cosine",
    "Cross-Scanner Representation Consistency at Depth 2",
)

# Use a focused range, explicitly labeled.
ax.set_ylim(0.98, 0.99)
ax.set_xticks(
    sorted(natural["edge_budget"].unique())
)

save_figure(
    fig,
    "figure_07_representation_consistency",
    "Depth-2 cross-scanner representation similarity",
)


# ==========================================================
# FIGURE 08: DISPERSION RETENTION
# ==========================================================

depth5 = smoothing[
    smoothing["depth"] == 5
].copy()

depth5["method_group"] = depth5["method"].map(
    lambda x: (
        "Backbone-only"
        if x == "Backbone-only"
        else (
            "Spatial-feature+MST"
            if "Spatial" in x
            else "Global-feature+MST"
        )
    )
)

summary = (
    depth5.groupby(
        ["method_group", "edge_budget"],
        as_index=False
    )["dispersion_retention"]
    .mean()
)

fig, ax = plt.subplots(figsize=(8, 5))

for method, group in summary.groupby("method_group"):

    group = group.sort_values("edge_budget")

    if method == "Backbone-only":
        color = COLORS["backbone"]
    elif "Spatial" in method:
        color = COLORS["spatial"]
    else:
        color = COLORS["global"]

    ax.plot(
        group["edge_budget"],
        group["dispersion_retention"],
        marker="o",
        linewidth=2,
        label=method,
        color=color,
    )

style_axes(
    ax,
    "Total edges per graph",
    "Dispersion retained relative to depth 0",
    "Embedding Dispersion Retention After 5 Steps",
)

ax.set_ylim(0, 1)

save_figure(
    fig,
    "figure_08_oversmoothing_dispersion",
    "Depth-5 dispersion retention averaged across scanners",
)


# ==========================================================
# 3. FIGURE INVENTORY
# ==========================================================

print("\n[3] Saving figure inventory...")

inventory = pd.DataFrame(created)

inventory_path = FIGURES / "FIGURE_INVENTORY.csv"

inventory.to_csv(inventory_path, index=False)


# ==========================================================
# 4. PHASE 2 README
# ==========================================================

print("\n[4] Writing Phase 2 README...")

phase2_readme = """# Phase 2: Research Design and Experimental Protocol

## Research Project

**Cross-Scanner Graph Topology Stability in Computational Histopathology**

This phase develops the experimental framework for investigating how
scanner-related variation affects graph construction, neighborhood
consistency, graph representations, and graph-based learning.

## Research Objectives

1. Quantify graph edge instability across matched scanner acquisitions.
2. Compare feature-based and spatially constrained graph construction.
3. Control graph density, connectivity, and candidate-pool effects.
4. Investigate representation stability and graph coarsening.
5. Evaluate whether stable topology preserves useful tissue information.
6. Assess generalization across appropriately identified independent
   specimens or analysis groups.

## Dataset Strategy

Primary pilot dataset:

- PLISM histopathology images.
- Matched AT2 and GT450 scanner acquisitions.
- GIVH staining condition.
- Registered tile pairs from the Owkin PLISM tile release.

The pilot is intended for feasibility assessment and protocol development.

Larger studies require careful validation of tissue identity, spatial
registration, region overlap, and biological independence.

## Research Design Components

The phase includes:

- Research questions and hypotheses.
- Literature and novelty assessment.
- Dataset selection and sampling criteria.
- Matched-budget graph construction protocols.
- Pretrained feature encoder selection.
- Graph stability metrics and random baselines.
- Specimen-aware statistical analysis plans.
- Graph coarsening and representation evaluation plans.
- Master experimental protocol.

## Planned Graph Comparisons

Candidate baselines include:

- Spatial-only graph.
- Global feature-similarity graph.
- Spatially constrained feature-similarity graph.
- Connectivity-preserving graph variants.

A shared fixed backbone must be evaluated separately from
scanner-dependent feature-selected edges.

## Feature Encoders

The research design considers:

- DINOv2-Small as a general visual feature baseline.
- Pathology-specific pretrained encoders, subject to availability
  and reproducibility checks.

## Statistical Considerations

Graph nodes and overlapping tile regions are not automatically
independent observations.

Inferential comparisons must use defensible biological or
higher-level independent analysis units where available.

## Phase Status

**Status: Research design completed for pilot experiments.**

The design may be refined prospectively before larger studies.
Pilot outcomes should not be used to claim independent validation
or confirmed research novelty.

## Project Folders

- `Docs/`: Research design documentation.
- `literature_review/`: Literature assessment.
- `protocols/`: Experimental procedures and controls.

## Related Phases

- Phase 1: `../dataset_exploration/`
- Phase 3A: `../experimental_research/`
- Phase 3B: `../multi_region_research/`
"""

PHASE2.mkdir(parents=True, exist_ok=True)

(PHASE2 / "README.md").write_text(
    phase2_readme,
    encoding="utf-8"
)


# ==========================================================
# 5. PHASE 3A README
# ==========================================================

print("\n[5] Writing Phase 3A README...")

phase3_readme = """# Phase 3A: PLISM Pilot Experimental Research

## Project

**Cross-Scanner Graph Topology Stability in Computational Histopathology**

This folder contains pilot experiments investigating graph topology
and representation stability across matched histopathology tiles
captured using different scanners.

## Status

**Phase 3A experimental diagnostics completed.**

The current findings are exploratory and limited to one
90-location tissue region.

## Dataset and Feature Representation

- Dataset: PLISM, using registered tiles distributed by Owkin.
- Scanner conditions: AT2 and GT450.
- Staining condition: GIVH.
- Matched nodes: 90.
- Feature encoder: `facebook/dinov2-small`.
- Feature dimensionality: 384.

The two scanner acquisitions correspond to the same selected
tile locations.

## Original Annotation Validation

Official PLISM WSI annotation analysis found:

| Property | Result |
|---|---:|
| Total WSI annotation rows | 310,947 |
| Acquisition groups | 91 |
| GIVH AT2 annotation coordinates | 3,417 |
| GIVH GT450 annotation coordinates | 3,417 |
| Shared coordinate strings | 3,417 |
| Shared coordinate tissue-label agreement | 100% |
| Label conflicts | 0 |

Original WSI coordinates have not been verified against the
registered Owkin tile-coordinate system.

Tissue-type identifiers do not establish patient-level or
specimen-level biological independence.

## Experiment Overview

### 3A.1: Dataset and Provenance Audits

Scripts 01-08 examine:

- Paired tile inventory.
- Scanner acquisition metadata.
- Specimen identity availability.
- Official PLISM annotation resources.
- Annotation schemas.
- Tissue labels and coordinate consistency.

### 3A.2: Matched Edge-Budget Stability

Script 09 compares global and spatial-feature graph construction
with equal edge budgets.

| Budget | Global Jaccard | Spatial Jaccard |
|---|---:|---:|
| 90 | 0.241379 | 0.538462 |
| 135 | 0.285714 | 0.708861 |
| 180 | 0.304348 | 0.791045 |
| 225 | 0.347305 | 0.829268 |

The unrestricted top-M construction resulted in disconnected graphs.

### 3A.3: Random and Connectivity Diagnostics

Script 10 assesses:

- Candidate-pool effects.
- Empirical random-overlap references.
- Graph fragmentation.
- Isolated nodes.

Raw stability scores alone are insufficient to compare methods
with different eligible candidate-pool sizes.

### 3A.4: Connectivity-Preserving Graph Construction

Script 11 introduces a common 89-edge spatial spanning tree
and adds feature-selected edges.

All tested graph configurations are connected.

Residual feature-edge agreement excludes the fixed backbone:

| Budget | Global residual Jaccard | Spatial residual Jaccard |
|---|---:|---:|
| 135 | 0.179487 | 0.483871 |
| 180 | 0.213333 | 0.716981 |
| 225 | 0.276995 | 0.813333 |

Spatially constrained graph construction produced greater
residual edge agreement at all three evaluated budgets.

### 3A.5: Node-Level Structural Stability

Script 12 measures:

- Residual degree distributions.
- Nodes connected only through the backbone.
- Neighbor-set consistency.
- Long-distance versus local edges.
- Cross-scanner degree variation.

### 3A.6: Representation Consistency

Script 13 uses deterministic row-normalized propagation
with self-loops and no learned GNN weights.

At propagation depth 2:

| Method | Budget | Mean paired-node cosine |
|---|---:|---:|
| Global + MST | 135 | 0.984529 |
| Spatial + MST | 135 | 0.985045 |
| Global + MST | 180 | 0.985101 |
| Spatial + MST | 180 | 0.985820 |
| Global + MST | 225 | 0.985559 |
| Spatial + MST | 225 | 0.986704 |
| Backbone-only | 89 | 0.984597 |

These differences are descriptive and have not been assessed
through independent biological replication.

### 3A.7: Oversmoothing and Topology Sensitivity

Script 14 evaluates:

- Within-graph embedding dispersion.
- Cross-scanner representation differences.
- Fixed-feature topology sensitivity.
- Dispersion retention at increasing propagation depths.

Repeated propagation increased cross-scanner similarity
but reduced node embedding dispersion.

This is consistent with smoothing and means that greater
similarity alone should not be interpreted as improved
representation quality.

## Final Figures

All final figures are saved in `outputs/figures/`.

| Figure | Description |
|---|---|
| 01 | Matched-budget graph stability |
| 02 | Stability above random reference |
| 03 | Graph fragmentation |
| 04 | Connected residual-edge stability |
| 05 | Node-neighborhood stability |
| 06 | Topology sensitivity |
| 07 | Representation consistency |
| 08 | Oversmoothing and dispersion retention |

Every figure is generated as a PNG and vector PDF.

The exact files are listed in
`outputs/figures/FIGURE_INVENTORY.csv`.

## Research Findings

The pilot provides evidence that spatially constrained
feature-based graphs have more stable scanner-paired
neighborhood structure under the tested conditions.

The effect remains visible after matching edge budgets,
enforcing connectivity, and excluding a fixed backbone.

However, representation differences are substantially smaller
than raw edge differences under deterministic propagation.

The observed representation consistency is accompanied
by decreased embedding dispersion at greater depths.

## Important Limitations

1. Only one 90-node tissue region was analyzed.
2. Biological specimen identities have not been verified.
3. The original WSI and Owkin coordinate systems have not
   been linked by a validated transform.
4. Candidate edge pools differ between global and spatial methods.
5. Fixed backbone edges artificially raise full-graph agreement.
6. Graph representation analysis is deterministic, not trained GNN evaluation.
7. Graph coarsening and pooling stability remain untested.
8. No downstream classification performance is established.
9. No independent specimen-level inferential conclusions are supported.

## Directory Structure

```text
experimental_research/
├── README.md
├── scripts/
├── Docs/
└── outputs/
    ├── tables/
    ├── reports/
    └── figures/
```

## Reproducing Figures

From the project root:

```bash
python experimental_research/scripts/15_finalize_phase3a.py
```

Requires NumPy, pandas, and Matplotlib.

## Next Research Phase

**Phase 3B: Multi-Region Controlled Experiments**

Location: `../multi_region_research/`

The next phase should prioritize spatially separated matched
regions, tissue attribution, independent analysis-unit verification,
and multi-region experimental replication before strong
generalization claims.

## Research References

- Official PLISM dataset:
  https://doi.org/10.1038/s41597-024-03122-5
- Owkin registered PLISM tiles:
  https://huggingface.co/datasets/owkin/plism-dataset-tiles

## Conclusion

Phase 3A establishes a reproducible exploratory baseline
for cross-scanner graph topology stability.

Its results motivate expanded evaluation, but do not yet
demonstrate clinically useful prediction improvements or
generalization across independent specimens.
"""

(PHASE3 / "README.md").write_text(
    phase3_readme,
    encoding="utf-8"
)


# ==========================================================
# 6. FINAL VALIDATION
# ==========================================================

print("\n[6] Validating final outputs...")

expected_figures = 8

if len(created) != expected_figures:
    raise RuntimeError(
        f"Expected {expected_figures} figures, "
        f"created {len(created)}"
    )

for record in created:
    for extension in ("png", "pdf"):
        path = FIGURES / (
            record["figure"] + "." + extension
        )

        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(
                f"Invalid figure output: {path}"
            )

for readme in [
    PHASE2 / "README.md",
    PHASE3 / "README.md",
]:
    if not readme.is_file():
        raise RuntimeError(f"Missing README: {readme}")


# ==========================================================
# FINISHED
# ==========================================================

print("\n" + "=" * 70)
print("PHASE 2 + PHASE 3A FINALIZATION COMPLETED")
print("=" * 70)

print(f"Final research figures: {len(created)}")
print(f"PNG files: {len(created)}")
print(f"PDF files: {len(created)}")

print("\nREADME files:")
print(PHASE2 / "README.md")
print(PHASE3 / "README.md")

print("\nFigure directory:")
print(FIGURES)

print("\nAll generated outputs validated.")
