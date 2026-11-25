
"""
PHASE 4 FINAL REPORTING - ARCHITECTURE AND WORKFLOW

Creates two publication diagrams in PNG and SVG:
  51_gated_residual_graphsage_architecture
  51_complete_research_workflow

Uses the frozen protocol for provenance checks.
Does not load features, model weights, or test results
into any inference process.

Existing outputs are never overwritten.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = (
    Path.home() / "MEDICAL_GNN_RESEARCH"
    / "phase4_main_research"
)

PROTOCOL = (
    ROOT / "configs/45_phase43_frozen_evaluation_protocol.json"
)
TEST_REPORT = (
    ROOT / "outputs/reports/47_heldout_evaluation_report.json"
)
TEST_AUDIT = (
    ROOT / "outputs/reports/48_heldout_results_integrity_audit.json"
)
FIG49 = (
    ROOT / "outputs/reports/49_figure_generation_report.json"
)
FIG50 = (
    ROOT / "outputs/reports/50_robustness_xai_figure_report.json"
)

OUT = ROOT / "outputs/figures"
REPORT_OUT = (
    ROOT / "outputs/reports/51_diagram_generation_report.json"
)

FIGURES = {
    "architecture_png":
        OUT / "51_gated_residual_graphsage_architecture.png",
    "architecture_svg":
        OUT / "51_gated_residual_graphsage_architecture.svg",
    "workflow_png":
        OUT / "51_complete_research_workflow.png",
    "workflow_svg":
        OUT / "51_complete_research_workflow.svg",
}

EXPECTED_PROTOCOL_SHA256 = (
    "e6c8516e2dd279ca58b7873050c227ebe3a958a39140869065b36d38b0af78a9"
)

INK = "#243447"
BLUE = "#DCEBFA"
GREEN = "#DDF3E7"
AMBER = "#FFF0D4"
PURPLE = "#ECE3FA"
GRAY = "#EFF2F5"
EDGE = "#54708C"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def block(ax, x, y, w, h, title, subtitle="",
          color=BLUE, title_size=10, subtitle_size=8):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.15,
        edgecolor=EDGE,
        facecolor=color,
    )
    ax.add_patch(patch)

    if subtitle:
        ax.text(
            x + w / 2, y + h * 0.65,
            title, ha="center", va="center",
            fontsize=title_size, weight="bold",
            color=INK,
        )
        ax.text(
            x + w / 2, y + h * 0.32,
            subtitle, ha="center", va="center",
            fontsize=subtitle_size, color=INK,
        )
    else:
        ax.text(
            x + w / 2, y + h / 2,
            title, ha="center", va="center",
            fontsize=title_size, weight="bold",
            color=INK,
        )


def arrow(ax, start, end, label=None, label_offset=(0, 0)):
    a = FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.5,
        color=EDGE,
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(a)

    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(
            mid_x + label_offset[0],
            mid_y + label_offset[1],
            label, ha="center", va="center",
            fontsize=8, color=INK,
        )


def setup(figsize, title):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(
        0.5, 0.965, title,
        ha="center", va="center",
        fontsize=16, weight="bold", color=INK,
    )
    return fig, ax


def export(fig, png, svg):
    fig.savefig(
        png, dpi=300, facecolor="white",
        bbox_inches="tight", pad_inches=0.15,
    )
    fig.savefig(
        svg, facecolor="white",
        bbox_inches="tight", pad_inches=0.15,
    )
    plt.close(fig)


print("=" * 70, flush=True)
print("PHASE 4 - ARCHITECTURE AND WORKFLOW DIAGRAMS", flush=True)
print("=" * 70)

print("\n[1] Verifying research provenance...", flush=True)

sources = [
    PROTOCOL, TEST_REPORT, TEST_AUDIT, FIG49, FIG50
]

for path in sources:
    require(path.is_file(), f"Missing source: {path}")

for path in [*FIGURES.values(), REPORT_OUT]:
    require(
        not path.exists(),
        f"Refusing to overwrite: {path}",
    )

require(
    sha256(PROTOCOL) == EXPECTED_PROTOCOL_SHA256,
    "Frozen protocol hash changed",
)

protocol = read_json(PROTOCOL)
test_report = read_json(TEST_REPORT)
test_audit = read_json(TEST_AUDIT)
figure49 = read_json(FIG49)
figure50 = read_json(FIG50)

require(
    protocol["protocol_status"] == "FROZEN",
    "Unexpected protocol status",
)
require(
    test_report["frozen_protocol_sha256"]
    == EXPECTED_PROTOCOL_SHA256,
    "Test protocol mismatch",
)
require(
    test_audit["evaluation_report_sha256"]
    == sha256(TEST_REPORT),
    "Test report provenance mismatch",
)
require(
    figure49["frozen_protocol_sha256"]
    == EXPECTED_PROTOCOL_SHA256,
    "Previous figure protocol mismatch",
)
require(
    figure50["additional_test_evaluation_performed"] is False,
    "Previous figures recorded additional test evaluation",
)

require(
    set(protocol["models"]) == {
        "ResidualGate05",
        "TopologyConsistentResidual",
    },
    "Frozen model mismatch",
)
require(
    protocol["research_scope"]["feature_dimension"] == 384
    and protocol["research_scope"]["nodes_per_graph"] == 36,
    "Research dimensionality mismatch",
)

print("Frozen research sources: VERIFIED", flush=True)

OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Model architecture diagram
# --------------------------------------------------

print("\n[2] Drawing gated residual GraphSAGE...", flush=True)

fig, ax = setup(
    (11, 9),
    "Gated Residual GraphSAGE Architecture",
)

block(
    ax, 0.31, 0.82, 0.38, 0.085,
    "DINOv2-Small Node Features",
    "X: 36 nodes × 384 dimensions",
    BLUE,
)

ax.text(
    0.5, 0.765,
    "Shared input features feed two parallel pathways",
    ha="center", fontsize=9, color=INK,
)

block(
    ax, 0.075, 0.64, 0.34, 0.085,
    "Identity Path",
    "L2 normalize(X)",
    GREEN,
)
block(
    ax, 0.585, 0.64, 0.34, 0.085,
    "GraphSAGE Layer 1",
    "384 → 256, spatial aggregation",
    PURPLE,
)
block(
    ax, 0.585, 0.51, 0.34, 0.075,
    "GELU Activation", color=PURPLE,
)
block(
    ax, 0.585, 0.38, 0.34, 0.085,
    "GraphSAGE Layer 2",
    "256 → 384, spatial aggregation",
    PURPLE,
)
block(
    ax, 0.585, 0.25, 0.34, 0.085,
    "Graph Embedding",
    "L2 normalize(GNN(X, A))",
    PURPLE,
)
block(
    ax, 0.075, 0.25, 0.34, 0.085,
    "Learned Residual Gate",
    "α = sigmoid(gate_logit)",
    AMBER,
)
block(
    ax, 0.31, 0.125, 0.38, 0.075,
    "Gated Residual Fusion",
    "normalize(identity + α × graph)",
    GREEN,
)
block(
    ax, 0.31, 0.016, 0.38, 0.067,
    "Final 384D Node Embeddings",
    "Cross-scanner retrieval",
    BLUE,
)

arrow(ax, (0.43, 0.82), (0.245, 0.735))
arrow(ax, (0.57, 0.82), (0.755, 0.735))
arrow(ax, (0.755, 0.64), (0.755, 0.585))
arrow(ax, (0.755, 0.51), (0.755, 0.465))
arrow(ax, (0.755, 0.38), (0.755, 0.335))
arrow(ax, (0.415, 0.682), (0.38, 0.203))
arrow(ax, (0.245, 0.25), (0.38, 0.203))
arrow(ax, (0.755, 0.25), (0.58, 0.203))
arrow(ax, (0.5, 0.125), (0.5, 0.083))

ax.text(
    0.5, 0.005,
    "λ=0: standard residual training   |   "
    "λ=1: additional topology-consistency loss during training",
    fontsize=8, ha="center", va="top", color=INK,
)

export(
    fig,
    FIGURES["architecture_png"],
    FIGURES["architecture_svg"],
)


# --------------------------------------------------
# 3. Full research workflow
# --------------------------------------------------

print("\n[3] Drawing full experimental workflow...", flush=True)

fig, ax = setup(
    (14, 10),
    "Cross-Scanner Histopathology Graph Research Workflow",
)

steps = [
    (
        "Matched Histopathology Tiles",
        "AT2 and GT450 scans",
        BLUE,
    ),
    (
        "DINOv2-Small Feature Extraction",
        "384-dimensional node embeddings",
        BLUE,
    ),
    (
        "Spatial Graph Construction",
        "6×6 grids, 36 nodes, 60 undirected edges",
        BLUE,
    ),
    (
        "Spatial Region Partition",
        "40 train / 10 validation / 10 test pairs",
        AMBER,
    ),
    (
        "Frozen Model Development",
        "Residual GraphSAGE; λ=0 versus λ=1",
        PURPLE,
    ),
    (
        "Validation Robustness Analysis",
        "Original and scanner-discordant adjacency",
        PURPLE,
    ),
    (
        "Validation Explainability Analysis",
        "Structural deletion, corrected IG, edge masks",
        GREEN,
    ),
    (
        "Evaluation Protocol Freeze",
        "Six checkpoint hashes and fixed test design",
        AMBER,
    ),
    (
        "Reserved Held-Out Evaluation",
        "10 paired regions, 3 seeds, 5 perturbations",
        GREEN,
    ),
    (
        "Verified Final Research Results",
        "Integrity audit, figures, and documentation",
        GREEN,
    ),
]

left = 0.16
width = 0.68
height = 0.064
start_y = 0.86
pitch = 0.083

for index, (title, subtitle, color) in enumerate(steps):
    y = start_y - index * pitch

    block(
        ax, left, y, width, height,
        title, subtitle, color,
        title_size=10, subtitle_size=8,
    )

    if index < len(steps) - 1:
        arrow(
            ax,
            (0.5, y),
            (0.5, y - pitch + height),
        )

ax.text(
    0.5, 0.055,
    "Held-out test is evaluated only after model "
    "and protocol freeze. No post-test tuning.",
    ha="center", fontsize=9, color=INK,
)

export(
    fig,
    FIGURES["workflow_png"],
    FIGURES["workflow_svg"],
)


# --------------------------------------------------
# 4. Provenance report
# --------------------------------------------------

print("\n[4] Saving diagram provenance...", flush=True)

report = {
    "phase": "4",
    "step": "final_reporting_51",
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "frozen_protocol_sha256": EXPECTED_PROTOCOL_SHA256,
    "source_sha256": {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sources
    },
    "diagram_sha256": {
        str(path.relative_to(ROOT)): sha256(path)
        for path in FIGURES.values()
    },
    "diagrams": list(FIGURES),
    "graph_archive_opened": False,
    "checkpoint_loaded": False,
    "model_inference_performed": False,
    "additional_test_evaluation_performed": False,
    "limitations": [
        "Architecture is a schematic, not a computational graph trace",
        "Workflow summarizes completed research stages",
        "Spatially held-out regions share biological source material",
        "Synthetic topology disagreement is not a real scanner shift",
    ],
}

with REPORT_OUT.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
    f.write("\n")

print("\n" + "=" * 70)
print("PHASE 4 - DIAGRAMS COMPLETED")
print("=" * 70)

for path in FIGURES.values():
    print("Saved:", path)

print("Provenance:", REPORT_OUT)
print("Additional test inference: NO")
print("Existing checkpoints: UNCHANGED")
