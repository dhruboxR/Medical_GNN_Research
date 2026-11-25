
"""
PHASE 4 FINAL REPORTING - ROBUSTNESS AND XAI FIGURES

Read-only use of verified prior results.

Creates:
  50_robustness_degradation.png
  50_ig_fidelity.png
  50_edge_fidelity.png
  50_edge_mask_distribution.png
  50_robustness_xai_figure_report.json

No graph archive access.
No training, model inference, or test reevaluation.
No existing output overwritten.
"""

from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = (
    Path.home()
    / "MEDICAL_GNN_RESEARCH"
    / "phase4_main_research"
)

PERFORMANCE = ROOT / "outputs/tables/49_paper_performance_table.csv"
FIG49_REPORT = ROOT / "outputs/reports/49_figure_generation_report.json"

IG_REPORT = ROOT / "outputs/reports/41_corrected_ig_audit_report.json"
EDGE_REPORT = ROOT / "outputs/reports/43_edge_mask_reliability_report.json"

EDGE_SOURCE = ROOT / "outputs/xai/42_learned_edge_masks.csv"
EDGE_SOURCE_REPORT = ROOT / "outputs/reports/42_learned_edge_mask_report.json"

OUT_DIR = ROOT / "outputs/figures"
REPORT_OUT = ROOT / "outputs/reports/50_robustness_xai_figure_report.json"

FIGURES = {
    "robustness": OUT_DIR / "50_robustness_degradation.png",
    "ig": OUT_DIR / "50_ig_fidelity.png",
    "edge": OUT_DIR / "50_edge_fidelity.png",
    "mask": OUT_DIR / "50_edge_mask_distribution.png",
}

MODELS = ("ResidualGate05", "TopologyConsistentResidual")
LABELS = ("Residual, λ=0", "Consistency, λ=1")
SCANNERS = ("AT2", "GT450")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite(value, name):
    v = float(value)
    require(np.isfinite(v), f"Invalid numerical value: {name}")
    return v


def prepare_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


print("=" * 70, flush=True)
print("PHASE 4 - ROBUSTNESS AND XAI FIGURE GENERATION", flush=True)
print("=" * 70, flush=True)

print("\n[1] Verifying source artifacts...", flush=True)

sources = (
    PERFORMANCE, FIG49_REPORT, IG_REPORT,
    EDGE_REPORT, EDGE_SOURCE, EDGE_SOURCE_REPORT
)

for path in sources:
    require(path.is_file(), f"Missing artifact: {path}")

for path in (*FIGURES.values(), REPORT_OUT):
    require(not path.exists(), f"Refusing overwrite: {path}")

fig49 = read_json(FIG49_REPORT)
ig = read_json(IG_REPORT)
edge = read_json(EDGE_REPORT)
edge_source = read_json(EDGE_SOURCE_REPORT)

require(
    fig49["table_sha256"] == sha256(PERFORMANCE),
    "Final performance table hash mismatch"
)
require(
    ig["step"] == "4.2.26"
    and edge["step"] == "4.2.28"
    and edge_source["step"] == "4.2.27",
    "Historical XAI phase mismatch"
)
require(
    ig["test_evaluated"] is False
    and edge["test_evaluated"] is False
    and edge_source["test_evaluated"] is False,
    "XAI source test status mismatch"
)
require(
    ig["graph_archive_sha256"] == edge["graph_archive_sha256"]
    == edge_source["graph_archive_sha256"],
    "XAI graph provenance mismatch"
)
require(
    edge_source["edge_table_sha256"] == sha256(EDGE_SOURCE),
    "Original learned edge masks changed"
)
require(
    edge["source_edge_sha256"] == sha256(EDGE_SOURCE),
    "Edge audit provenance mismatch"
)

print("Sources and SHA256 references: VERIFIED", flush=True)


# -------------------------------------------------
# 2. Robustness degradation
# -------------------------------------------------

print("\n[2] Creating robustness degradation figure...", flush=True)

performance = read_csv(PERFORMANCE)

require(len(performance) == 4, "Expected four performance rows")

lookup = {}

for row in performance:
    key = (row["model"], row["split"])
    require(key not in lookup, "Duplicate performance row")
    lookup[key] = row

expected = {
    (model, split)
    for model in MODELS
    for split in ("validation", "held_out_test")
}
require(set(lookup) == expected, "Incomplete performance table")

groups = ("Validation", "Held-out test")
x = np.arange(2)
width = 0.34

fig, ax = plt.subplots(figsize=(8, 5))

for j, model in enumerate(MODELS):
    values = [
        finite(
            lookup[(model, split)]["robustness_drop_pp"],
            "robustness_drop_pp",
        )
        for split in ("validation", "held_out_test")
    ]

    bars = ax.bar(
        x + (j - 0.5) * width,
        values,
        width,
        label=LABELS[j],
    )
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)

ax.set_xticks(x, groups)
ax.set_ylabel("Recall@1 degradation (percentage points)")
ax.set_title("Retrieval degradation under scanner-discordant topology")
ax.legend()
prepare_axes(ax)
save_figure(fig, FIGURES["robustness"])


# -------------------------------------------------
# 3. Corrected IG fidelity
# -------------------------------------------------

print("\n[3] Creating corrected IG fidelity figure...", flush=True)

ig_models = ig["model_summary"]
require(set(ig_models) == set(MODELS), "IG model coverage mismatch")

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(2)

for j, model in enumerate(MODELS):
    record = ig_models[model]
    values = [
        finite(
            record[f"{scanner.lower()}_signed_positive_excess"],
            "IG signed fidelity",
        )
        for scanner in SCANNERS
    ]

    bars = ax.bar(
        x + (j - 0.5) * width,
        values,
        width,
        label=LABELS[j],
    )
    ax.bar_label(bars, fmt="%+.3f", padding=3, fontsize=10)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x, SCANNERS)
ax.set_ylabel("Deletion score-drop excess")
ax.set_title("Corrected signed Integrated Gradients fidelity")
ax.legend()
prepare_axes(ax)
save_figure(fig, FIGURES["ig"])


# -------------------------------------------------
# 4. Learned edge-mask fidelity
# -------------------------------------------------

print("\n[4] Creating learned edge-mask fidelity figure...", flush=True)

edge_models = edge["model_results"]
require(set(edge_models) == set(MODELS), "Edge model coverage mismatch")

fig, ax = plt.subplots(figsize=(8, 5))

for j, model in enumerate(MODELS):
    record = edge_models[model]
    values = [
        finite(
            record[f"{scanner.lower()}_fidelity_excess"],
            "Edge fidelity",
        )
        for scanner in SCANNERS
    ]

    bars = ax.bar(
        x + (j - 0.5) * width,
        values,
        width,
        label=LABELS[j],
    )
    ax.bar_label(bars, fmt="%+.3f", padding=3, fontsize=10)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x, SCANNERS)
ax.set_ylabel("Deletion score-drop excess")
ax.set_title("Learned edge-mask deletion fidelity")
ax.legend()
prepare_axes(ax)
save_figure(fig, FIGURES["edge"])


# -------------------------------------------------
# 5. Learned edge-mask distribution
# -------------------------------------------------

print("\n[5] Creating learned mask distribution figure...", flush=True)

mask_rows = read_csv(EDGE_SOURCE)
require(
    len(mask_rows) == 7200,
    "Expected 7200 original learned edge mask weights"
)

weights = {m: [] for m in MODELS}
unique_keys = set()

for row in mask_rows:
    model = row["model"]
    require(model in MODELS, "Unexpected model in mask source")

    key = (
        model,
        int(row["seed"]),
        row["region_id"],
        row["scanner"],
        int(row["edge_index"]),
    )
    require(key not in unique_keys, "Duplicate edge mask")
    unique_keys.add(key)

    w = finite(row["mask_weight"], "mask_weight")
    require(0 <= w <= 1, "Mask outside [0, 1]")
    weights[model].append(w)

for model in MODELS:
    require(
        len(weights[model]) == 3600,
        "Incomplete learned mask data"
    )

fig, ax = plt.subplots(figsize=(9, 5))

bins = np.linspace(0, 1, 26)

for model, label in zip(MODELS, LABELS):
    ax.hist(
        weights[model],
        bins=bins,
        alpha=0.55,
        density=True,
        label=label,
    )

ax.set_xlabel("Learned edge-mask weight")
ax.set_ylabel("Density")
ax.set_title("Distribution of optimized edge masks")
ax.legend()
prepare_axes(ax)
save_figure(fig, FIGURES["mask"])


# -------------------------------------------------
# 6. Save provenance
# -------------------------------------------------

print("\n[6] Saving figure provenance...", flush=True)

report = {
    "phase": "4",
    "step": "final_reporting_50",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_sha256": {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sources
    },
    "figures_sha256": {
        str(path.relative_to(ROOT)): sha256(path)
        for path in FIGURES.values()
    },
    "mask_rows": len(mask_rows),
    "graph_archive_opened": False,
    "checkpoint_loaded": False,
    "model_training_performed": False,
    "model_inference_performed": False,
    "additional_test_evaluation_performed": False,
    "limitations": [
        "Validation XAI results are exploratory",
        "IG and edge fidelity use different interventions",
        "Learned edge masks were optimized on validation regions",
        "Edge deletion random controls are not edge-type matched",
        "Histopathology regions share biological source material",
    ],
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
with REPORT_OUT.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2, allow_nan=False)
    f.write("\n")

print("\n" + "=" * 70)
print("PHASE 4 - ADDITIONAL FIGURES COMPLETED")
print("=" * 70)

for path in FIGURES.values():
    print("Saved:", path)

print("Provenance:", REPORT_OUT)
print("Additional test inference: NO")
print("Previous models: UNCHANGED")
