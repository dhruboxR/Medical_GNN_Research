
"""
PHASE 5.1 - RESEARCH RESULTS FIGURES

Read-only use of finalized research artifacts.

Creates:
  49_paper_performance_table.csv
  49_retrieval_comparison.png
  49_primary_robustness_effect.png
  49_figure_generation_report.json

No model inference, training, tuning, or graph access.
All reported values come from verified prior outputs.
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

SYNTHESIS = ROOT / "outputs/reports/44_research_synthesis_report.json"
TEST_REPORT = ROOT / "outputs/reports/47_heldout_evaluation_report.json"
TEST_SUMMARY = ROOT / "outputs/tables/47_heldout_retrieval_summary.csv"
TEST_AUDIT = ROOT / "outputs/reports/48_heldout_results_integrity_audit.json"

OUT = ROOT / "outputs/figures"
TABLE_OUT = ROOT / "outputs/tables/49_paper_performance_table.csv"
FIG1 = OUT / "49_retrieval_comparison.png"
FIG2 = OUT / "49_primary_robustness_effect.png"
REPORT_OUT = ROOT / "outputs/reports/49_figure_generation_report.json"

MODELS = ("ResidualGate05", "TopologyConsistentResidual")
EXPECTED_PROTOCOL_HASH = (
    "e6c8516e2dd279ca58b7873050c227ebe3a958a39140869065b36d38b0af78a9"
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def save_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)


print("=" * 65)
print("PHASE 5.1 - FINAL RESEARCH FIGURES")
print("=" * 65)

print("\n[1] Checking verified sources...")

for path in (SYNTHESIS, TEST_REPORT, TEST_SUMMARY, TEST_AUDIT):
    check(path.is_file(), f"Missing source: {path}")

for path in (TABLE_OUT, FIG1, FIG2, REPORT_OUT):
    check(not path.exists(), f"Refusing overwrite: {path}")

synthesis = load_json(SYNTHESIS)
test = load_json(TEST_REPORT)
audit = load_json(TEST_AUDIT)

check(synthesis["step"] == "4.2.29", "Wrong synthesis source")
check(test["step"] == "4.4.1", "Wrong test report")
check(audit["step"] == "4.4.2", "Wrong audit report")
check(audit["status"] == "HELD_OUT_RESULTS_VERIFIED",
      "Test audit not verified")

check(test["frozen_protocol_sha256"] == EXPECTED_PROTOCOL_HASH,
      "Frozen protocol mismatch")
check(audit["frozen_protocol_sha256"] == EXPECTED_PROTOCOL_HASH,
      "Audit protocol mismatch")
check(audit["evaluation_report_sha256"] == sha256(TEST_REPORT),
      "Evaluation report modified")
check(test["summary_csv_sha256"] == sha256(TEST_SUMMARY),
      "Test summary modified")
check(test["test_evaluated"] is True,
      "Missing test evaluation")
check(audit["additional_test_inference_performed"] is False,
      "Unexpected additional test inference")

validation = {
    row["model"]: row
    for row in synthesis["models"]
}

test_rows = load_csv(TEST_SUMMARY)

heldout = {
    (row["model"], row["condition"]): row
    for row in test_rows
}

check(set(validation) == set(MODELS), "Validation model mismatch")
check(set(heldout) == {
    (m, c)
    for m in MODELS
    for c in ("normal_spatial", "scanner_discordant_random")
}, "Test model-condition mismatch")

print("Historical and held-out sources: VERIFIED")

print("\n[2] Preparing paper performance table...")

rows = []

for model in MODELS:
    val = validation[model]

    train_conditions = {
        "normal": float(val["normal_recall_at_1"]),
        "discordant": float(val["discordant_recall_at_1"]),
    }

    test_conditions = {
        "normal": float(
            heldout[(model, "normal_spatial")]["mean_recall_at_1"]
        ),
        "discordant": float(
            heldout[(model, "scanner_discordant_random")]["mean_recall_at_1"]
        ),
    }

    for split, conditions in (
        ("validation", train_conditions),
        ("held_out_test", test_conditions),
    ):
        rows.append({
            "model": model,
            "split": split,
            "normal_recall_percent":
                conditions["normal"] * 100,
            "discordant_recall_percent":
                conditions["discordant"] * 100,
            "robustness_drop_pp":
                (conditions["normal"] - conditions["discordant"]) * 100,
        })

fields = [
    "model", "split",
    "normal_recall_percent",
    "discordant_recall_percent",
    "robustness_drop_pp",
]

save_csv(TABLE_OUT, rows, fields)

print("\n[3] Creating retrieval comparison figure...")

# Each split-condition group contains both frozen models.
categories = [
    ("Validation\nNormal", "validation", "normal_recall_percent"),
    ("Validation\nDiscordant", "validation", "discordant_recall_percent"),
    ("Test\nNormal", "held_out_test", "normal_recall_percent"),
    ("Test\nDiscordant", "held_out_test", "discordant_recall_percent"),
]

lookup = {
    (r["model"], r["split"]): r
    for r in rows
}

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5.5))

for index, model in enumerate(MODELS):
    y = [
        lookup[(model, split)][field]
        for _, split, field in categories
    ]

    positions = x + (index - 0.5) * width

    bars = ax.bar(
        positions,
        y,
        width,
        label=(
            "Residual, λ=0"
            if index == 0 else "Consistency, λ=1"
        ),
    )

    ax.bar_label(
        bars,
        fmt="%.1f",
        padding=3,
        fontsize=9,
    )

ax.set_xticks(x)
ax.set_xticklabels([label for label, _, _ in categories])
ax.set_ylabel("Recall@1 (%)")
ax.set_ylim(0, 110)
ax.set_title("Cross-scanner retrieval under graph perturbations")
ax.legend(loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.text(
    0.5, 0.01,
    "Validation and held-out test regions share biological source material.",
    ha="center",
    fontsize=9,
)

fig.tight_layout(rect=(0, 0.035, 1, 1))
fig.savefig(FIG1, dpi=300, bbox_inches="tight")
plt.close(fig)

print("\n[4] Creating primary effect figure...")

primary = float(
    test["primary_metric"]["difference_percentage_points"]
)

interval = test["paired_region_bootstrap"]["metrics"][
    "discordant_recall_difference_pp"
]

lower = float(interval["lower_95"])
upper = float(interval["upper_95"])

check(lower <= primary <= upper,
      "Primary estimate outside bootstrap interval")

check(abs(primary - float(audit["primary_difference_pp"])) < 1e-6,
      "Primary difference audit mismatch")

fig, ax = plt.subplots(figsize=(8, 3.4))

ax.errorbar(
    [primary],
    [0],
    xerr=np.array([[primary - lower], [upper - primary]]),
    fmt="o",
    markersize=9,
    capsize=6,
    linewidth=2,
)

ax.axvline(0, linestyle="--", linewidth=1)
ax.set_yticks([])
ax.set_xlabel("Consistency minus residual Recall@1 (percentage points)")
ax.set_title("Frozen held-out scanner-discordant improvement")
ax.set_xlim(min(0, lower - 2), upper + 2)

ax.text(
    primary,
    0.13,
    f"+{primary:.2f} pp  [{lower:.2f}, {upper:.2f}]",
    ha="center",
    fontsize=10,
)

ax.set_ylim(-0.4, 0.45)
fig.text(
    0.5, 0.01,
    "Descriptive 95% bootstrap over 10 spatial regions; not patient-level inference.",
    ha="center",
    fontsize=9,
)

fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(FIG2, dpi=300, bbox_inches="tight")
plt.close(fig)

print("\n[5] Saving figure provenance...")

report = {
    "phase": "5",
    "step": "5.1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_synthesis_sha256": sha256(SYNTHESIS),
    "source_test_report_sha256": sha256(TEST_REPORT),
    "source_test_summary_sha256": sha256(TEST_SUMMARY),
    "source_test_audit_sha256": sha256(TEST_AUDIT),
    "frozen_protocol_sha256": EXPECTED_PROTOCOL_HASH,
    "primary_difference_pp": primary,
    "primary_interval": [lower, upper],
    "table_sha256": sha256(TABLE_OUT),
    "figure_retrieval_sha256": sha256(FIG1),
    "figure_primary_effect_sha256": sha256(FIG2),
    "graph_archive_opened": False,
    "model_inference_performed": False,
    "model_training_performed": False,
    "additional_test_evaluation_performed": False,
    "limitations": [
        "Synthetic adjacency changes are not real scanner shifts",
        "Test regions are not independent biological specimens",
        "Bootstrap interval is descriptive",
        "Exploratory validation results preceded the test freeze",
    ],
}

save_json(REPORT_OUT, report)

print("\n" + "=" * 65)
print("PHASE 5.1 - FIGURES COMPLETED")
print("=" * 65)
print("Table:", TABLE_OUT)
print("Figure 1:", FIG1)
print("Figure 2:", FIG2)
print("Provenance:", REPORT_OUT)
print("Graph archive opened: NO")
print("Additional test inference: NO")
print("STEP 5.1 COMPLETED")
