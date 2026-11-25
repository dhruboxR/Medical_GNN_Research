
"""
PHASE 4.4.2 - FINAL HELD-OUT RESULTS AUDIT

READ-ONLY analysis of Phase 4.4.1 outputs.

Verifies:
  - Frozen protocol and evaluation-report hashes
  - All four saved evaluation CSV hashes
  - Complete seed/region/condition/repetition design
  - Reproduced aggregate Recall@1 and MRR
  - Reproduced paired primary difference
  - Reproduced region-bootstrap confidence interval
  - Positive/negative/unchanged region counts

Does not load graph features or checkpoints.
Does not perform model inference.
Does not modify the held-out evaluation.
"""

from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json

import numpy as np


ROOT = (
    Path.home()
    / "MEDICAL_GNN_RESEARCH"
    / "phase4_main_research"
)

CONFIG = ROOT / "configs/45_phase43_frozen_evaluation_protocol.json"
REPORT = ROOT / "outputs/reports/47_heldout_evaluation_report.json"

DETAIL = ROOT / "outputs/tables/47_heldout_retrieval_detail.csv"
SEED_TABLE = ROOT / "outputs/tables/47_heldout_retrieval_by_seed.csv"
REGION_TABLE = ROOT / "outputs/tables/47_heldout_paired_regions.csv"
SUMMARY_TABLE = ROOT / "outputs/tables/47_heldout_retrieval_summary.csv"

OUTPUT = ROOT / "outputs/reports/48_heldout_results_integrity_audit.json"

EXPECTED_PROTOCOL_HASH = (
    "e6c8516e2dd279ca58b7873050c227ebe3a958a39140869065b36d38b0af78a9"
)

MODELS = ("ResidualGate05", "TopologyConsistentResidual")
SEEDS = (42, 123, 2026)
CONDITIONS = ("normal_spatial", "scanner_discordant_random")
REPETITIONS = 5
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 43001

TOL = 1e-6


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
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


def close(actual, expected, label, tolerance=TOL):
    require(
        np.isfinite(actual)
        and np.isfinite(expected)
        and abs(actual - expected) <= tolerance,
        f"Mismatch in {label}: "
        f"calculated={actual}, reported={expected}"
    )


def mean(items, field):
    return float(np.mean([float(row[field]) for row in items]))


def save_new_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(
            content, f, indent=2,
            sort_keys=True, allow_nan=False
        )
        f.write("\n")


print("=" * 70)
print("PHASE 4.4.2 - HELD-OUT RESULTS INTEGRITY AUDIT")
print("=" * 70)


# --------------------------------------------------
# 1. Verify provenance and saved outputs
# --------------------------------------------------

print("\n[1] Verifying frozen evaluation artifacts...")

for path in (
    CONFIG, REPORT, DETAIL, SEED_TABLE,
    REGION_TABLE, SUMMARY_TABLE
):
    require(path.is_file(), f"Missing file: {path}")

require(
    not OUTPUT.exists(),
    "Audit already exists: refusing overwrite"
)

require(
    sha256(CONFIG) == EXPECTED_PROTOCOL_HASH,
    "Frozen protocol changed"
)

protocol = read_json(CONFIG)
report = read_json(REPORT)

require(
    protocol["protocol_status"] == "FROZEN",
    "Frozen protocol invalid"
)

require(
    report["frozen_protocol_sha256"] == EXPECTED_PROTOCOL_HASH,
    "Evaluation used different protocol"
)

require(
    report["step"] == "4.4.1"
    and report["test_evaluated"] is True,
    "Expected completed held-out evaluation"
)

require(
    report["model_training_performed"] is False
    and report["checkpoint_selection_performed"] is False
    and report["hyperparameter_tuning_performed"] is False,
    "Unexpected adaptive model operation"
)

hashes = {
    "detail_csv_sha256": DETAIL,
    "seed_csv_sha256": SEED_TABLE,
    "region_csv_sha256": REGION_TABLE,
    "summary_csv_sha256": SUMMARY_TABLE,
}

for field, path in hashes.items():
    require(
        report[field] == sha256(path),
        f"Saved table hash mismatch: {path}"
    )

detail = read_csv(DETAIL)
seed_rows = read_csv(SEED_TABLE)
region_rows = read_csv(REGION_TABLE)
summary_rows = read_csv(SUMMARY_TABLE)

require(len(detail) == 360, "Expected 360 detail rows")
require(len(seed_rows) == 12, "Expected 12 seed rows")
require(len(region_rows) == 10, "Expected 10 region rows")
require(len(summary_rows) == 4, "Expected 4 summary rows")

print("Saved CSV hashes and row counts: VERIFIED")


# --------------------------------------------------
# 2. Check complete evaluation design
# --------------------------------------------------

print("\n[2] Reconstructing all test measurements...")

regions = sorted(set(row["region_id"] for row in detail))

require(len(regions) == 10, "Expected 10 test regions")

detail_index = {}

for row in detail:
    model = row["model"]
    seed = int(row["seed"])
    region = row["region_id"]
    condition = row["condition"]
    repetition = int(row["repetition"])

    require(model in MODELS, "Unexpected model")
    require(seed in SEEDS, "Unexpected seed")
    require(condition in CONDITIONS, "Unexpected condition")

    expected_repetitions = (
        (0,) if condition == "normal_spatial"
        else tuple(range(REPETITIONS))
    )

    require(
        repetition in expected_repetitions,
        "Unexpected repetition"
    )

    recall = float(row["recall_at_1"])
    mrr = float(row["mrr"])

    require(
        np.isfinite(recall)
        and np.isfinite(mrr)
        and 0 <= recall <= mrr <= 1 + TOL,
        "Invalid retrieval metrics"
    )

    key = (model, seed, region, condition, repetition)

    require(
        key not in detail_index,
        "Duplicate test result"
    )

    detail_index[key] = row

expected_keys = {
    (model, seed, region, condition, rep)
    for model in MODELS
    for seed in SEEDS
    for region in regions
    for condition in CONDITIONS
    for rep in (
        (0,) if condition == "normal_spatial"
        else range(REPETITIONS)
    )
}

require(
    set(detail_index) == expected_keys,
    "Missing or additional test observations"
)

print("All 360 prespecified observations: VERIFIED")


# --------------------------------------------------
# 3. Reproduce summary and seed tables
# --------------------------------------------------

print("\n[3] Verifying aggregate results...")

summary_index = {}
for row in summary_rows:
    key = (row["model"], row["condition"])
    require(key not in summary_index, "Duplicate summary")
    summary_index[key] = row

require(
    set(summary_index) == {
        (m, c) for m in MODELS for c in CONDITIONS
    },
    "Missing model-condition summary"
)

seed_index = {}
for row in seed_rows:
    key = (row["model"], int(row["seed"]), row["condition"])
    require(key not in seed_index, "Duplicate seed row")
    seed_index[key] = row

require(
    set(seed_index) == {
        (m, s, c)
        for m in MODELS
        for s in SEEDS
        for c in CONDITIONS
    },
    "Missing seed summaries"
)

summary_values = {}

for model in MODELS:
    for condition in CONDITIONS:
        subset = [
            row for row in detail
            if row["model"] == model
            and row["condition"] == condition
        ]

        expected_count = 30 if condition == "normal_spatial" else 150

        require(
            len(subset) == expected_count,
            "Incorrect condition sample count"
        )

        rec = mean(subset, "recall_at_1")
        mrr = mean(subset, "mrr")

        saved = summary_index[(model, condition)]

        close(rec, float(saved["mean_recall_at_1"]), "summary Recall@1")
        close(mrr, float(saved["mean_mrr"]), "summary MRR")
        require(
            int(saved["measurements"]) == expected_count,
            "Incorrect summary count"
        )

        summary_values[(model, condition)] = {
            "recall_at_1": rec,
            "mrr": mrr,
        }

        for seed in SEEDS:
            seed_subset = [
                row for row in subset
                if int(row["seed"]) == seed
            ]

            saved_seed = seed_index[(model, seed, condition)]

            close(
                mean(seed_subset, "recall_at_1"),
                float(saved_seed["mean_recall_at_1"]),
                "seed Recall@1"
            )
            close(
                mean(seed_subset, "mrr"),
                float(saved_seed["mean_mrr"]),
                "seed MRR"
            )

print("Aggregate Recall@1 and MRR: VERIFIED")


# --------------------------------------------------
# 4. Reproduce paired region measurements
# --------------------------------------------------

print("\n[4] Verifying paired region differences...")

region_index = {}
for row in region_rows:
    region = row["region_id"]
    require(
        region not in region_index,
        "Duplicate region audit"
    )
    region_index[region] = row

require(
    set(region_index) == set(regions),
    "Missing paired region"
)

paired = {}

for region in regions:
    values = defaultdict(list)

    for seed in SEEDS:
        for condition in CONDITIONS:
            for repetition in (
                (0,) if condition == "normal_spatial"
                else range(REPETITIONS)
            ):
                old = detail_index[(
                    "ResidualGate05", seed,
                    region, condition, repetition
                )]
                new = detail_index[(
                    "TopologyConsistentResidual", seed,
                    region, condition, repetition
                )]

                values[(condition, "recall")].append(
                    float(new["recall_at_1"])
                    - float(old["recall_at_1"])
                )

                values[(condition, "mrr")].append(
                    float(new["mrr"]) - float(old["mrr"])
                )

    pair = {
        "discordant_recall_difference_pp":
            float(np.mean(
                values[("scanner_discordant_random", "recall")]
            ) * 100),
        "discordant_mrr_difference":
            float(np.mean(
                values[("scanner_discordant_random", "mrr")]
            )),
        "normal_recall_difference_pp":
            float(np.mean(
                values[("normal_spatial", "recall")]
            ) * 100),
        "normal_mrr_difference":
            float(np.mean(
                values[("normal_spatial", "mrr")]
            )),
    }

    for field, result in pair.items():
        close(
            result,
            float(region_index[region][field]),
            f"{region} {field}"
        )

    paired[region] = pair

print("All ten paired region differences: VERIFIED")


# --------------------------------------------------
# 5. Reproduce primary result and bootstrap
# --------------------------------------------------

print("\n[5] Reproducing primary effect and uncertainty...")

old_normal = summary_values[
    ("ResidualGate05", "normal_spatial")
]["recall_at_1"]

new_normal = summary_values[
    ("TopologyConsistentResidual", "normal_spatial")
]["recall_at_1"]

old_discordant = summary_values[
    ("ResidualGate05", "scanner_discordant_random")
]["recall_at_1"]

new_discordant = summary_values[
    ("TopologyConsistentResidual", "scanner_discordant_random")
]["recall_at_1"]

primary_pp = (new_discordant - old_discordant) * 100

close(
    primary_pp,
    float(report["primary_metric"]["difference_percentage_points"]),
    "primary test difference"
)

secondary = report["secondary_results"]

checks = {
    "normal_difference_pp": (new_normal - old_normal) * 100,
    "residual_normal_recall": old_normal,
    "consistency_normal_recall": new_normal,
    "residual_discordant_recall": old_discordant,
    "consistency_discordant_recall": new_discordant,
    "residual_robustness_drop_pp": (old_normal - old_discordant) * 100,
    "consistency_robustness_drop_pp": (new_normal - new_discordant) * 100,
}

for name, value in checks.items():
    close(value, float(secondary[name]), name)

rng = np.random.default_rng(BOOTSTRAP_SEED)

draw_indices = rng.integers(
    0, len(regions),
    size=(BOOTSTRAP_REPS, len(regions))
)

bootstrap = {}

for field in (
    "discordant_recall_difference_pp",
    "discordant_mrr_difference",
    "normal_recall_difference_pp",
    "normal_mrr_difference",
):
    values = np.asarray(
        [paired[region][field] for region in regions],
        dtype=np.float64
    )

    estimates = values[draw_indices].mean(axis=1)
    lower, upper = np.quantile(
        estimates, [0.025, 0.975]
    )

    observed = {
        "mean": float(values.mean()),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "positive_regions": int(np.sum(values > 1e-10)),
        "negative_regions": int(np.sum(values < -1e-10)),
        "unchanged_regions": int(np.sum(np.abs(values) <= 1e-10)),
    }

    saved = report["paired_region_bootstrap"]["metrics"][field]

    for metric, value in observed.items():
        if metric.endswith("_regions"):
            require(
                value == int(saved[metric]),
                f"Region sign count mismatch: {field}"
            )
        else:
            close(
                value, float(saved[metric]),
                f"{field} {metric}"
            )

    bootstrap[field] = observed

primary_interval = bootstrap[
    "discordant_recall_difference_pp"
]

print(
    f"Primary difference: {primary_pp:+.4f} pp"
)
print(
    "Descriptive 95% interval: "
    f"[{primary_interval['lower_95']:+.4f}, "
    f"{primary_interval['upper_95']:+.4f}] pp"
)
print(
    "Improved regions:",
    primary_interval["positive_regions"], "/ 10"
)


# --------------------------------------------------
# 6. Save audit without altering evaluation outputs
# --------------------------------------------------

print("\n[6] Writing final integrity audit...")

audit = {
    "phase": "4.4",
    "step": "4.4.2",
    "status": "HELD_OUT_RESULTS_VERIFIED",
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "frozen_protocol_sha256": sha256(CONFIG),
    "evaluation_report_sha256": sha256(REPORT),
    "verified_source_hashes": {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (
            DETAIL, SEED_TABLE,
            REGION_TABLE, SUMMARY_TABLE
        )
    },
    "verified_observations": len(detail),
    "verified_seed_summaries": len(seed_rows),
    "verified_regions": len(regions),
    "primary_difference_pp": float(primary_pp),
    "primary_bootstrap": primary_interval,
    "summary_values": [
        {
            "model": model,
            "condition": condition,
            **summary_values[(model, condition)]
        }
        for model in MODELS
        for condition in CONDITIONS
    ],
    "test_evaluated_previously": True,
    "additional_test_inference_performed": False,
    "training_performed": False,
    "checkpoint_selection_performed": False,
    "hyperparameter_tuning_performed": False,
    "graph_archive_opened": False,
    "checkpoints_opened": False,
    "evaluation_source_files_modified": False,
    "next_stage": "Phase_5_results_and_manuscript",
}

save_new_json(OUTPUT, audit)

print("\n" + "=" * 70)
print("PHASE 4.4.2 - RESULTS AUDIT COMPLETED")
print("=" * 70)

print("Primary effect:", f"{primary_pp:+.4f} pp")
print(
    "Interval:",
    f"[{primary_interval['lower_95']:+.4f}, "
    f"{primary_interval['upper_95']:+.4f}] pp"
)
print("Saved:", OUTPUT)
print("Additional model inference: NO")
print("Source artifacts modified: NO")
print("Test evaluation: ALREADY COMPLETED")
print("Next: PHASE 5")
