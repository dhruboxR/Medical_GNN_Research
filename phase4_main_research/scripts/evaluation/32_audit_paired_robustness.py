
"""
PHASE 4.2.17 - PAIRED ROBUSTNESS AUDIT

Compare Phase 4.2.15 ResidualGate05 against
Phase 4.2.16 TopologyConsistentResidual.

Both models are evaluated on the same:
    10 validation regions
    3 training seeds
    5 topology randomization repetitions

Metrics:
    Paired Recall@1 difference
    Paired MRR difference
    Per-seed differences
    Per-region differences
    Descriptive region-cluster bootstrap interval

No training.
No checkpoint selection.
No test evaluation.
No existing artifact overwritten.
"""

from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json

import numpy as np


BASE = (
    Path.home()
    / "MEDICAL_GNN_RESEARCH"
    / "phase4_main_research"
)

OLD_TABLE = (
    BASE / "outputs/tables"
    / "30_topology_robustness_by_region.csv"
)

NEW_TABLE = (
    BASE / "outputs/tables"
    / "31_topology_consistent_robustness.csv"
)

OLD_REPORT = (
    BASE / "outputs/reports"
    / "30_topology_robustness_report.json"
)

NEW_REPORT = (
    BASE / "outputs/reports"
    / "31_topology_consistent_report.json"
)

PAIRED_OUTPUT = (
    BASE / "outputs/tables"
    / "32_paired_robustness_differences.csv"
)

REGION_OUTPUT = (
    BASE / "outputs/tables"
    / "32_robustness_by_region.csv"
)

SEED_OUTPUT = (
    BASE / "outputs/tables"
    / "32_robustness_by_seed.csv"
)

REPORT_OUTPUT = (
    BASE / "outputs/reports"
    / "32_paired_robustness_audit.json"
)

SEEDS = (42, 123, 2026)
REPETITIONS = range(5)
BOOTSTRAP_REPETITIONS = 5000
BOOTSTRAP_SEED = 4217


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def read_csv(path):
    with path.open(
        newline="", encoding="utf-8"
    ) as stream:
        return list(csv.DictReader(stream))


def save_csv_new(path, rows, columns):
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    with path.open(
        "x", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(rows)


def save_json_new(path, data):
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    with path.open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(data, stream, indent=2)


def key_for(row):
    return (
        int(row["seed"]),
        int(row["repetition"]),
        row["region_id"],
    )


print("=" * 70)
print("PHASE 4.2.17 - PAIRED ROBUSTNESS AUDIT")
print("=" * 70)


# --------------------------------------------------
# 1. Source verification
# --------------------------------------------------

print("\n[1] Verifying historical robustness outputs...")

for path in (
    OLD_TABLE,
    NEW_TABLE,
    OLD_REPORT,
    NEW_REPORT,
):
    if not path.is_file():
        raise RuntimeError(
            f"Missing input: {path}"
        )

for path in (
    PAIRED_OUTPUT,
    REGION_OUTPUT,
    SEED_OUTPUT,
    REPORT_OUTPUT,
):
    if path.exists():
        raise RuntimeError(
            f"Refusing overwrite: {path}"
        )

old_report = read_json(OLD_REPORT)
new_report = read_json(NEW_REPORT)

if (
    old_report["detail_table_sha256"]
    != sha256(OLD_TABLE)
):
    raise RuntimeError(
        "Historical robustness CSV hash mismatch."
    )

if (
    new_report["robustness_table_sha256"]
    != sha256(NEW_TABLE)
):
    raise RuntimeError(
        "Consistency robustness CSV hash mismatch."
    )

if (
    old_report["graph_archive_sha256"]
    != new_report["graph_archive_sha256"]
    or old_report["protocol_sha256"]
    != new_report["protocol_sha256"]
):
    raise RuntimeError(
        "Experiment provenance mismatch."
    )

if (
    old_report["test_evaluated"]
    or new_report["test_evaluated"]
):
    raise RuntimeError(
        "Prior test evaluation detected."
    )

if (
    old_report["repetitions"] != 5
    or old_report["validation_regions"] != 10
    or new_report["validation_pairs"] != 10
):
    raise RuntimeError(
        "Unexpected experimental design."
    )


# --------------------------------------------------
# 2. Match exact experimental conditions
# --------------------------------------------------

print("\n[2] Matching seed, region, and repetition...")

old_rows = [
    row for row in read_csv(OLD_TABLE)
    if (
        row["model"] == "ResidualGate05"
        and row["condition"]
        == "scanner_discordant_random"
    )
]

new_rows = read_csv(NEW_TABLE)

if len(old_rows) != 150 or len(new_rows) != 150:
    raise RuntimeError(
        "Expected 150 rows per experiment."
    )

old_index = {}
new_index = {}

for row in old_rows:
    key = key_for(row)
    if key in old_index:
        raise RuntimeError(
            "Duplicate historical condition."
        )
    old_index[key] = row

for row in new_rows:
    key = key_for(row)
    if key in new_index:
        raise RuntimeError(
            "Duplicate consistency condition."
        )
    new_index[key] = row

if set(old_index) != set(new_index):
    raise RuntimeError(
        "Paired experimental keys do not match."
    )

regions = sorted({
    key[2] for key in old_index
})

if len(regions) != 10:
    raise RuntimeError(
        "Expected 10 validation regions."
    )

expected_keys = {
    (seed, repetition, region)
    for seed in SEEDS
    for repetition in REPETITIONS
    for region in regions
}

if set(old_index) != expected_keys:
    raise RuntimeError(
        "Incomplete seed/repetition/region design."
    )

paired_rows = []

for key in sorted(expected_keys):
    old = old_index[key]
    new = new_index[key]

    old_recall = float(old["recall_at_1"])
    new_recall = float(new["recall_at_1"])

    old_mrr = float(old["mrr"])
    new_mrr = float(new["mrr"])

    if not all(
        np.isfinite(v)
        for v in (
            old_recall, new_recall,
            old_mrr, new_mrr
        )
    ):
        raise RuntimeError(
            "Non-finite retrieval metric."
        )

    if any(
        not 0 <= v <= 1
        for v in (
            old_recall, new_recall,
            old_mrr, new_mrr
        )
    ):
        raise RuntimeError(
            "Retrieval metric outside [0,1]."
        )

    paired_rows.append({
        "seed": key[0],
        "repetition": key[1],
        "region_id": key[2],
        "baseline_recall_at_1": old_recall,
        "consistency_recall_at_1": new_recall,
        "recall_difference_pp":
            (new_recall - old_recall) * 100,
        "baseline_mrr": old_mrr,
        "consistency_mrr": new_mrr,
        "mrr_difference": new_mrr - old_mrr,
    })

print("Paired measurements:", len(paired_rows))
print("Distinct regions:", len(regions))
print("Test partition: NOT EVALUATED")


# --------------------------------------------------
# 3. Aggregate by training seed
# --------------------------------------------------

print("\n[3] Computing seed-level improvements...")

seed_rows = []

for seed in SEEDS:
    subset = [
        row for row in paired_rows
        if row["seed"] == seed
    ]

    result = {
        "seed": seed,
        "measurements": len(subset),
        "baseline_mean_recall_at_1": float(
            np.mean([
                r["baseline_recall_at_1"]
                for r in subset
            ])
        ),
        "consistency_mean_recall_at_1": float(
            np.mean([
                r["consistency_recall_at_1"]
                for r in subset
            ])
        ),
        "recall_improvement_pp": float(
            np.mean([
                r["recall_difference_pp"]
                for r in subset
            ])
        ),
        "mrr_improvement": float(
            np.mean([
                r["mrr_difference"]
                for r in subset
            ])
        ),
    }

    seed_rows.append(result)

    print(
        f"  Seed {seed} | "
        f"Baseline "
        f"{result['baseline_mean_recall_at_1']:.4f} | "
        f"Consistency "
        f"{result['consistency_mean_recall_at_1']:.4f} | "
        f"Difference "
        f"{result['recall_improvement_pp']:+.2f} pp"
    )


# --------------------------------------------------
# 4. Aggregate by independent spatial region
# --------------------------------------------------

print("\n[4] Computing region-level differences...")

region_rows = []

for region in regions:
    subset = [
        row for row in paired_rows
        if row["region_id"] == region
    ]

    if len(subset) != 15:
        raise RuntimeError(
            "Region missing paired measurements."
        )

    region_rows.append({
        "region_id": region,
        "measurements": len(subset),
        "baseline_mean_recall_at_1": float(
            np.mean([
                r["baseline_recall_at_1"]
                for r in subset
            ])
        ),
        "consistency_mean_recall_at_1": float(
            np.mean([
                r["consistency_recall_at_1"]
                for r in subset
            ])
        ),
        "recall_improvement_pp": float(
            np.mean([
                r["recall_difference_pp"]
                for r in subset
            ])
        ),
        "mrr_improvement": float(
            np.mean([
                r["mrr_difference"]
                for r in subset
            ])
        ),
    })

region_differences = np.array(
    [
        r["recall_improvement_pp"]
        for r in region_rows
    ],
    dtype=np.float64,
)

positive_regions = int(
    (region_differences > 1e-10).sum()
)
negative_regions = int(
    (region_differences < -1e-10).sum()
)
unchanged_regions = (
    len(regions)
    - positive_regions
    - negative_regions
)

print("Regions improved:", positive_regions)
print("Regions worsened:", negative_regions)
print("Regions unchanged:", unchanged_regions)


# --------------------------------------------------
# 5. Descriptive region-cluster bootstrap
# --------------------------------------------------

print("\n[5] Estimating region-level uncertainty...")

rng = np.random.default_rng(
    BOOTSTRAP_SEED
)

bootstrapped_differences = np.empty(
    BOOTSTRAP_REPETITIONS,
    dtype=np.float64,
)

for i in range(BOOTSTRAP_REPETITIONS):
    indices = rng.integers(
        0,
        len(regions),
        size=len(regions),
    )

    bootstrapped_differences[i] = (
        region_differences[indices].mean()
    )

lower, upper = np.quantile(
    bootstrapped_differences,
    [0.025, 0.975],
)

overall_difference = float(
    region_differences.mean()
)

overall_old = float(
    np.mean([
        r["baseline_recall_at_1"]
        for r in paired_rows
    ])
)

overall_new = float(
    np.mean([
        r["consistency_recall_at_1"]
        for r in paired_rows
    ])
)

overall_mrr_difference = float(
    np.mean([
        r["mrr_difference"]
        for r in paired_rows
    ])
)

# Verify reported historical means.
if abs(
    overall_old
    - old_report["models"][
        next(
            i for i, item in enumerate(
                old_report["models"]
            )
            if (
                item["model"] == "ResidualGate05"
                and item["condition"]
                == "scanner_discordant_random"
            )
        )
    ]["mean_recall_at_1"]
) > 1e-6:
    raise RuntimeError(
        "Historical recall reproduction failed."
    )

if abs(
    overall_new
    - new_report[
        "mean_scanner_discordant_recall_at_1"
    ]
) > 1e-6:
    raise RuntimeError(
        "Consistency recall reproduction failed."
    )

print(
    "Overall paired improvement:",
    f"{overall_difference:+.2f} pp"
)

print(
    "Descriptive 95% region bootstrap interval:",
    f"[{lower:+.2f}, {upper:+.2f}] pp"
)


# --------------------------------------------------
# 6. Save the audit
# --------------------------------------------------

print("\n[6] Saving paired audit outputs...")

save_csv_new(
    PAIRED_OUTPUT,
    paired_rows,
    [
        "seed", "repetition", "region_id",
        "baseline_recall_at_1",
        "consistency_recall_at_1",
        "recall_difference_pp",
        "baseline_mrr",
        "consistency_mrr",
        "mrr_difference",
    ],
)

save_csv_new(
    REGION_OUTPUT,
    region_rows,
    [
        "region_id",
        "measurements",
        "baseline_mean_recall_at_1",
        "consistency_mean_recall_at_1",
        "recall_improvement_pp",
        "mrr_improvement",
    ],
)

save_csv_new(
    SEED_OUTPUT,
    seed_rows,
    [
        "seed",
        "measurements",
        "baseline_mean_recall_at_1",
        "consistency_mean_recall_at_1",
        "recall_improvement_pp",
        "mrr_improvement",
    ],
)

report = {
    "phase": "4.2",
    "step": "4.2.17",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "experiment": "paired_robustness_audit",
    "source_old_report_sha256": sha256(
        OLD_REPORT
    ),
    "source_new_report_sha256": sha256(
        NEW_REPORT
    ),
    "graph_archive_sha256":
        old_report["graph_archive_sha256"],
    "protocol_sha256":
        old_report["protocol_sha256"],
    "paired_measurements": len(paired_rows),
    "training_seeds": list(SEEDS),
    "topology_repetitions": 5,
    "validation_regions": len(regions),
    "baseline_discordant_recall_at_1":
        overall_old,
    "consistency_discordant_recall_at_1":
        overall_new,
    "paired_recall_improvement_pp":
        overall_difference,
    "paired_mrr_improvement":
        overall_mrr_difference,
    "regions_improved": positive_regions,
    "regions_worsened": negative_regions,
    "regions_unchanged": unchanged_regions,
    "bootstrap": {
        "method":
            "percentile_region_cluster_bootstrap",
        "resampled_unit": "validation_region",
        "repetitions": BOOTSTRAP_REPETITIONS,
        "seed": BOOTSTRAP_SEED,
        "confidence_level": 0.95,
        "lower_pp": float(lower),
        "upper_pp": float(upper),
        "interpretation":
            "Descriptive uncertainty among the "
            "10 spatial validation regions only",
    },
    "seed_results": seed_rows,
    "region_results": region_rows,
    "paired_table_sha256": sha256(
        PAIRED_OUTPUT
    ),
    "region_table_sha256": sha256(
        REGION_OUTPUT
    ),
    "seed_table_sha256": sha256(
        SEED_OUTPUT
    ),
    "training_performed": False,
    "checkpoint_selection_performed": False,
    "test_evaluated": False,
    "limitations": [
        "Repeated perturbations are correlated",
        "Training seeds share the same tissue regions",
        "Bootstrap does not represent new patients",
        "Validation data has been repeatedly analyzed",
        "No independent biological generalization",
        "Synthetic topology perturbations only",
    ],
    "xai_planned": True,
}

save_json_new(
    REPORT_OUTPUT,
    report
)

print("\n" + "=" * 70)
print("PHASE 4.2.17 - PAIRED AUDIT COMPLETED")
print("=" * 70)

print(
    "Baseline discordant R@1:",
    f"{overall_old:.4f}"
)
print(
    "Consistency discordant R@1:",
    f"{overall_new:.4f}"
)
print(
    "Paired improvement:",
    f"{overall_difference:+.2f} pp"
)
print(
    "Descriptive bootstrap interval:",
    f"[{lower:+.2f}, {upper:+.2f}] pp"
)
print(
    "Improved / worsened / unchanged regions:",
    positive_regions,
    negative_regions,
    unchanged_regions,
)

print("\nSaved:")
print(PAIRED_OUTPUT)
print(REGION_OUTPUT)
print(SEED_OUTPUT)
print(REPORT_OUTPUT)

print("\nTraining: NO")
print("Checkpoints modified: NO")
print("Test evaluated: NO")
print("XAI: PLANNED")
print("\nSTEP 4.2.17 COMPLETED")
