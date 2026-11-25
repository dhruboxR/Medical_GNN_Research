
"""
PHASE 4.2.21 - DEGREE-CONTROLLED XAI AUDIT

Input:
    Phase 4.2.20 node importance and summary CSVs.

Controls:
    Permute GT450 node attribution within equal-degree
    groups, keeping AT2 attribution fixed.

Measures:
    Observed and degree-controlled Spearman agreement
    Observed and degree-controlled top-5 overlap
    Excess agreement above degree-controlled null
    Paired differences between lambda=0 and lambda=1
    Descriptive region-cluster bootstrap intervals

No training or model inference.
No test evaluation.
No existing files overwritten.
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

DETAIL = (
    BASE / "outputs/xai/35_structural_node_importance.csv"
)

OLD_SUMMARY = (
    BASE / "outputs/xai/35_structural_explanation_summary.csv"
)

SOURCE_REPORT = (
    BASE / "outputs/reports/35_structural_xai_report.json"
)

NULL_TABLE = (
    BASE / "outputs/xai/36_degree_controlled_xai.csv"
)

REGION_TABLE = (
    BASE / "outputs/xai/36_paired_xai_by_region.csv"
)

REPORT = (
    BASE / "outputs/reports/36_degree_controlled_xai_report.json"
)

MODELS = (
    "ResidualGate05",
    "TopologyConsistentResidual",
)

SEEDS = (42, 123, 2026)
TOP_K = 5

NULL_REPETITIONS = 1000
BOOTSTRAP_REPETITIONS = 5000
RANDOM_SEED = 42121


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
        b""):
            h.update(chunk)

    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save_csv_new(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(rows)


def save_json_new(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as stream:
        json.dump(content, stream, indent=2)


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)

    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)

    sorted_values = values[order]
    start = 0

    while start < len(values):
        end = start + 1

        while (
            end < len(values)
            and sorted_values[end] == sorted_values[start]
        ):
            end += 1

        rank = (start + 1 + end) / 2.0
        result[order[start:end]] = rank
        start = end

    return result


def spearman(x, y):
    a = average_ranks(x)
    b = average_ranks(y)

    a = a - a.mean()
    b = b - b.mean()

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator <= 1e-12:
        return np.nan

    return float(np.dot(a, b) / denominator)


def top_k_indices(values):
    return set(
        np.argsort(
            -np.asarray(values, dtype=np.float64),
            kind="stable",
        )[:TOP_K].tolist()
    )


def overlap(x, y):
    return len(top_k_indices(x) & top_k_indices(y))


def summarize(values):
    values = np.asarray(values, dtype=np.float64)

    if not np.isfinite(values).all():
        raise RuntimeError("Non-finite summary input.")

    return float(values.mean())


print("=" * 70)
print("PHASE 4.2.21 - DEGREE-CONTROLLED XAI AUDIT")
print("=" * 70)


# --------------------------------------------------
# 1. Verify prior XAI outputs
# --------------------------------------------------

print("\n[1] Verifying source explanations...")

for path in (DETAIL, OLD_SUMMARY, SOURCE_REPORT):
    if not path.is_file():
        raise RuntimeError(f"Missing source: {path}")

for path in (NULL_TABLE, REGION_TABLE, REPORT):
    if path.exists():
        raise RuntimeError(f"Refusing overwrite: {path}")

source = read_json(SOURCE_REPORT)

if source["step"] != "4.2.20":
    raise RuntimeError("Unexpected source experiment.")

if source["test_evaluated"] is not False:
    raise RuntimeError("Source test protocol violation.")

if source["detail_table_sha256"] != sha256(DETAIL):
    raise RuntimeError("Importance CSV hash mismatch.")

if source["summary_table_sha256"] != sha256(OLD_SUMMARY):
    raise RuntimeError("Summary CSV hash mismatch.")

detail = read_csv(DETAIL)
previous = read_csv(OLD_SUMMARY)

if len(detail) != 4320:
    raise RuntimeError(
        "Expected 2 models x 3 seeds x 10 regions "
        "x 2 scanners x 36 nodes."
    )

if len(previous) != 60:
    raise RuntimeError(
        "Expected 60 model-seed-region summary rows."
    )


# --------------------------------------------------
# 2. Reconstruct complete explanation pairs
# --------------------------------------------------

print("\n[2] Reconstructing paired scanner attributions...")

grouped = defaultdict(dict)
degree_by_node = {}

for row in detail:
    model = row["model"]
    seed = int(row["seed"])
    region = row["region_id"]
    scanner = row["scanner"]
    node = int(row["node_index"])
    degree = int(row["node_degree"])
    importance = float(row["importance"])

    if model not in MODELS or seed not in SEEDS:
        raise RuntimeError("Unexpected model or seed.")

    if scanner not in ("AT2", "GT450"):
        raise RuntimeError("Unexpected scanner.")

    if not 0 <= node < 36 or degree not in (2, 3, 4):
        raise RuntimeError("Invalid graph node or degree.")

    if not np.isfinite(importance):
        raise RuntimeError("Non-finite attribution.")

    if node in degree_by_node:
        if degree_by_node[node] != degree:
            raise RuntimeError("Node degree changed.")
    else:
        degree_by_node[node] = degree

    key = (model, seed, region, scanner)

    if node in grouped[key]:
        raise RuntimeError("Duplicate explanation node.")

    grouped[key][node] = importance

if set(degree_by_node) != set(range(36)):
    raise RuntimeError("Incomplete node degree definition.")

degrees = np.array(
    [degree_by_node[i] for i in range(36)]
)

degree_groups = {
    degree: np.where(degrees == degree)[0]
    for degree in sorted(set(degrees))
}

print(
    "Degree groups:",
    {k: len(v) for k, v in degree_groups.items()}
)

# The 6x6 four-neighbor grid has
# 4 corners, 16 non-corner boundary nodes,
# and 16 interior nodes.
if {
    k: len(v) for k, v in degree_groups.items()
} != {2: 4, 3: 16, 4: 16}:
    raise RuntimeError("Unexpected spatial grid degrees.")

regions = sorted({
    key[2] for key in grouped
})

if len(regions) != 10:
    raise RuntimeError("Expected 10 validation regions.")

expected_keys = {
    (model, seed, region, scanner)
    for model in MODELS
    for seed in SEEDS
    for region in regions
    for scanner in ("AT2", "GT450")
}

if set(grouped) != expected_keys:
    raise RuntimeError("Incomplete XAI design.")

previous_index = {}

for row in previous:
    key = (
        row["model"],
        int(row["seed"]),
        row["region_id"],
    )

    if key in previous_index:
        raise RuntimeError("Duplicate XAI summary row.")

    previous_index[key] = row

expected_summaries = {
    (model, seed, region)
    for model in MODELS
    for seed in SEEDS
    for region in regions
}

if set(previous_index) != expected_summaries:
    raise RuntimeError("Incomplete prior XAI summaries.")


# --------------------------------------------------
# 3. Degree-aware permutation null distributions
# --------------------------------------------------

print("\n[3] Computing degree-controlled null distributions...")

rng = np.random.default_rng(RANDOM_SEED)

null_rows = []

for model in MODELS:
    for seed in SEEDS:
        for region in regions:
            a = np.array([
                grouped[(model, seed, region, "AT2")][i]
                for i in range(36)
            ])

            b = np.array([
                grouped[(model, seed, region, "GT450")][i]
                for i in range(36)
            ])

            observed_rho = spearman(a, b)
            observed_overlap = overlap(a, b)

            if not np.isfinite(observed_rho):
                raise RuntimeError(
                    "Undefined attribution correlation."
                )

            old = previous_index[(model, seed, region)]

            if abs(
                observed_rho
                - float(old["attribution_rank_correlation"])
            ) > 1e-8:
                raise RuntimeError(
                    "Existing rank correlation not reproduced."
                )

            if observed_overlap != int(old["top5_overlap"]):
                raise RuntimeError(
                    "Existing top-5 overlap not reproduced."
                )

            null_rho = np.empty(
                NULL_REPETITIONS, dtype=np.float64
            )

            null_overlap = np.empty(
                NULL_REPETITIONS, dtype=np.float64
            )

            for repeat in range(NULL_REPETITIONS):
                permuted_b = b.copy()

                for indices in degree_groups.values():
                    permuted_b[indices] = b[
                        rng.permutation(indices)
                    ]

                null_rho[repeat] = spearman(
                    a, permuted_b
                )

                null_overlap[repeat] = overlap(
                    a, permuted_b
                )

            if not np.isfinite(null_rho).all():
                raise RuntimeError(
                    "Undefined null attribution correlation."
                )

            mean_null_rho = summarize(null_rho)
            mean_null_overlap = summarize(null_overlap)

            # Monte Carlo tail probabilities are
            # descriptive, not independent-patient
            # inferential p-values.
            rho_tail = (
                1 + np.sum(null_rho >= observed_rho)
            ) / (NULL_REPETITIONS + 1)

            overlap_tail = (
                1 + np.sum(
                    null_overlap >= observed_overlap
                )
            ) / (NULL_REPETITIONS + 1)

            null_rows.append({
                "model": model,
                "seed": seed,
                "region_id": region,
                "observed_spearman": observed_rho,
                "null_mean_spearman": mean_null_rho,
                "excess_spearman":
                    observed_rho - mean_null_rho,
                "spearman_null_tail_fraction":
                    float(rho_tail),
                "observed_top5_overlap":
                    observed_overlap,
                "null_mean_top5_overlap":
                    mean_null_overlap,
                "excess_top5_overlap":
                    observed_overlap - mean_null_overlap,
                "overlap_null_tail_fraction":
                    float(overlap_tail),
            })

    subset = [
        row for row in null_rows
        if row["model"] == model
    ]

    print(
        f"  {model} | "
        f"Observed rho "
        f"{summarize([r['observed_spearman'] for r in subset]):.4f} | "
        f"Null rho "
        f"{summarize([r['null_mean_spearman'] for r in subset]):.4f} | "
        f"Excess rho "
        f"{summarize([r['excess_spearman'] for r in subset]):+.4f}"
    )


# --------------------------------------------------
# 4. Paired model comparison by spatial region
# --------------------------------------------------

print("\n[4] Comparing models by validation region...")

index = {
    (row["model"], row["seed"], row["region_id"]): row
    for row in null_rows
}

region_rows = []

metrics = (
    "observed_spearman",
    "excess_spearman",
    "observed_top5_overlap",
    "excess_top5_overlap",
)

for region in regions:
    record = {"region_id": region}

    for metric in metrics:
        baseline_values = []
        consistency_values = []

        for seed in SEEDS:
            baseline_values.append(
                index[
                    ("ResidualGate05", seed, region)
                ][metric]
            )

            consistency_values.append(
                index[
                    ("TopologyConsistentResidual", seed, region)
                ][metric]
            )

        baseline_mean = summarize(baseline_values)
        consistency_mean = summarize(consistency_values)

        record[f"baseline_{metric}"] = baseline_mean
        record[f"consistency_{metric}"] = consistency_mean
        record[f"difference_{metric}"] = (
            consistency_mean - baseline_mean
        )

    region_rows.append(record)


# --------------------------------------------------
# 5. Descriptive region-cluster bootstrap
# --------------------------------------------------

print("\n[5] Computing descriptive uncertainty intervals...")

bootstrap_rng = np.random.default_rng(
    RANDOM_SEED + 1
)

bootstrap_results = {}

for metric in metrics:
    differences = np.asarray(
        [
            row[f"difference_{metric}"]
            for row in region_rows
        ],
        dtype=np.float64,
    )

    samples = np.empty(
        BOOTSTRAP_REPETITIONS,
        dtype=np.float64,
    )

    for iteration in range(BOOTSTRAP_REPETITIONS):
        selected = bootstrap_rng.integers(
            0, len(differences),
            size=len(differences),
        )

        samples[iteration] = differences[
            selected
        ].mean()

    lower, upper = np.quantile(
        samples, [0.025, 0.975]
    )

    bootstrap_results[metric] = {
        "mean_paired_difference":
            float(differences.mean()),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "regions_with_positive_difference":
            int(np.sum(differences > 1e-10)),
        "regions_with_negative_difference":
            int(np.sum(differences < -1e-10)),
    }

    print(
        f"  {metric} | "
        f"Difference {differences.mean():+.4f} | "
        f"Descriptive interval "
        f"[{lower:+.4f}, {upper:+.4f}]"
    )


# --------------------------------------------------
# 6. Save results and provenance
# --------------------------------------------------

print("\n[6] Saving degree-aware XAI audit...")

null_fields = [
    "model", "seed", "region_id",
    "observed_spearman",
    "null_mean_spearman",
    "excess_spearman",
    "spearman_null_tail_fraction",
    "observed_top5_overlap",
    "null_mean_top5_overlap",
    "excess_top5_overlap",
    "overlap_null_tail_fraction",
]

save_csv_new(
    NULL_TABLE,
    null_rows,
    null_fields,
)

region_fields = ["region_id"]

for metric in metrics:
    region_fields += [
        f"baseline_{metric}",
        f"consistency_{metric}",
        f"difference_{metric}",
    ]

save_csv_new(
    REGION_TABLE,
    region_rows,
    region_fields,
)

model_summaries = {}

for model in MODELS:
    subset = [
        row for row in null_rows
        if row["model"] == model
    ]

    model_summaries[model] = {
        metric: summarize(
            [row[metric] for row in subset]
        )
        for metric in (
            "observed_spearman",
            "null_mean_spearman",
            "excess_spearman",
            "observed_top5_overlap",
            "null_mean_top5_overlap",
            "excess_top5_overlap",
        )
    }

report = {
    "phase": "4.2",
    "step": "4.2.21",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "experiment": "degree_controlled_structural_xai",
    "source_report_sha256":
        sha256(SOURCE_REPORT),
    "source_detail_sha256":
        sha256(DETAIL),
    "source_summary_sha256":
        sha256(OLD_SUMMARY),
    "graph_archive_sha256":
        source["graph_archive_sha256"],
    "protocol_sha256":
        source["protocol_sha256"],
    "models": list(MODELS),
    "seeds": list(SEEDS),
    "validation_regions": len(regions),
    "nodes_per_graph": 36,
    "top_k": TOP_K,
    "degree_groups": {
        str(k): len(v)
        for k, v in degree_groups.items()
    },
    "null_method":
        "permute_GT450_importance_within_equal_degree_groups",
    "null_repetitions": NULL_REPETITIONS,
    "model_summary": model_summaries,
    "paired_region_bootstrap": {
        "unit": "spatial_region",
        "repetitions": BOOTSTRAP_REPETITIONS,
        "results": bootstrap_results,
    },
    "null_table_sha256":
        sha256(NULL_TABLE),
    "region_table_sha256":
        sha256(REGION_TABLE),
    "training_performed": False,
    "model_inference_performed": False,
    "checkpoint_selection_performed": False,
    "test_evaluated": False,
    "limitations": [
        "Controls node degree, not all spatial confounders",
        "Null permutations are conditional on observed scores",
        "Same validation regions have been repeatedly examined",
        "Bootstrap does not represent independent patients",
        "Explanation scores use a retrieval training objective",
        "No biologically validated relevance labels",
        "No degree-matched deletion fidelity control yet",
    ],
}

save_json_new(REPORT, report)

print("\n" + "=" * 70)
print("PHASE 4.2.21 - DEGREE-CONTROLLED XAI COMPLETED")
print("=" * 70)

for model, result in model_summaries.items():
    print(
        f"{model} | "
        f"Observed rho {result['observed_spearman']:.4f} | "
        f"Degree-null rho {result['null_mean_spearman']:.4f} | "
        f"Excess rho {result['excess_spearman']:+.4f} | "
        f"Excess top-5 overlap "
        f"{result['excess_top5_overlap']:+.3f}"
    )

print("\nSaved:")
print(NULL_TABLE)
print(REGION_TABLE)
print(REPORT)

print("\nTraining: NO")
print("Model inference: NO")
print("Existing checkpoints: UNCHANGED")
print("Test partition: NOT EVALUATED")
print("XAI: IN PROGRESS")
print("\nSTEP 4.2.21 COMPLETED")
