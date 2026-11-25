
"""
PHASE 4.2.23 - INTEGRATED GRADIENTS RELIABILITY AUDIT

Uses Phase 4.2.22 saved outputs:
    37_ig_node_attributions.csv
    37_ig_feature_channels.csv
    37_ig_explanation_summary.csv

Evaluates:
    IG completeness, absolute and relative errors
    Degree-matched deletion fidelity
    Degree-controlled node attribution agreement
    Channel attribution agreement reproduction
    Paired region-level model differences
    Descriptive 95% region bootstrap intervals

No training or model inference.
No test evaluation or checkpoint selection.
Historical artifacts preserved.
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

NODE_SOURCE = BASE / "outputs/xai/37_ig_node_attributions.csv"
CHANNEL_SOURCE = BASE / "outputs/xai/37_ig_feature_channels.csv"
SUMMARY_SOURCE = BASE / "outputs/xai/37_ig_explanation_summary.csv"
SOURCE_REPORT = BASE / "outputs/reports/37_integrated_gradients_report.json"

DETAIL_OUT = BASE / "outputs/xai/38_ig_reliability_detail.csv"
REGION_OUT = BASE / "outputs/xai/38_ig_paired_region_comparison.csv"
REPORT_OUT = BASE / "outputs/reports/38_ig_reliability_report.json"

MODELS = ("ResidualGate05", "TopologyConsistentResidual")
SEEDS = (42, 123, 2026)
SCANNERS = ("AT2", "GT450")

TOP_K = 5
NULL_REPETITIONS = 1000
BOOTSTRAP_REPETITIONS = 5000


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def read_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv_new(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_json_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def stable_rng(*items):
    key = "|".join(map(str, items)).encode("utf-8")
    h = hashlib.sha256(key).digest()
    return np.random.default_rng(
        int.from_bytes(h[:8], "little")
    )


def average_ranks(values):
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranked = np.empty(len(x), dtype=np.float64)

    sorted_x = x[order]
    start = 0

    while start < len(x):
        end = start + 1

        while (
            end < len(x)
            and sorted_x[end] == sorted_x[start]
        ):
            end += 1

        ranked[order[start:end]] = (
            start + 1 + end
        ) / 2.0
        start = end

    return ranked


def spearman(a, b):
    x = average_ranks(a)
    y = average_ranks(b)

    x -= x.mean()
    y -= y.mean()

    norm = np.linalg.norm(x) * np.linalg.norm(y)

    if norm <= 1e-12:
        raise RuntimeError("Undefined rank correlation.")

    return float(np.dot(x, y) / norm)


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom <= 1e-12:
        raise RuntimeError("Undefined channel cosine.")

    return float(np.dot(a, b) / denom)


def top5_overlap(a, b):
    left = set(
        np.argsort(-a, kind="stable")[:TOP_K]
    )
    right = set(
        np.argsort(-b, kind="stable")[:TOP_K]
    )

    return len(left & right)


def mean(rows, key):
    x = np.asarray(
        [float(r[key]) for r in rows],
        dtype=np.float64,
    )

    if not np.isfinite(x).all():
        raise RuntimeError(f"Invalid metric: {key}")

    return float(x.mean())


print("=" * 70)
print("PHASE 4.2.23 - INTEGRATED GRADIENTS RELIABILITY")
print("=" * 70)


# ---------------------------------------------------
# 1. Verify provenance
# ---------------------------------------------------

print("\n[1] Verifying saved Integrated Gradients outputs...")

for path in (
    NODE_SOURCE,
    CHANNEL_SOURCE,
    SUMMARY_SOURCE,
    SOURCE_REPORT,
):
    if not path.is_file():
        raise RuntimeError(f"Missing source: {path}")

for path in (DETAIL_OUT, REGION_OUT, REPORT_OUT):
    if path.exists():
        raise RuntimeError(f"Refusing overwrite: {path}")

source = read_json(SOURCE_REPORT)

if (
    source["step"] != "4.2.22"
    or source["test_evaluated"] is not False
    or source["training_performed"] is not False
):
    raise RuntimeError("Unexpected source protocol.")

checks = {
    "node_table_sha256": NODE_SOURCE,
    "channel_table_sha256": CHANNEL_SOURCE,
    "summary_table_sha256": SUMMARY_SOURCE,
}

for field, path in checks.items():
    if source[field] != sha256(path):
        raise RuntimeError(f"Source hash mismatch: {path}")

nodes = read_csv(NODE_SOURCE)
channels = read_csv(CHANNEL_SOURCE)
summaries = read_csv(SUMMARY_SOURCE)

if len(nodes) != 2 * 3 * 10 * 2 * 36:
    raise RuntimeError("Incomplete node attribution table.")

if len(channels) != 2 * 3 * 10 * 2 * 384:
    raise RuntimeError("Incomplete feature channel table.")

if len(summaries) != 2 * 3 * 10:
    raise RuntimeError("Incomplete XAI summaries.")


# ---------------------------------------------------
# 2. Reconstruct node and channel profiles
# ---------------------------------------------------

print("\n[2] Reconstructing matched scanner profiles...")

node_data = defaultdict(dict)
channel_data = defaultdict(dict)
degree_map = {}

for row in nodes:
    model = row["model"]
    seed = int(row["seed"])
    region = row["region_id"]
    scanner = row["scanner"]

    index = int(row["node_index"])
    degree = int(row["node_degree"])
    importance = float(row["ig_absolute_importance"])

    if (
        model not in MODELS
        or seed not in SEEDS
        or scanner not in SCANNERS
        or not 0 <= index < 36
        or degree not in (2, 3, 4)
        or not np.isfinite(importance)
        or importance < 0
    ):
        raise RuntimeError("Invalid node attribution row.")

    if index in degree_map and degree_map[index] != degree:
        raise RuntimeError("Node degree mismatch.")

    degree_map[index] = degree

    key = (model, seed, region, scanner)

    if index in node_data[key]:
        raise RuntimeError("Duplicate node attribution.")

    node_data[key][index] = importance

for row in channels:
    model = row["model"]
    seed = int(row["seed"])
    region = row["region_id"]
    scanner = row["scanner"]

    index = int(row["feature_channel"])
    importance = float(row["ig_absolute_importance"])

    if (
        model not in MODELS
        or seed not in SEEDS
        or scanner not in SCANNERS
        or not 0 <= index < 384
        or not np.isfinite(importance)
        or importance < 0
    ):
        raise RuntimeError("Invalid channel attribution row.")

    key = (model, seed, region, scanner)

    if index in channel_data[key]:
        raise RuntimeError("Duplicate channel attribution.")

    channel_data[key][index] = importance

regions = sorted({
    row["region_id"] for row in summaries
})

if len(regions) != 10:
    raise RuntimeError("Expected ten validation regions.")

expected_profile_keys = {
    (model, seed, region, scanner)
    for model in MODELS
    for seed in SEEDS
    for region in regions
    for scanner in SCANNERS
}

if (
    set(node_data) != expected_profile_keys
    or set(channel_data) != expected_profile_keys
):
    raise RuntimeError("Missing attribution profiles.")

if set(degree_map) != set(range(36)):
    raise RuntimeError("Incomplete node degree map.")

degree = np.array(
    [degree_map[i] for i in range(36)]
)

degree_groups = {
    d: np.where(degree == d)[0]
    for d in (2, 3, 4)
}

if {
    d: len(indices)
    for d, indices in degree_groups.items()
} != {2: 4, 3: 16, 4: 16}:
    raise RuntimeError("Unexpected 6x6 grid degrees.")

node_vectors = {}
channel_vectors = {}

for key in expected_profile_keys:
    if set(node_data[key]) != set(range(36)):
        raise RuntimeError("Incomplete 36-node vector.")

    if set(channel_data[key]) != set(range(384)):
        raise RuntimeError("Incomplete 384-channel vector.")

    node_vectors[key] = np.array([
        node_data[key][i] for i in range(36)
    ])

    channel_vectors[key] = np.array([
        channel_data[key][i] for i in range(384)
    ])

summary_index = {}

for row in summaries:
    key = (
        row["model"],
        int(row["seed"]),
        row["region_id"],
    )

    if key in summary_index:
        raise RuntimeError("Duplicate summary.")

    summary_index[key] = row

expected_summary_keys = {
    (model, seed, region)
    for model in MODELS
    for seed in SEEDS
    for region in regions
}

if set(summary_index) != expected_summary_keys:
    raise RuntimeError("Incomplete summary design.")

print("Validated model-seed-region units:", len(summary_index))
print("Degree groups:", {
    int(d): len(indices)
    for d, indices in degree_groups.items()
})


# ---------------------------------------------------
# 3. Degree-stratified explanation nulls
# ---------------------------------------------------

print("\n[3] Checking explanation agreement against degree nulls...")

detail = []

for model in MODELS:
    for seed in SEEDS:
        for region in regions:
            key = (model, seed, region)
            saved = summary_index[key]

            a = node_vectors[key + ("AT2",)]
            b = node_vectors[key + ("GT450",)]

            ca = channel_vectors[key + ("AT2",)]
            cb = channel_vectors[key + ("GT450",)]

            observed_rho = spearman(a, b)
            observed_overlap = top5_overlap(a, b)
            observed_channel_cos = cosine(ca, cb)

            if abs(
                observed_rho - float(saved["node_spearman"])
            ) > 1e-8:
                raise RuntimeError("IG Spearman reproduction failed.")

            if observed_overlap != int(
                saved["node_top5_overlap"]
            ):
                raise RuntimeError("IG top-5 reproduction failed.")

            if abs(
                observed_channel_cos
                - float(saved["channel_importance_cosine"])
            ) > 1e-8:
                raise RuntimeError("Channel cosine reproduction failed.")

            null_rho = []
            null_overlap = []

            rng = stable_rng("phase423", seed, region)

            for _ in range(NULL_REPETITIONS):
                shuffled = b.copy()

                for indices in degree_groups.values():
                    shuffled[indices] = b[
                        rng.permutation(indices)
                    ]

                null_rho.append(spearman(a, shuffled))
                null_overlap.append(
                    top5_overlap(a, shuffled)
                )

            null_rho = np.asarray(null_rho)
            null_overlap = np.asarray(null_overlap)

            null_mean_rho = float(null_rho.mean())
            null_mean_overlap = float(null_overlap.mean())

            detail.append({
                "model": model,
                "seed": seed,
                "region_id": region,
                "observed_node_spearman":
                    observed_rho,
                "degree_null_node_spearman":
                    null_mean_rho,
                "excess_node_spearman":
                    observed_rho - null_mean_rho,
                "observed_top5_overlap":
                    observed_overlap,
                "degree_null_top5_overlap":
                    null_mean_overlap,
                "excess_top5_overlap":
                    observed_overlap - null_mean_overlap,
                "channel_importance_cosine":
                    observed_channel_cos,
                "at2_completeness_abs_error":
                    float(saved["at2_completeness_abs_error"]),
                "gt450_completeness_abs_error":
                    float(saved["gt450_completeness_abs_error"]),
                "at2_completeness_relative_error":
                    float(saved["at2_completeness_relative_error"]),
                "gt450_completeness_relative_error":
                    float(saved["gt450_completeness_relative_error"]),
                "at2_fidelity_excess":
                    float(saved["at2_fidelity_excess"]),
                "gt450_fidelity_excess":
                    float(saved["gt450_fidelity_excess"]),
            })

print("Attribution agreement reproduction: VERIFIED")


# ---------------------------------------------------
# 4. Aggregate fidelity and completeness
# ---------------------------------------------------

print("\n[4] Auditing completeness and deletion fidelity...")

metrics = (
    "observed_node_spearman",
    "excess_node_spearman",
    "observed_top5_overlap",
    "excess_top5_overlap",
    "channel_importance_cosine",
    "at2_fidelity_excess",
    "gt450_fidelity_excess",
    "at2_completeness_abs_error",
    "gt450_completeness_abs_error",
    "at2_completeness_relative_error",
    "gt450_completeness_relative_error",
)

model_summary = {}

for model in MODELS:
    subset = [r for r in detail if r["model"] == model]

    model_summary[model] = {
        metric: mean(subset, metric)
        for metric in metrics
    }

    fidelity_at2 = model_summary[model][
        "at2_fidelity_excess"
    ]

    fidelity_gt = model_summary[model][
        "gt450_fidelity_excess"
    ]

    if not all(
        np.isfinite(float(value))
        for value in model_summary[model].values()
    ):
        raise RuntimeError("Non-finite XAI metric.")

    print(
        f"  {model} | "
        f"Excess node rho "
        f"{model_summary[model]['excess_node_spearman']:+.4f} | "
        f"Excess top-5 "
        f"{model_summary[model]['excess_top5_overlap']:+.3f}"
    )

    print(
        f"    AT2 fidelity excess {fidelity_at2:+.4f} | "
        f"GT450 fidelity excess {fidelity_gt:+.4f}"
    )

    print(
        f"    Mean absolute completeness error: "
        f"AT2 {model_summary[model]['at2_completeness_abs_error']:.6f}, "
        f"GT450 {model_summary[model]['gt450_completeness_abs_error']:.6f}"
    )


# ---------------------------------------------------
# 5. Paired region-cluster uncertainty
# ---------------------------------------------------

print("\n[5] Computing paired model comparisons...")

paired_metrics = (
    "observed_node_spearman",
    "excess_node_spearman",
    "observed_top5_overlap",
    "excess_top5_overlap",
    "channel_importance_cosine",
    "at2_fidelity_excess",
    "gt450_fidelity_excess",
)

detail_index = {
    (r["model"], r["seed"], r["region_id"]): r
    for r in detail
}

region_rows = []
region_differences = defaultdict(list)

for region in regions:
    result = {"region_id": region}

    for metric in paired_metrics:
        old = np.mean([
            detail_index[
                ("ResidualGate05", seed, region)
            ][metric]
            for seed in SEEDS
        ])

        new = np.mean([
            detail_index[
                ("TopologyConsistentResidual", seed, region)
            ][metric]
            for seed in SEEDS
        ])

        difference = float(new - old)

        result[f"difference_{metric}"] = difference
        region_differences[metric].append(difference)

    region_rows.append(result)

bootstrap_summary = {}

for metric in paired_metrics:
    differences = np.asarray(
        region_differences[metric],
        dtype=np.float64,
    )

    rng = stable_rng("phase423_bootstrap", metric)

    draws = rng.integers(
        0, len(regions),
        size=(BOOTSTRAP_REPETITIONS, len(regions)),
    )

    samples = differences[draws].mean(axis=1)

    lower, upper = np.quantile(
        samples, [0.025, 0.975]
    )

    bootstrap_summary[metric] = {
        "mean_difference": float(differences.mean()),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "regions_positive": int((differences > 0).sum()),
        "regions_negative": int((differences < 0).sum()),
    }

    print(
        f"  {metric} | "
        f"Difference {differences.mean():+.4f} | "
        f"Interval [{lower:+.4f}, {upper:+.4f}]"
    )


# ---------------------------------------------------
# 6. Save new artifacts
# ---------------------------------------------------

print("\n[6] Saving reliability audit...")

detail_fields = [
    "model", "seed", "region_id",
    *metrics,
    "degree_null_node_spearman",
    "degree_null_top5_overlap",
]

# Preserve the exact field order without duplicates.
detail_fields = list(dict.fromkeys(detail_fields))

save_csv_new(
    DETAIL_OUT,
    detail,
    detail_fields,
)

region_fields = (
    ["region_id"]
    + [
        f"difference_{metric}"
        for metric in paired_metrics
    ]
)

save_csv_new(
    REGION_OUT,
    region_rows,
    region_fields,
)

report = {
    "phase": "4.2",
    "step": "4.2.23",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "experiment": "integrated_gradients_reliability_audit",
    "source_ig_report_sha256": sha256(SOURCE_REPORT),
    "source_node_table_sha256": sha256(NODE_SOURCE),
    "source_channel_table_sha256": sha256(CHANNEL_SOURCE),
    "source_summary_table_sha256": sha256(SUMMARY_SOURCE),
    "graph_archive_sha256":
        source["graph_archive_sha256"],
    "protocol_sha256": source["protocol_sha256"],
    "models": list(MODELS),
    "seeds": list(SEEDS),
    "validation_regions": len(regions),
    "degree_null_repetitions": NULL_REPETITIONS,
    "degree_null_method":
        "GT450_attribution_permuted_within_node_degree",
    "model_summary": model_summary,
    "paired_region_bootstrap": {
        "sampling_unit": "validation_region",
        "repetitions": BOOTSTRAP_REPETITIONS,
        "results": bootstrap_summary,
    },
    "detail_table_sha256": sha256(DETAIL_OUT),
    "region_table_sha256": sha256(REGION_OUT),
    "training_performed": False,
    "model_inference_performed": False,
    "checkpoint_selection_performed": False,
    "test_evaluated": False,
    "limitations": [
        "IG uses the zero-feature baseline",
        "Completeness is checked for 32-step approximation",
        "Fidelity deletion changes input feature distribution",
        "Random deletion controls matched degree only",
        "High absolute channel cosine may conceal signed differences",
        "Bootstrap regions share one biological source context",
        "Same validation data have been repeatedly analyzed",
        "Attributions lack biological ground-truth annotations",
    ],
}

save_json_new(REPORT_OUT, report)

print("\n" + "=" * 70)
print("PHASE 4.2.23 - IG RELIABILITY AUDIT COMPLETED")
print("=" * 70)

for model in MODELS:
    m = model_summary[model]

    print(
        f"{model} | "
        f"Node rho {m['observed_node_spearman']:.4f} | "
        f"Excess rho {m['excess_node_spearman']:+.4f} | "
        f"Top-5 overlap {m['observed_top5_overlap']:.2f}"
    )

    print(
        f"  AT2 fidelity excess "
        f"{m['at2_fidelity_excess']:+.4f} | "
        f"GT450 fidelity excess "
        f"{m['gt450_fidelity_excess']:+.4f}"
    )

print("\nSaved:")
print(DETAIL_OUT)
print(REGION_OUT)
print(REPORT_OUT)

print("\nTraining: NO")
print("Model inference: NO")
print("Checkpoints modified: NO")
print("Held-out test evaluated: NO")
print("XAI: IN PROGRESS")
print("\nSTEP 4.2.23 COMPLETED")
