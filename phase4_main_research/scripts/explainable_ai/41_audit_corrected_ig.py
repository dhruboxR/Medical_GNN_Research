
"""
PHASE 4.2.26 - CORRECTED IG PAIRED RELIABILITY AUDIT

Sources:
  Phase 4.2.25 corrected IG node importance
  Phase 4.2.25 corrected IG detail
  Phase 4.2.25 corrected IG summary/report

Measures:
  - Reproduced scanner attribution correlations
  - Degree-stratified null correlations
  - Signed and absolute attribution agreement
  - Paired model differences
  - Top-5 deletion fidelity excess
  - Positive-fidelity region counts
  - Descriptive region-cluster bootstrap intervals

No model inference.
No training.
No checkpoint selection.
No test evaluation.
"""

from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import csv

import numpy as np


BASE = (
    Path.home()
    / "MEDICAL_GNN_RESEARCH"
    / "phase4_main_research"
)

SOURCE_NODES = (
    BASE / "outputs/xai/40_ig_node_importance.csv"
)
SOURCE_DETAIL = (
    BASE / "outputs/xai/40_ig_corrected_detail.csv"
)
SOURCE_SUMMARY = (
    BASE / "outputs/xai/40_ig_corrected_summary.csv"
)
SOURCE_REPORT = (
    BASE / "outputs/reports/40_corrected_ig_report.json"
)

DETAIL_OUT = (
    BASE / "outputs/xai/41_corrected_ig_degree_audit.csv"
)
REGION_OUT = (
    BASE / "outputs/xai/41_corrected_ig_paired_regions.csv"
)
REPORT_OUT = (
    BASE / "outputs/reports/41_corrected_ig_audit_report.json"
)

MODELS = (
    "ResidualGate05",
    "TopologyConsistentResidual",
)

SEEDS = (42, 123, 2026)
SCANNERS = ("AT2", "GT450")

TOP_K = 5
NULL_REPS = 1000
BOOTSTRAP_REPS = 5000


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)

    return h.hexdigest()


def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_csv(path):
    with path.open(
        newline="", encoding="utf-8"
    ) as f:
        return list(csv.DictReader(f))


def save_csv(path, rows, fields):
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    with path.open(
        "x", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, data):
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    with path.open(
        "x", encoding="utf-8"
    ) as f:
        json.dump(data, f, indent=2)


def seeded_rng(*parts):
    value = "|".join(map(str, parts))
    digest = hashlib.sha256(
        value.encode("utf-8")
    ).digest()

    return np.random.default_rng(
        int.from_bytes(digest[:8], "little")
    )


def rankdata(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    result = np.empty(len(x), dtype=np.float64)

    sorted_x = x[order]
    start = 0

    while start < len(x):
        end = start + 1

        while (
            end < len(x)
            and sorted_x[end] == sorted_x[start]
        ):
            end += 1

        result[order[start:end]] = (
            start + 1 + end
        ) / 2.0

        start = end

    return result


def spearman(x, y):
    x = rankdata(x)
    y = rankdata(y)

    x -= x.mean()
    y -= y.mean()

    denom = (
        np.linalg.norm(x)
        * np.linalg.norm(y)
    )

    if denom <= 1e-12:
        return None

    return float(np.dot(x, y) / denom)


def top5_overlap(x, y):
    a = set(
        np.argsort(-x, kind="stable")[:TOP_K]
    )
    b = set(
        np.argsort(-y, kind="stable")[:TOP_K]
    )

    return len(a & b)


def mean_metric(rows, metric):
    values = np.array(
        [float(r[metric]) for r in rows],
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise RuntimeError(
            f"Non-finite metric: {metric}"
        )

    return float(values.mean())


print("=" * 70)
print("PHASE 4.2.26 - CORRECTED IG RELIABILITY")
print("=" * 70)


# -------------------------------------------------
# 1. Verify sources
# -------------------------------------------------

print("\n[1] Verifying corrected IG sources...")

for path in (
    SOURCE_NODES,
    SOURCE_DETAIL,
    SOURCE_SUMMARY,
    SOURCE_REPORT,
):
    if not path.is_file():
        raise RuntimeError(
            f"Missing source: {path}"
        )

for path in (
    DETAIL_OUT,
    REGION_OUT,
    REPORT_OUT,
):
    if path.exists():
        raise RuntimeError(
            f"Refusing overwrite: {path}"
        )

source = load_json(SOURCE_REPORT)

if (
    source["step"] != "4.2.25"
    or source["test_evaluated"] is not False
    or source["training_performed"] is not False
):
    raise RuntimeError(
        "Unexpected source protocol."
    )

checks = {
    "node_table_sha256": SOURCE_NODES,
    "detail_table_sha256": SOURCE_DETAIL,
    "summary_table_sha256": SOURCE_SUMMARY,
}

for field, path in checks.items():
    if source[field] != sha256(path):
        raise RuntimeError(
            f"Hash mismatch: {path}"
        )

nodes = load_csv(SOURCE_NODES)
details = load_csv(SOURCE_DETAIL)
summaries = load_csv(SOURCE_SUMMARY)

if len(nodes) != 4320:
    raise RuntimeError(
        "Expected 4320 node attributions."
    )

if len(details) != 60:
    raise RuntimeError(
        "Expected 60 model-seed-region records."
    )

if len(summaries) != 2:
    raise RuntimeError(
        "Expected two model summaries."
    )


# -------------------------------------------------
# 2. Reconstruct attribution vectors
# -------------------------------------------------

print("\n[2] Reconstructing feature-node explanations...")

vectors = defaultdict(dict)
degree_map = {}

for row in nodes:
    model = row["model"]
    seed = int(row["seed"])
    region = row["region_id"]
    scanner = row["scanner"]

    node = int(row["node_index"])
    degree = int(row["degree"])

    if (
        model not in MODELS
        or seed not in SEEDS
        or scanner not in SCANNERS
        or not 0 <= node < 36
        or degree not in (2, 3, 4)
    ):
        raise RuntimeError(
            "Invalid attribution metadata."
        )

    signed = float(row["signed_ig"])
    absolute = float(row["absolute_ig"])

    if (
        not np.isfinite(signed)
        or not np.isfinite(absolute)
        or absolute < 0
    ):
        raise RuntimeError(
            "Invalid attribution value."
        )

    if node in degree_map:
        if degree_map[node] != degree:
            raise RuntimeError(
                "Inconsistent node degree."
            )
    else:
        degree_map[node] = degree

    key = (
        model, seed, region, scanner
    )

    if node in vectors[key]:
        raise RuntimeError(
            "Duplicate attribution node."
        )

    vectors[key][node] = (
        signed, absolute
    )

regions = sorted({
    row["region_id"] for row in details
})

if len(regions) != 10:
    raise RuntimeError(
        "Expected 10 validation regions."
    )

expected_vectors = {
    (model, seed, region, scanner)
    for model in MODELS
    for seed in SEEDS
    for region in regions
    for scanner in SCANNERS
}

if set(vectors) != expected_vectors:
    raise RuntimeError(
        "Incomplete attribution design."
    )

if set(degree_map) != set(range(36)):
    raise RuntimeError(
        "Incomplete node degrees."
    )

degree = np.array([
    degree_map[i] for i in range(36)
])

groups = {
    d: np.where(degree == d)[0]
    for d in (2, 3, 4)
}

if {
    d: len(idx) for d, idx in groups.items()
} != {2: 4, 3: 16, 4: 16}:
    raise RuntimeError(
        "Unexpected graph-degree distribution."
    )

profiles = {}

for key, values in vectors.items():
    if set(values) != set(range(36)):
        raise RuntimeError(
            "Missing node attribution."
        )

    profiles[key] = {
        "signed": np.array([
            values[i][0] for i in range(36)
        ]),
        "absolute": np.array([
            values[i][1] for i in range(36)
        ]),
    }

detail_index = {}

for row in details:
    key = (
        row["model"],
        int(row["seed"]),
        row["region_id"],
    )

    if key in detail_index:
        raise RuntimeError(
            "Duplicate summary row."
        )

    detail_index[key] = row

expected_detail = {
    (model, seed, region)
    for model in MODELS
    for seed in SEEDS
    for region in regions
}

if set(detail_index) != expected_detail:
    raise RuntimeError(
        "Incomplete detail records."
    )

print("Validated region-seed-model units:", len(details))
print("Test: NOT EVALUATED")


# -------------------------------------------------
# 3. Reproduce and adjust IG agreement
# -------------------------------------------------

print("\n[3] Evaluating degree-controlled agreement...")

results = []

for seed in SEEDS:
    for region in regions:

        # Apply the same permutation draws to both
        # models to reduce Monte Carlo comparison noise.
        rng = seeded_rng(
            "phase4226", seed, region
        )

        permutations = []

        for _ in range(NULL_REPS):
            p = np.arange(36)

            for indices in groups.values():
                p[indices] = rng.permutation(
                    indices
                )

            permutations.append(p)

        for model in MODELS:
            source_row = detail_index[
                (model, seed, region)
            ]

            output = {
                "model": model,
                "seed": seed,
                "region_id": region,
            }

            for attribution in (
                "absolute", "signed"
            ):
                a = profiles[
                    (model, seed, region, "AT2")
                ][attribution]

                b = profiles[
                    (model, seed, region, "GT450")
                ][attribution]

                observed_rho = spearman(a, b)
                observed_overlap = top5_overlap(a, b)

                if observed_rho is None:
                    raise RuntimeError(
                        "Undefined attribution correlation."
                    )

                original_name = (
                    f"{attribution}_node_spearman"
                )

                original_overlap = (
                    f"{attribution}_top5_overlap"
                )

                if abs(
                    observed_rho
                    - float(source_row[original_name])
                ) > 1e-8:
                    raise RuntimeError(
                        f"Correlation reproduction failed: "
                        f"{model}, {seed}, {region}"
                    )

                if observed_overlap != int(
                    source_row[original_overlap]
                ):
                    raise RuntimeError(
                        "Top-5 overlap reproduction failed."
                    )

                null_rho = np.array([
                    spearman(a, b[p])
                    for p in permutations
                ], dtype=np.float64)

                null_overlap = np.array([
                    top5_overlap(a, b[p])
                    for p in permutations
                ], dtype=np.float64)

                if not np.isfinite(
                    null_rho
                ).all():
                    raise RuntimeError(
                        "Invalid null correlations."
                    )

                output[
                    f"{attribution}_observed_rho"
                ] = observed_rho

                output[
                    f"{attribution}_null_rho"
                ] = float(null_rho.mean())

                output[
                    f"{attribution}_excess_rho"
                ] = (
                    observed_rho
                    - float(null_rho.mean())
                )

                output[
                    f"{attribution}_observed_overlap"
                ] = observed_overlap

                output[
                    f"{attribution}_null_overlap"
                ] = float(null_overlap.mean())

                output[
                    f"{attribution}_excess_overlap"
                ] = (
                    observed_overlap
                    - float(null_overlap.mean())
                )

            for scanner in SCANNERS:
                prefix = scanner.lower()

                for ranking in (
                    "absolute",
                    "signed_positive",
                ):
                    field = (
                        f"{prefix}_{ranking}_excess"
                    )

                    output[field] = float(
                        source_row[field]
                    )

                completion = (
                    f"{prefix}_completeness_error"
                )

                output[completion] = float(
                    source_row[completion]
                )

            if not all(
                np.isfinite(v)
                for k, v in output.items()
                if k not in (
                    "model", "seed", "region_id"
                )
            ):
                raise RuntimeError(
                    "Invalid audit metric."
                )

            results.append(output)

print("Prior IG correlations and overlaps: VERIFIED")


# -------------------------------------------------
# 4. Summarize models and regional differences
# -------------------------------------------------

print("\n[4] Comparing models and fidelity...")

metrics = [
    "absolute_observed_rho",
    "absolute_excess_rho",
    "absolute_observed_overlap",
    "absolute_excess_overlap",
    "signed_observed_rho",
    "signed_excess_rho",
    "signed_observed_overlap",
    "signed_excess_overlap",
    "at2_absolute_excess",
    "gt450_absolute_excess",
    "at2_signed_positive_excess",
    "gt450_signed_positive_excess",
]

model_summary = {}

for model in MODELS:
    subset = [
        r for r in results
        if r["model"] == model
    ]

    model_summary[model] = {
        metric: mean_metric(subset, metric)
        for metric in metrics
    }

    model_summary[model][
        "positive_at2_signed_fidelity_units"
    ] = sum(
        r["at2_signed_positive_excess"] > 0
        for r in subset
    )

    model_summary[model][
        "positive_gt450_signed_fidelity_units"
    ] = sum(
        r["gt450_signed_positive_excess"] > 0
        for r in subset
    )

    print(
        f"  {model} | "
        f"Absolute rho "
        f"{model_summary[model]['absolute_observed_rho']:.4f} | "
        f"Excess rho "
        f"{model_summary[model]['absolute_excess_rho']:+.4f} | "
        f"AT2 signed fidelity "
        f"{model_summary[model]['at2_signed_positive_excess']:+.4f}"
    )


# -------------------------------------------------
# 5. Paired region-cluster bootstrap
# -------------------------------------------------

print("\n[5] Computing descriptive regional uncertainty...")

result_index = {
    (r["model"], r["seed"], r["region_id"]): r
    for r in results
}

region_rows = []
bootstrap = {}

for region in regions:
    row = {"region_id": region}

    for metric in metrics:
        old = np.mean([
            result_index[
                ("ResidualGate05", seed, region)
            ][metric]
            for seed in SEEDS
        ])

        new = np.mean([
            result_index[
                ("TopologyConsistentResidual", seed, region)
            ][metric]
            for seed in SEEDS
        ])

        row[
            f"difference_{metric}"
        ] = float(new - old)

    region_rows.append(row)

for metric in metrics:
    differences = np.array([
        r[f"difference_{metric}"]
        for r in region_rows
    ])

    rng = seeded_rng(
        "phase4226_bootstrap", metric
    )

    draws = rng.integers(
        0, len(regions),
        size=(
            BOOTSTRAP_REPS,
            len(regions),
        ),
    )

    means = differences[draws].mean(axis=1)

    lower, upper = np.quantile(
        means, [0.025, 0.975]
    )

    bootstrap[metric] = {
        "difference": float(
            differences.mean()
        ),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "positive_regions": int(
            np.sum(differences > 0)
        ),
        "negative_regions": int(
            np.sum(differences < 0)
        ),
    }

    if metric in (
        "absolute_excess_rho",
        "signed_excess_rho",
        "absolute_excess_overlap",
        "signed_excess_overlap",
        "at2_signed_positive_excess",
        "gt450_signed_positive_excess",
    ):
        print(
            f"  {metric} | "
            f"Difference {differences.mean():+.4f} | "
            f"Interval [{lower:+.4f}, {upper:+.4f}]"
        )


# -------------------------------------------------
# 6. Save provenance and results
# -------------------------------------------------

print("\n[6] Saving corrected IG audit...")

detail_columns = [
    "model", "seed", "region_id",
    *[
        f"{kind}_{suffix}"
        for kind in ("absolute", "signed")
        for suffix in (
            "observed_rho",
            "null_rho",
            "excess_rho",
            "observed_overlap",
            "null_overlap",
            "excess_overlap",
        )
    ],
    "at2_absolute_excess",
    "gt450_absolute_excess",
    "at2_signed_positive_excess",
    "gt450_signed_positive_excess",
    "at2_completeness_error",
    "gt450_completeness_error",
]

save_csv(
    DETAIL_OUT, results, detail_columns
)

region_columns = [
    "region_id",
    *[
        f"difference_{m}"
        for m in metrics
    ],
]

save_csv(
    REGION_OUT, region_rows,
    region_columns
)

report = {
    "phase": "4.2",
    "step": "4.2.26",
    "generated_utc":
        datetime.now(timezone.utc).isoformat(),
    "experiment":
        "corrected_integrated_gradients_paired_audit",
    "source_report_sha256":
        sha256(SOURCE_REPORT),
    "source_nodes_sha256":
        sha256(SOURCE_NODES),
    "source_detail_sha256":
        sha256(SOURCE_DETAIL),
    "source_summary_sha256":
        sha256(SOURCE_SUMMARY),
    "graph_archive_sha256":
        source["graph_archive_sha256"],
    "protocol_sha256":
        source["protocol_sha256"],
    "models": list(MODELS),
    "seeds": list(SEEDS),
    "regions": len(regions),
    "top_k": TOP_K,
    "degree_null_repetitions": NULL_REPS,
    "degree_null_method":
        "permute_GT450_attributions_within_degree_groups",
    "model_summary": model_summary,
    "paired_region_bootstrap": {
        "resampled_unit": "spatial_region",
        "repetitions": BOOTSTRAP_REPS,
        "results": bootstrap,
    },
    "detail_table_sha256":
        sha256(DETAIL_OUT),
    "region_table_sha256":
        sha256(REGION_OUT),
    "training_performed": False,
    "model_inference_performed": False,
    "checkpoint_selection_performed": False,
    "test_evaluated": False,
    "limitations": [
        "Only ten spatial validation regions",
        "Training seeds and repetitions are not independent patients",
        "Confidence intervals are exploratory and descriptive",
        "Null permutations control node degree only",
        "Positive deletion fidelity does not establish biological relevance",
        "Multiple metrics were inspected after model development",
        "No independent slide or patient generalization",
    ],
}

save_json(
    REPORT_OUT, report
)

print("\n" + "=" * 70)
print("PHASE 4.2.26 - CORRECTED IG AUDIT COMPLETED")
print("=" * 70)

for model in MODELS:
    s = model_summary[model]

    print(
        f"{model} | "
        f"Absolute rho {s['absolute_observed_rho']:.4f} | "
        f"Excess rho {s['absolute_excess_rho']:+.4f} | "
        f"Signed AT2 fidelity {s['at2_signed_positive_excess']:+.4f}"
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
print("\nSTEP 4.2.26 COMPLETED")
