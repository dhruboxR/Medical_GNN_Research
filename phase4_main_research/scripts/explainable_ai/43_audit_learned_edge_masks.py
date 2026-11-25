
"""
PHASE 4.2.28 - LEARNED EDGE-MASK RELIABILITY AUDIT

Audits Phase 4.2.27 learned GNNExplainer-style masks.

Measurements:
  - Mask saturation and selectivity
  - Edge endpoint-degree type distributions
  - Reproduction of saved Spearman and top-12 overlap
  - Edge-type-stratified permutation nulls
  - Paired region-level comparison of models
  - Descriptive spatial-region bootstrap intervals
  - Saved deletion-fidelity and objective preservation

No model inference or training.
No test evaluation.
No historical output overwrite.

Note: This is a structural-null and diagnostic audit.
It does NOT recompute edge-deletion fidelity against
new edge-type-matched deletion interventions.
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

EDGE_SOURCE = BASE / "outputs/xai/42_learned_edge_masks.csv"
DETAIL_SOURCE = BASE / "outputs/xai/42_edge_mask_explanation_detail.csv"
SUMMARY_SOURCE = BASE / "outputs/xai/42_edge_mask_summary.csv"
SOURCE_REPORT = BASE / "outputs/reports/42_learned_edge_mask_report.json"

MASK_OUT = BASE / "outputs/xai/43_edge_mask_diagnostics.csv"
PAIR_OUT = BASE / "outputs/xai/43_edge_mask_paired_audit.csv"
REGION_OUT = BASE / "outputs/xai/43_edge_mask_region_comparison.csv"
REPORT_OUT = BASE / "outputs/reports/43_edge_mask_reliability_report.json"

MODELS = (
    "ResidualGate05",
    "TopologyConsistentResidual",
)

SEEDS = (42, 123, 2026)
SCANNERS = ("AT2", "GT450")

NODES = 36
EDGES = 60
TOP_K = 12
NULL_REPS = 1000
BOOTSTRAP_REPS = 5000


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(content, stream, indent=2)


def stable_rng(*parts):
    key = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return np.random.default_rng(
        int.from_bytes(digest[:8], "little")
    )


def average_ranks(values):
    x = np.asarray(values, dtype=np.float64)
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


def spearman(a, b):
    x = average_ranks(a)
    y = average_ranks(b)

    x -= x.mean()
    y -= y.mean()

    denominator = (
        np.linalg.norm(x) * np.linalg.norm(y)
    )

    if denominator <= 1e-12:
        return None

    return float(np.dot(x, y) / denominator)


def overlap(a, b):
    left = set(
        np.argsort(-a, kind="stable")[:TOP_K]
    )
    right = set(
        np.argsort(-b, kind="stable")[:TOP_K]
    )
    return len(left & right)


def mean(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        raise RuntimeError("Invalid mean input.")
    return float(values.mean())


print("=" * 70, flush=True)
print("PHASE 4.2.28 - LEARNED EDGE-MASK RELIABILITY AUDIT", flush=True)
print("=" * 70, flush=True)


# --------------------------------------------------
# 1. Verify source artifacts and provenance
# --------------------------------------------------

print("\n[1] Verifying saved edge masks...", flush=True)

for path in (
    EDGE_SOURCE, DETAIL_SOURCE,
    SUMMARY_SOURCE, SOURCE_REPORT
):
    if not path.is_file():
        raise RuntimeError(f"Missing source: {path}")

for path in (
    MASK_OUT, PAIR_OUT, REGION_OUT, REPORT_OUT
):
    if path.exists():
        raise RuntimeError(f"Refusing overwrite: {path}")

source = load_json(SOURCE_REPORT)

if (
    source["step"] != "4.2.27"
    or source["test_evaluated"] is not False
    or source["model_training_performed"] is not False
    or source["explainer_optimization_performed"] is not True
):
    raise RuntimeError("Unexpected source protocol.")

for key, path in (
    ("edge_table_sha256", EDGE_SOURCE),
    ("detail_table_sha256", DETAIL_SOURCE),
    ("summary_table_sha256", SUMMARY_SOURCE),
):
    if source[key] != sha256(path):
        raise RuntimeError(f"Source hash mismatch: {path}")

if (
    source["undirected_edges"] != EDGES
    or source["top_k"] != TOP_K
    or source["validation_regions"] != 10
    or tuple(source["seeds"]) != SEEDS
):
    raise RuntimeError("Unexpected explanation configuration.")

edge_rows = load_csv(EDGE_SOURCE)
detail_rows = load_csv(DETAIL_SOURCE)
summary_rows = load_csv(SUMMARY_SOURCE)

if len(edge_rows) != 7200:
    raise RuntimeError("Expected 7200 edge weights.")

if len(detail_rows) != 60:
    raise RuntimeError("Expected 60 paired explanation records.")

if len(summary_rows) != 2:
    raise RuntimeError("Expected two model summaries.")


# --------------------------------------------------
# 2. Reconstruct masks and edge structural groups
# --------------------------------------------------

print("\n[2] Reconstructing complete edge-mask profiles...", flush=True)

masks = defaultdict(dict)
edge_endpoints = {}

for row in edge_rows:
    model = row["model"]
    seed = int(row["seed"])
    region = row["region_id"]
    scanner = row["scanner"]

    index = int(row["edge_index"])
    u = int(row["source_node"])
    v = int(row["target_node"])
    weight = float(row["mask_weight"])

    if (
        model not in MODELS
        or seed not in SEEDS
        or scanner not in SCANNERS
        or not 0 <= index < EDGES
        or not 0 <= u < v < NODES
        or not np.isfinite(weight)
        or not 0 <= weight <= 1
    ):
        raise RuntimeError("Invalid learned-mask record.")

    if index in edge_endpoints:
        if edge_endpoints[index] != (u, v):
            raise RuntimeError("Edge indexing mismatch.")
    else:
        edge_endpoints[index] = (u, v)

    key = (model, seed, region, scanner)

    if index in masks[key]:
        raise RuntimeError("Duplicate edge-mask entry.")

    masks[key][index] = weight

regions = sorted(set(
    row["region_id"] for row in detail_rows
))

if len(regions) != 10:
    raise RuntimeError("Expected ten validation regions.")

expected_keys = {
    (model, seed, region, scanner)
    for model in MODELS
    for seed in SEEDS
    for region in regions
    for scanner in SCANNERS
}

if set(masks) != expected_keys:
    raise RuntimeError("Incomplete scanner-mask design.")

if set(edge_endpoints) != set(range(EDGES)):
    raise RuntimeError("Incomplete edge indexing.")

vectors = {}

for key, items in masks.items():
    if set(items) != set(range(EDGES)):
        raise RuntimeError("Incomplete 60-edge mask.")
    vectors[key] = np.asarray(
        [items[i] for i in range(EDGES)],
        dtype=np.float64,
    )

degree = np.zeros(NODES, dtype=np.int64)

for u, v in edge_endpoints.values():
    degree[u] += 1
    degree[v] += 1

if {
    d: int((degree == d).sum())
    for d in (2, 3, 4)
} != {2: 4, 3: 16, 4: 16}:
    raise RuntimeError("Unexpected 6x6 node degrees.")

edge_types = defaultdict(list)

for i in range(EDGES):
    u, v = edge_endpoints[i]
    edge_type = tuple(
        sorted((int(degree[u]), int(degree[v])))
    )
    edge_types[edge_type].append(i)

groups = {
    label: np.asarray(indices, dtype=np.int64)
    for label, indices in edge_types.items()
}

print(
    "Endpoint-degree edge types:",
    {str(k): len(v) for k, v in sorted(groups.items())},
    flush=True,
)


# --------------------------------------------------
# 3. Validate saved explanation measurements
# --------------------------------------------------

print("\n[3] Reproducing prior agreement results...", flush=True)

detail_index = {}

for row in detail_rows:
    key = (
        row["model"],
        int(row["seed"]),
        row["region_id"],
    )
    if key in detail_index:
        raise RuntimeError("Duplicate detail row.")
    detail_index[key] = row

expected_detail = {
    (model, seed, region)
    for model in MODELS
    for seed in SEEDS
    for region in regions
}

if set(detail_index) != expected_detail:
    raise RuntimeError("Incomplete saved explanation results.")

mask_diagnostics = []
paired_results = []

for seed in SEEDS:
    for region in regions:
        # Reuse identical null permutations across models.
        rng = stable_rng("phase4228", seed, region)
        permutations = []

        for _ in range(NULL_REPS):
            permutation = np.arange(EDGES)

            for indices in groups.values():
                permutation[indices] = rng.permutation(
                    indices
                )

            permutations.append(permutation)

        for model in MODELS:
            original = detail_index[
                (model, seed, region)
            ]

            a = vectors[
                (model, seed, region, "AT2")
            ]
            b = vectors[
                (model, seed, region, "GT450")
            ]

            observed_rho = spearman(a, b)
            observed_overlap = overlap(a, b)

            if original["edge_mask_spearman"] in ("", "None"):
                if observed_rho is not None:
                    raise RuntimeError("Undefined-rho mismatch.")
            elif (
                observed_rho is None
                or abs(
                    observed_rho
                    - float(original["edge_mask_spearman"])
                ) > 1e-8
            ):
                raise RuntimeError(
                    "Historical mask correlation mismatch."
                )

            if observed_overlap != int(
                original["top12_overlap"]
            ):
                raise RuntimeError(
                    "Historical top-12 overlap mismatch."
                )

            null_rhos = [
                spearman(a, b[p])
                for p in permutations
            ]

            valid_rhos = [
                value for value in null_rhos
                if value is not None
            ]

            null_overlap = [
                overlap(a, b[p])
                for p in permutations
            ]

            null_mean_rho = (
                mean(valid_rhos)
                if valid_rhos else None
            )

            null_mean_overlap = mean(null_overlap)

            paired_results.append({
                "model": model,
                "seed": seed,
                "region_id": region,
                "observed_rho": observed_rho,
                "edge_type_null_rho": null_mean_rho,
                "excess_rho": (
                    observed_rho - null_mean_rho
                    if (
                        observed_rho is not None
                        and null_mean_rho is not None
                    )
                    else None
                ),
                "observed_top12_overlap": observed_overlap,
                "null_top12_overlap": null_mean_overlap,
                "excess_top12_overlap":
                    observed_overlap - null_mean_overlap,
                "at2_fidelity_excess": float(
                    original["at2_fidelity_excess"]
                ),
                "gt450_fidelity_excess": float(
                    original["gt450_fidelity_excess"]
                ),
                "at2_objective_change": float(
                    original["at2_objective_change"]
                ),
                "gt450_objective_change": float(
                    original["gt450_objective_change"]
                ),
            })

            for scanner, values in (
                ("AT2", a),
                ("GT450", b),
            ):
                mask_diagnostics.append({
                    "model": model,
                    "seed": seed,
                    "region_id": region,
                    "scanner": scanner,
                    "mean_weight": float(values.mean()),
                    "std_weight": float(values.std()),
                    "min_weight": float(values.min()),
                    "max_weight": float(values.max()),
                    "p10_weight": float(
                        np.quantile(values, 0.1)
                    ),
                    "p90_weight": float(
                        np.quantile(values, 0.9)
                    ),
                    "fraction_below_0_1": float(
                        (values < 0.1).mean()
                    ),
                    "fraction_above_0_9": float(
                        (values > 0.9).mean()
                    ),
                    "fraction_above_half": float(
                        (values > 0.5).mean()
                    ),
                    "range_weight": float(
                        values.max() - values.min()
                    ),
                })

print("Saved edge agreement: VERIFIED", flush=True)


# --------------------------------------------------
# 4. Model-level mask selectivity and fidelity
# --------------------------------------------------

print("\n[4] Auditing selectivity and fidelity...", flush=True)

model_results = {}

for model in MODELS:
    diag = [
        r for r in mask_diagnostics
        if r["model"] == model
    ]

    pairs = [
        r for r in paired_results
        if r["model"] == model
    ]

    if len(diag) != 60 or len(pairs) != 30:
        raise RuntimeError("Incomplete model audit.")

    model_results[model] = {
        "mean_mask_weight": mean(
            [r["mean_weight"] for r in diag]
        ),
        "mean_mask_std": mean(
            [r["std_weight"] for r in diag]
        ),
        "mean_mask_range": mean(
            [r["range_weight"] for r in diag]
        ),
        "fraction_above_half": mean(
            [r["fraction_above_half"] for r in diag]
        ),
        "fraction_below_0_1": mean(
            [r["fraction_below_0_1"] for r in diag]
        ),
        "fraction_above_0_9": mean(
            [r["fraction_above_0_9"] for r in diag]
        ),
        "near_constant_masks_std_below_0_01": sum(
            r["std_weight"] < 0.01 for r in diag
        ),
        "observed_top12_overlap": mean(
            [r["observed_top12_overlap"] for r in pairs]
        ),
        "null_top12_overlap": mean(
            [r["null_top12_overlap"] for r in pairs]
        ),
        "excess_top12_overlap": mean(
            [r["excess_top12_overlap"] for r in pairs]
        ),
        "at2_fidelity_excess": mean(
            [r["at2_fidelity_excess"] for r in pairs]
        ),
        "gt450_fidelity_excess": mean(
            [r["gt450_fidelity_excess"] for r in pairs]
        ),
        "at2_positive_fidelity_units": sum(
            r["at2_fidelity_excess"] > 0 for r in pairs
        ),
        "gt450_positive_fidelity_units": sum(
            r["gt450_fidelity_excess"] > 0 for r in pairs
        ),
        "at2_objective_change": mean(
            [r["at2_objective_change"] for r in pairs]
        ),
        "gt450_objective_change": mean(
            [r["gt450_objective_change"] for r in pairs]
        ),
    }

    valid_rho = [
        r["observed_rho"] for r in pairs
        if r["observed_rho"] is not None
    ]

    valid_excess = [
        r["excess_rho"] for r in pairs
        if r["excess_rho"] is not None
    ]

    model_results[model]["mean_observed_rho"] = (
        mean(valid_rho) if valid_rho else None
    )
    model_results[model]["mean_excess_rho"] = (
        mean(valid_excess) if valid_excess else None
    )
    model_results[model]["defined_rho_units"] = len(valid_rho)

    print(
        f"  {model} | "
        f"Mean mask {model_results[model]['mean_mask_weight']:.4f} | "
        f"Mask std {model_results[model]['mean_mask_std']:.4f} | "
        f"Excess top12 "
        f"{model_results[model]['excess_top12_overlap']:+.3f} | "
        f"AT2 fidelity "
        f"{model_results[model]['at2_fidelity_excess']:+.4f}",
        flush=True,
    )


# --------------------------------------------------
# 5. Descriptive paired region comparisons
# --------------------------------------------------

print("\n[5] Computing paired regional uncertainty...", flush=True)

comparison_metrics = (
    "excess_top12_overlap",
    "at2_fidelity_excess",
    "gt450_fidelity_excess",
    "at2_objective_change",
    "gt450_objective_change",
)

if all(
    r["excess_rho"] is not None
    for r in paired_results
):
    comparison_metrics = (
        "excess_rho",
        *comparison_metrics,
    )

index = {
    (r["model"], r["seed"], r["region_id"]): r
    for r in paired_results
}

region_rows = []

for region in regions:
    row = {"region_id": region}

    for metric in comparison_metrics:
        baseline = mean([
            index[
                ("ResidualGate05", seed, region)
            ][metric]
            for seed in SEEDS
        ])

        regularized = mean([
            index[
                ("TopologyConsistentResidual", seed, region)
            ][metric]
            for seed in SEEDS
        ])

        row[f"baseline_{metric}"] = baseline
        row[f"regularized_{metric}"] = regularized
        row[f"difference_{metric}"] = (
            regularized - baseline
        )

    region_rows.append(row)

bootstrap = {}

for metric in comparison_metrics:
    differences = np.array([
        r[f"difference_{metric}"]
        for r in region_rows
    ])

    rng = stable_rng(
        "phase4228_bootstrap", metric
    )

    selections = rng.integers(
        0, len(regions),
        size=(BOOTSTRAP_REPS, len(regions)),
    )

    distribution = differences[selections].mean(axis=1)

    lower, upper = np.quantile(
        distribution, (0.025, 0.975)
    )

    bootstrap[metric] = {
        "mean_difference": float(
            differences.mean()
        ),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "positive_regions": int(
            (differences > 0).sum()
        ),
        "negative_regions": int(
            (differences < 0).sum()
        ),
    }

    print(
        f"  {metric} | "
        f"Difference {differences.mean():+.4f} | "
        f"Interval [{lower:+.4f}, {upper:+.4f}]",
        flush=True,
    )


# --------------------------------------------------
# 6. Save new research artifacts
# --------------------------------------------------

print("\n[6] Saving edge-mask reliability audit...", flush=True)

mask_columns = [
    "model", "seed", "region_id", "scanner",
    "mean_weight", "std_weight",
    "min_weight", "max_weight",
    "p10_weight", "p90_weight",
    "fraction_below_0_1",
    "fraction_above_0_9",
    "fraction_above_half",
    "range_weight",
]

pair_columns = [
    "model", "seed", "region_id",
    "observed_rho", "edge_type_null_rho",
    "excess_rho",
    "observed_top12_overlap",
    "null_top12_overlap",
    "excess_top12_overlap",
    "at2_fidelity_excess",
    "gt450_fidelity_excess",
    "at2_objective_change",
    "gt450_objective_change",
]

region_columns = ["region_id"]

for metric in comparison_metrics:
    region_columns += [
        f"baseline_{metric}",
        f"regularized_{metric}",
        f"difference_{metric}",
    ]

save_csv(MASK_OUT, mask_diagnostics, mask_columns)
save_csv(PAIR_OUT, paired_results, pair_columns)
save_csv(REGION_OUT, region_rows, region_columns)

report = {
    "phase": "4.2",
    "step": "4.2.28",
    "generated_utc":
        datetime.now(timezone.utc).isoformat(),
    "experiment": "learned_edge_mask_reliability_audit",
    "source_report_sha256": sha256(SOURCE_REPORT),
    "source_edge_sha256": sha256(EDGE_SOURCE),
    "source_detail_sha256": sha256(DETAIL_SOURCE),
    "source_summary_sha256": sha256(SUMMARY_SOURCE),
    "graph_archive_sha256": source["graph_archive_sha256"],
    "protocol_sha256": source["protocol_sha256"],
    "models": list(MODELS),
    "seeds": list(SEEDS),
    "validation_regions": len(regions),
    "edges_per_graph": EDGES,
    "top_k": TOP_K,
    "endpoint_degree_edge_type_counts": {
        f"{u}-{v}": len(indices)
        for (u, v), indices in groups.items()
    },
    "null_method":
        "permute_GT450_edge_weights_within_endpoint_degree_types",
    "null_repetitions": NULL_REPS,
    "model_results": model_results,
    "paired_region_bootstrap": {
        "unit": "spatial_region",
        "repetitions": BOOTSTRAP_REPS,
        "results": bootstrap,
    },
    "mask_table_sha256": sha256(MASK_OUT),
    "pair_table_sha256": sha256(PAIR_OUT),
    "region_table_sha256": sha256(REGION_OUT),
    "training_performed": False,
    "model_inference_performed": False,
    "explainer_optimization_performed": False,
    "checkpoint_selection_performed": False,
    "test_evaluated": False,
    "limitations": [
        "Edge masks were learned on the validation set",
        "Null controls endpoint-degree type only",
        "Fidelity still uses historical unmatched random edge deletions",
        "This audit does not rerun edge interventions",
        "Raw mask sparsity is affected by row normalization",
        "Uniform mask scaling may leave aggregation unchanged",
        "Mask selectivity does not prove biological relevance",
        "Regions share one biological source condition",
        "Region bootstrap is descriptive, not patient-level inference",
        "No independent scanner-shift generalization tested",
    ],
}

save_json(REPORT_OUT, report)

print("\n" + "=" * 70)
print("PHASE 4.2.28 - EDGE-MASK AUDIT COMPLETED")
print("=" * 70)

for model, result in model_results.items():
    print(
        f"{model} | "
        f"Mean mask {result['mean_mask_weight']:.4f} | "
        f"Mask std {result['mean_mask_std']:.4f} | "
        f"Observed overlap "
        f"{result['observed_top12_overlap']:.2f} | "
        f"Edge-type excess overlap "
        f"{result['excess_top12_overlap']:+.3f}"
    )

print("\nSaved:")
print(MASK_OUT)
print(PAIR_OUT)
print(REGION_OUT)
print(REPORT_OUT)

print("\nTraining: NO")
print("Model inference: NO")
print("Checkpoints modified: NO")
print("Held-out test evaluated: NO")
print("XAI: RELIABILITY AUDIT")
print("\nSTEP 4.2.28 COMPLETED")
