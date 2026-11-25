
"""
PHASE 4.2.24 - IG NUMERICAL CONVERGENCE AUDIT

Investigates the large completeness residuals found in
Phase 4.2.23.

Baselines:
  zero: original IG baseline
  training_mean: per-scanner feature mean, training split only

Integration:
  32, 128, 256 midpoint steps
  batched gradient calculations

Target:
  Negative bidirectional within-region InfoNCE
  with the opposite scanner embeddings fixed.

Verifies:
  Source provenance
  Original 32-step zero-baseline completeness
  Checkpoint retrieval reproduction
  Completeness changes with integration resolution

Does not train models or evaluate test data.
Never overwrites historical results.
"""

from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


BASE = (
    Path.home()
    / "MEDICAL_GNN_RESEARCH"
    / "phase4_main_research"
)

GRAPH = BASE / "outputs/features/15_stage_a_spatial_graphs.npz"
GRAPH_REPORT = BASE / "outputs/reports/15_graph_dataset_report.json"
PROTOCOL = BASE / "configs/16_initial_gnn_experimental_protocol.json"
ARCH = BASE / "configs/25_stability_aware_gnn_architecture_v1.json"

IG_REPORT = BASE / "outputs/reports/37_integrated_gradients_report.json"
IG_SUMMARY = BASE / "outputs/xai/37_ig_explanation_summary.csv"
IG_AUDIT = BASE / "outputs/reports/38_ig_reliability_report.json"

SOURCES = {
    "ResidualGate05": {
        "report": BASE / "outputs/reports/28_gate_init05_report.json",
        "folder": BASE / "outputs/checkpoints/28_residual_gate_init05",
        "filename": "gate_init05_seed_{seed}.pt",
        "lambda": 0.0,
    },
    "TopologyConsistentResidual": {
        "report": BASE / "outputs/reports/31_topology_consistent_report.json",
        "folder": BASE / "outputs/checkpoints/31_topology_consistent_residual",
        "filename": "topology_consistent_seed_{seed}.pt",
        "lambda": 1.0,
    },
}

DETAIL_OUT = BASE / "outputs/xai/39_ig_convergence_detail.csv"
SUMMARY_OUT = BASE / "outputs/xai/39_ig_convergence_summary.csv"
REPORT_OUT = BASE / "outputs/reports/39_ig_convergence_report.json"

SEEDS = (42, 123, 2026)
STEPS = (32, 128, 256)
BASELINES = ("zero", "training_mean")
INTEGRATION_BATCH = 16
TEMPERATURE = 0.07

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()


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


def save_json_new(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def normalized_adjacency(edges):
    if edges.shape != (2, 120):
        raise RuntimeError("Incorrect edge shape.")

    matrix = torch.zeros(36, 36, dtype=torch.float32)
    seen = set()

    for u, v in edges.T.tolist():
        u, v = int(u), int(v)

        if (
            not 0 <= u < 36
            or not 0 <= v < 36
            or u == v
            or (u, v) in seen
        ):
            raise RuntimeError("Invalid graph edge.")

        seen.add((u, v))
        matrix[v, u] = 1.0

    if len(seen) != 120:
        raise RuntimeError("Expected 120 directed edges.")

    if any((v, u) not in seen for u, v in seen):
        raise RuntimeError("Graph must be reciprocal.")

    if not torch.equal(matrix, matrix.T):
        raise RuntimeError("Graph symmetry mismatch.")

    degree = matrix.sum(dim=1, keepdim=True)

    if (degree <= 0).any():
        raise RuntimeError("Isolated graph node.")

    return matrix / degree


class SAGEConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neighbor_linear = nn.Linear(
            in_dim, out_dim, bias=False
        )

    def forward(self, x, weights):
        return (
            self.self_linear(x)
            + self.neighbor_linear(
                torch.matmul(weights, x)
            )
        )


class ResidualGraphSAGE(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer(
            "neighbor_weights", weights.clone()
        )
        self.conv1 = SAGEConv(384, 256)
        self.conv2 = SAGEConv(256, 384)
        self.gate_logit = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        identity = F.normalize(x, dim=-1)

        hidden = F.gelu(
            self.conv1(x, self.neighbor_weights)
        )

        graph = F.normalize(
            self.conv2(
                hidden, self.neighbor_weights
            ),
            dim=-1,
        )

        return F.normalize(
            identity
            + torch.sigmoid(self.gate_logit) * graph,
            dim=-1,
        )


def score_single(a, b):
    logits = (a @ b.T) / TEMPERATURE
    targets = torch.arange(
        36, device=logits.device
    )

    return -0.5 * (
        F.cross_entropy(logits, targets)
        + F.cross_entropy(logits.T, targets)
    )


def score_batch(explained, opponent, scanner):
    """
    explained: (K,36,384)
    opponent:  (36,384)

    Output: sum of K independent scalar objectives.

    Summing, rather than averaging, avoids an
    unintended factor of 1/K in IG gradients.
    """
    if explained.ndim != 3:
        raise RuntimeError("Expected batched embeddings.")

    k = explained.shape[0]
    opponent = opponent.unsqueeze(0).expand(
        k, -1, -1
    )

    if scanner == "AT2":
        logits = torch.bmm(
            explained, opponent.transpose(1, 2)
        ) / TEMPERATURE
    else:
        logits = torch.bmm(
            opponent, explained.transpose(1, 2)
        ) / TEMPERATURE

    targets = torch.arange(
        36, device=logits.device
    ).expand(k, 36).reshape(-1)

    forward = F.cross_entropy(
        logits.reshape(-1, 36),
        targets,
        reduction="none",
    ).reshape(k, 36).mean(dim=1)

    reverse = F.cross_entropy(
        logits.transpose(1, 2).reshape(-1, 36),
        targets,
        reduction="none",
    ).reshape(k, 36).mean(dim=1)

    return (-0.5 * (forward + reverse)).sum()


def integrated_gradients(
    model, features, baseline, opponent,
    scanner, steps
):
    displacement = features - baseline

    gradient_total = torch.zeros_like(features)

    for start in range(0, steps, INTEGRATION_BATCH):
        end = min(
            start + INTEGRATION_BATCH,
            steps,
        )

        alpha = (
            torch.arange(
                start, end,
                dtype=features.dtype,
                device=DEVICE,
            ) + 0.5
        ) / steps

        interpolated = (
            baseline.unsqueeze(0)
            + alpha[:, None, None]
            * displacement.unsqueeze(0)
        ).detach().requires_grad_(True)

        outputs = model(interpolated)

        objective = score_batch(
            outputs, opponent, scanner
        )

        gradients = torch.autograd.grad(
            objective,
            interpolated,
            retain_graph=False,
            create_graph=False,
        )[0]

        gradient_total += gradients.detach().sum(
            dim=0
        )

    ig = displacement * (
        gradient_total / steps
    )

    with torch.no_grad():
        real_output = model(features)
        reference_output = model(baseline)

        if scanner == "AT2":
            real_score = score_single(
                real_output, opponent
            )
            reference_score = score_single(
                reference_output, opponent
            )
        else:
            real_score = score_single(
                opponent, real_output
            )
            reference_score = score_single(
                opponent, reference_output
            )

        actual_delta = float(
            (real_score - reference_score).item()
        )

    attributed_delta = float(
        ig.sum().item()
    )

    absolute_error = abs(
        actual_delta - attributed_delta
    )

    relative_error = (
        absolute_error
        / max(abs(actual_delta), 1e-6)
    )

    values = (
        actual_delta, attributed_delta,
        absolute_error, relative_error
    )

    if not all(math.isfinite(v) for v in values):
        raise RuntimeError(
            "Non-finite IG completeness result."
        )

    return {
        "actual_score_delta": actual_delta,
        "ig_attribution_sum": attributed_delta,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
    }


print("=" * 70, flush=True)
print("PHASE 4.2.24 - IG CONVERGENCE DIAGNOSTIC", flush=True)
print("=" * 70, flush=True)


# --------------------------------------------------
# 1. Verify source experiments
# --------------------------------------------------

print("\n[1] Checking experiment provenance...", flush=True)

required = [
    GRAPH, GRAPH_REPORT, PROTOCOL, ARCH,
    IG_REPORT, IG_SUMMARY, IG_AUDIT,
]

required += [
    spec["report"] for spec in SOURCES.values()
]

for path in required:
    if not path.is_file():
        raise RuntimeError(
            f"Missing input: {path}"
        )

for path in (DETAIL_OUT, SUMMARY_OUT, REPORT_OUT):
    if path.exists():
        raise RuntimeError(
            f"Refusing overwrite: {path}"
        )

graph_hash = sha256(GRAPH)
protocol_hash = sha256(PROTOCOL)
arch_hash = sha256(ARCH)

if (
    read_json(GRAPH_REPORT)["graph_archive_sha256"]
    != graph_hash
):
    raise RuntimeError("Graph hash mismatch.")

if (
    read_json(PROTOCOL)["source_graph_sha256"]
    != graph_hash
):
    raise RuntimeError("Protocol hash mismatch.")

architecture = read_json(ARCH)

if (
    architecture["graph_archive_sha256"] != graph_hash
    or architecture["initial_protocol_sha256"]
    != protocol_hash
):
    raise RuntimeError("Architecture hash mismatch.")

ig_report = read_json(IG_REPORT)

if (
    ig_report["graph_archive_sha256"] != graph_hash
    or ig_report["protocol_sha256"]
    != protocol_hash
    or ig_report["summary_table_sha256"]
    != sha256(IG_SUMMARY)
    or ig_report["test_evaluated"] is not False
):
    raise RuntimeError("Original IG provenance mismatch.")

if (
    read_json(IG_AUDIT)["source_ig_report_sha256"]
    != sha256(IG_REPORT)
):
    raise RuntimeError("IG audit provenance mismatch.")

references = {}

for name, spec in SOURCES.items():
    reference = read_json(spec["report"])

    if (
        reference["graph_archive_sha256"] != graph_hash
        or reference["protocol_sha256"] != protocol_hash
        or reference["parent_architecture_sha256"]
        != arch_hash
        or reference["test_evaluated"] is not False
    ):
        raise RuntimeError(
            f"Model provenance mismatch: {name}"
        )

    references[name] = reference

old_ig = {}

for item in read_csv(IG_SUMMARY):
    key = (
        item["model"],
        int(item["seed"]),
        item["region_id"],
    )

    if key in old_ig:
        raise RuntimeError(
            "Duplicate original IG summary."
        )

    old_ig[key] = item


# --------------------------------------------------
# 2. Load train and validation features
# --------------------------------------------------

print("\n[2] Preparing training-derived baselines...", flush=True)

with np.load(GRAPH, allow_pickle=False) as archive:
    features = archive["node_features"].copy()
    edges = archive["edge_index"].copy()
    regions = archive["region_id"].astype(str)
    splits = archive["spatial_split"].astype(str)
    scanners = archive["scanner"].astype(str)
    tile_ids = archive["tile_id"].astype(str)

if (
    features.shape != (120, 36, 384)
    or features.dtype != np.float32
    or not np.isfinite(features).all()
):
    raise RuntimeError("Invalid graph feature tensor.")

if Counter(splits) != Counter({
    "train": 80,
    "validation": 20,
    "test": 20,
}):
    raise RuntimeError("Unexpected spatial partitions.")

train_means = {}

for scanner in ("AT2", "GT450"):
    mask = (
        (splits == "train")
        & (scanners == scanner)
    )

    if int(mask.sum()) != 40:
        raise RuntimeError(
            "Incorrect training scanner count."
        )

    scanner_features = features[mask]

    mean_vector = scanner_features.mean(
        axis=(0, 1),
        dtype=np.float64,
    ).astype(np.float32)

    if np.linalg.norm(mean_vector) < 1e-8:
        raise RuntimeError(
            "Degenerate training mean baseline."
        )

    train_means[scanner] = torch.from_numpy(
        np.broadcast_to(
            mean_vector,
            (36, 384),
        ).copy()
    ).to(DEVICE)

by_region = defaultdict(dict)

for index, region in enumerate(regions):
    if splits[index] != "validation":
        continue

    scanner = scanners[index]

    if scanner in by_region[region]:
        raise RuntimeError(
            "Duplicate validation scanner graph."
        )

    by_region[region][scanner] = index

if len(by_region) != 10:
    raise RuntimeError(
        "Expected ten validation regions."
    )

pairs = {}

for region, scanners_index in sorted(
    by_region.items()
):
    if set(scanners_index) != {"AT2", "GT450"}:
        raise RuntimeError(
            "Incomplete scanner pair."
        )

    ia = scanners_index["AT2"]
    ib = scanners_index["GT450"]

    if not np.array_equal(
        tile_ids[ia], tile_ids[ib]
    ):
        raise RuntimeError(
            "Scanner node ordering mismatch."
        )

    pairs[region] = (
        torch.from_numpy(
            features[ia]
        ).to(DEVICE),
        torch.from_numpy(
            features[ib]
        ).to(DEVICE),
    )

weights = normalized_adjacency(edges).to(DEVICE)

expected_keys = {
    (model, seed, region)
    for model in SOURCES
    for seed in SEEDS
    for region in pairs
}

if set(old_ig) != expected_keys:
    raise RuntimeError(
        "Original IG summary design incomplete."
    )

print("Validation region pairs:", len(pairs))
print("Integration steps:", STEPS)
print("Baseline conditions:", BASELINES)
print("Device:", DEVICE)
print("Test partition: RESERVED")


# --------------------------------------------------
# 3. Compute numerical completeness
# --------------------------------------------------

print("\n[3] Running IG convergence comparisons...", flush=True)

detail_rows = []

for model_name, spec in SOURCES.items():
    reference = references[model_name]

    for seed in SEEDS:
        checkpoint = (
            spec["folder"]
            / spec["filename"].format(seed=seed)
        )

        if sha256(checkpoint) != (
            reference["checkpoint_hashes"][str(seed)]
        ):
            raise RuntimeError(
                "Checkpoint integrity mismatch."
            )

        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        if (
            int(payload["seed"]) != seed
            or payload["graph_archive_sha256"]
            != graph_hash
            or payload["protocol_sha256"]
            != protocol_hash
            or payload["parent_architecture_sha256"]
            != arch_hash
            or payload["test_evaluated"] is not False
        ):
            raise RuntimeError(
                "Checkpoint metadata mismatch."
            )

        if model_name == "TopologyConsistentResidual":
            if payload["lambda_topology"] != 1.0:
                raise RuntimeError(
                    "Unexpected consistency loss weight."
                )

        model = ResidualGraphSAGE(weights).to(DEVICE)

        model.load_state_dict(
            payload["model_state_dict"],
            strict=True,
        )

        model.eval()

        for parameter in model.parameters():
            parameter.requires_grad_(False)

        recalls = []

        for region, (at2, gt450) in pairs.items():
            with torch.no_grad():
                emb_a = model(at2)
                emb_b = model(gt450)

                retrieval = float(
                    (
                        (emb_a @ emb_b.T).argmax(-1)
                        == torch.arange(36, device=DEVICE)
                    ).float().mean().item()
                )

                recalls.append(retrieval)

            for scanner in ("AT2", "GT450"):
                if scanner == "AT2":
                    inputs = at2
                    opponent = emb_b.detach()
                else:
                    inputs = gt450
                    opponent = emb_a.detach()

                for baseline_name in BASELINES:
                    if baseline_name == "zero":
                        baseline = torch.zeros_like(
                            inputs
                        )
                    else:
                        baseline = train_means[scanner]

                    for steps in STEPS:
                        metrics = integrated_gradients(
                            model,
                            inputs,
                            baseline,
                            opponent,
                            scanner,
                            steps,
                        )

                        record = {
                            "model": model_name,
                            "seed": seed,
                            "region_id": region,
                            "scanner": scanner,
                            "baseline": baseline_name,
                            "steps": steps,
                            **metrics,
                        }

                        detail_rows.append(record)

                        if (
                            baseline_name == "zero"
                            and steps == 32
                        ):
                            original = old_ig[
                                (model_name, seed, region)
                            ]

                            field = (
                                "at2_completeness_abs_error"
                                if scanner == "AT2"
                                else "gt450_completeness_abs_error"
                            )

                            old_error = float(
                                original[field]
                            )

                            if abs(
                                metrics["absolute_error"]
                                - old_error
                            ) > 0.002:
                                raise RuntimeError(
                                    "Original 32-step IG "
                                    "completeness not reproduced: "
                                    f"{model_name}, {seed}, "
                                    f"{region}, {scanner}. "
                                    f"Expected {old_error:.6f}, "
                                    f"obtained "
                                    f"{metrics['absolute_error']:.6f}"
                                )

        expected_recall = next(
            row["validation_recall_at_1"]
            for row in reference["seed_results"]
            if int(row["seed"]) == seed
        )

        if abs(
            np.mean(recalls) - expected_recall
        ) > 1e-5:
            raise RuntimeError(
                "Checkpoint retrieval reproduction failed."
            )

        print(
            f"  {model_name}, seed {seed}: "
            "IG convergence completed",
            flush=True,
        )


# --------------------------------------------------
# 4. Summarize completeness and convergence
# --------------------------------------------------

print("\n[4] Aggregating completeness statistics...", flush=True)

summary_rows = []

for model_name in SOURCES:
    for baseline_name in BASELINES:
        for steps in STEPS:
            subset = [
                row for row in detail_rows
                if (
                    row["model"] == model_name
                    and row["baseline"]
                    == baseline_name
                    and row["steps"] == steps
                )
            ]

            if len(subset) != 60:
                raise RuntimeError(
                    "Expected 60 scanner explanations."
                )

            errors = np.asarray([
                row["absolute_error"]
                for row in subset
            ])

            relative = np.asarray([
                row["relative_error"]
                for row in subset
            ])

            deltas = np.asarray([
                row["actual_score_delta"]
                for row in subset
            ])

            result = {
                "model": model_name,
                "baseline": baseline_name,
                "steps": steps,
                "explanations": len(subset),
                "mean_abs_error": float(errors.mean()),
                "median_abs_error": float(
                    np.median(errors)
                ),
                "p90_abs_error": float(
                    np.quantile(errors, 0.9)
                ),
                "mean_relative_error": float(
                    relative.mean()
                ),
                "median_relative_error": float(
                    np.median(relative)
                ),
                "mean_abs_score_delta": float(
                    np.abs(deltas).mean()
                ),
            }

            summary_rows.append(result)

            print(
                f"  {model_name:26s} | "
                f"{baseline_name:13s} | "
                f"{steps:3d} steps | "
                f"Mean error "
                f"{result['mean_abs_error']:.6f} | "
                f"Median relative "
                f"{result['median_relative_error']:.4f}",
                flush=True,
            )


# --------------------------------------------------
# 5. Save new artifacts
# --------------------------------------------------

print("\n[5] Saving convergence audit...", flush=True)

detail_fields = [
    "model",
    "seed",
    "region_id",
    "scanner",
    "baseline",
    "steps",
    "actual_score_delta",
    "ig_attribution_sum",
    "absolute_error",
    "relative_error",
]

save_csv_new(
    DETAIL_OUT, detail_rows, detail_fields
)

summary_fields = [
    "model",
    "baseline",
    "steps",
    "explanations",
    "mean_abs_error",
    "median_abs_error",
    "p90_abs_error",
    "mean_relative_error",
    "median_relative_error",
    "mean_abs_score_delta",
]

save_csv_new(
    SUMMARY_OUT,
    summary_rows,
    summary_fields,
)

report = {
    "phase": "4.2",
    "step": "4.2.24",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "experiment": "ig_completeness_convergence",
    "graph_archive_sha256": graph_hash,
    "protocol_sha256": protocol_hash,
    "architecture_sha256": arch_hash,
    "source_ig_report_sha256": sha256(IG_REPORT),
    "source_ig_audit_sha256": sha256(IG_AUDIT),
    "source_model_reports_sha256": {
        model: sha256(spec["report"])
        for model, spec in SOURCES.items()
    },
    "baselines": list(BASELINES),
    "baseline_training_only": True,
    "integration_steps": list(STEPS),
    "integration_batch_size": INTEGRATION_BATCH,
    "integration_method": "midpoint",
    "explanation_target":
        "negative_bidirectional_within_region_InfoNCE",
    "opposite_scanner_embeddings_fixed": True,
    "validation_region_pairs": len(pairs),
    "training_seeds": list(SEEDS),
    "summary": summary_rows,
    "detail_table_sha256": sha256(DETAIL_OUT),
    "summary_table_sha256": sha256(SUMMARY_OUT),
    "training_performed": False,
    "checkpoint_selection_performed": False,
    "test_evaluated": False,
    "limitations": [
        "All completeness comparisons use the same validation regions",
        "Nonzero mean baseline is still a reference intervention",
        "Zero baseline passes through near-zero normalized inputs",
        "Convergence does not prove clinical explanation validity",
        "No new node attribution rankings are selected",
        "No independent biological specimens were evaluated",
    ],
}

save_json_new(REPORT_OUT, report)

print("\n" + "=" * 70)
print("PHASE 4.2.24 - IG CONVERGENCE COMPLETED")
print("=" * 70)

for row in summary_rows:
    print(
        f"{row['model']} | "
        f"{row['baseline']} | "
        f"{row['steps']} steps | "
        f"Mean completeness error "
        f"{row['mean_abs_error']:.6f}"
    )

print("\nSaved:")
print(DETAIL_OUT)
print(SUMMARY_OUT)
print(REPORT_OUT)

print("\nOriginal 32-step IG: REPRODUCED")
print("Training: NO")
print("Checkpoints modified: NO")
print("Test evaluated: NO")
print("XAI: NUMERICAL VALIDITY UNDER REVIEW")
print("\nSTEP 4.2.24 COMPLETED")
