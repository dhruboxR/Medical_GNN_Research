
"""
PHASE 4.2.12 - RESIDUAL GATE SENSITIVITY

Evaluates saved Phase 4.2.11 models without retraining.

For each seed:
    alpha = 0, 0.1, learned, 0.25, 0.5,
            1.0, 2.0, 4.0
    plus the graph-only pathway.

Metrics:
    Within-region Recall@1, MRR
    Global 360-candidate Recall@1, MRR
    Median positive cosine
    Mean off-diagonal node cosine

Restrictions:
    Validation only
    Test partition excluded
    No checkpoint selection
    No model weight changes
    Existing artifacts preserved
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

MODEL_REPORT = (
    BASE / "outputs/reports"
    / "26_gated_residual_graphsage_report.json"
)

CHECKPOINT_DIR = (
    BASE / "outputs/checkpoints"
    / "26_gated_residual_graphsage"
)

TABLE = (
    BASE / "outputs/tables"
    / "27_residual_gate_sensitivity.csv"
)

REPORT = (
    BASE / "outputs/reports"
    / "27_residual_gate_diagnostics.json"
)

SEEDS = (42, 123, 2026)

GATE_VALUES = (
    0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_csv_new(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_json_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def neighbor_matrix(edges):
    if edges.shape != (2, 120):
        raise RuntimeError("Incorrect edge shape.")

    a = torch.zeros(36, 36, dtype=torch.float32)
    seen = set()

    for source, target in edges.T.tolist():
        if (
            not 0 <= source < 36
            or not 0 <= target < 36
            or source == target
            or (source, target) in seen
        ):
            raise RuntimeError("Invalid graph edge.")

        seen.add((source, target))
        a[target, source] = 1.0

    if len(seen) != 120:
        raise RuntimeError("Unexpected edge count.")

    if any((v, u) not in seen for u, v in seen):
        raise RuntimeError("Graph is not reciprocal.")

    degrees = a.sum(-1, keepdim=True)

    if torch.any(degrees <= 0):
        raise RuntimeError("Isolated graph node.")

    return a / degrees


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


class GatedResidualGraphSAGE(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer(
            "neighbor_weights", weights.clone()
        )

        self.conv1 = SAGEConv(384, 256)
        self.conv2 = SAGEConv(256, 384)

        self.gate_logit = nn.Parameter(
            torch.zeros((), dtype=torch.float32)
        )

    def branches(self, x):
        identity = F.normalize(x, dim=-1)

        h = F.gelu(
            self.conv1(x, self.neighbor_weights)
        )

        g = self.conv2(h, self.neighbor_weights)
        graph = F.normalize(g, dim=-1)

        return identity, graph

    def forward(
        self, x, alpha=None, graph_only=False
    ):
        identity, graph = self.branches(x)

        if graph_only:
            return graph

        if alpha is None:
            alpha = float(
                torch.sigmoid(self.gate_logit).item()
            )

        return F.normalize(
            identity + alpha * graph,
            dim=-1
        )


def retrieval(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)

    scores = a @ b.T
    n = scores.shape[0]

    target = torch.arange(
        n, device=scores.device
    )

    positives = scores.diagonal()

    ranks = 1 + (
        scores > positives.unsqueeze(1)
    ).sum(-1)

    return {
        "recall_at_1": float(
            (scores.argmax(-1) == target)
            .float().mean().item()
        ),
        "mrr": float(
            (1.0 / ranks.float()).mean().item()
        ),
        "median_positive_cosine": float(
            positives.median().item()
        ),
    }


def off_diagonal_cosine(features):
    similarity = (
        features @ features.transpose(-1, -2)
    )

    mask = ~torch.eye(
        36, dtype=torch.bool, device=features.device
    )

    return float(
        similarity[..., mask].mean().item()
    )


@torch.inference_mode()
def evaluate(model, validation, alpha, graph_only):
    model.eval()

    at2 = model(
        validation[:, 0],
        alpha=alpha,
        graph_only=graph_only,
    )

    gt450 = model(
        validation[:, 1],
        alpha=alpha,
        graph_only=graph_only,
    )

    within_recall = []
    within_mrr = []

    for i in range(len(validation)):
        result = retrieval(at2[i], gt450[i])

        within_recall.append(result["recall_at_1"])
        within_mrr.append(result["mrr"])

    global_result = retrieval(
        at2.reshape(-1, 384),
        gt450.reshape(-1, 384),
    )

    return {
        "within_recall_at_1": float(
            np.mean(within_recall)
        ),
        "within_mrr": float(np.mean(within_mrr)),
        "global_recall_at_1":
            global_result["recall_at_1"],
        "global_mrr": global_result["mrr"],
        "global_positive_cosine":
            global_result["median_positive_cosine"],
        "at2_offdiagonal_cosine":
            off_diagonal_cosine(at2),
        "gt450_offdiagonal_cosine":
            off_diagonal_cosine(gt450),
    }


print("=" * 70, flush=True)
print("PHASE 4.2.12 - RESIDUAL GATE DIAGNOSTICS", flush=True)
print("=" * 70, flush=True)


# -------------------------------------------------
# 1. Verify provenance
# -------------------------------------------------

print("\n[1] Verifying frozen model sources...", flush=True)

for path in (
    GRAPH,
    GRAPH_REPORT,
    PROTOCOL,
    ARCH,
    MODEL_REPORT,
):
    if not path.is_file():
        raise RuntimeError(f"Missing source: {path}")

for path in (TABLE, REPORT):
    if path.exists():
        raise RuntimeError(
            f"Refusing to overwrite: {path}"
        )

graph_hash = sha256(GRAPH)
protocol_hash = sha256(PROTOCOL)
arch_hash = sha256(ARCH)

if load_json(GRAPH_REPORT)["graph_archive_sha256"] != graph_hash:
    raise RuntimeError("Graph SHA256 mismatch.")

if load_json(PROTOCOL)["source_graph_sha256"] != graph_hash:
    raise RuntimeError("Protocol mismatch.")

architecture = load_json(ARCH)

if (
    architecture["graph_archive_sha256"] != graph_hash
    or architecture["initial_protocol_sha256"] != protocol_hash
):
    raise RuntimeError("Architecture mismatch.")

model_report = load_json(MODEL_REPORT)

if (
    model_report["graph_archive_sha256"] != graph_hash
    or model_report["protocol_sha256"] != protocol_hash
    or model_report["architecture_sha256"] != arch_hash
):
    raise RuntimeError("Model experiment mismatch.")

if model_report["test_evaluated"]:
    raise RuntimeError(
        "Previous experiment unexpectedly used test."
    )


# -------------------------------------------------
# 2. Load validation pairs only
# -------------------------------------------------

print("\n[2] Preparing validation-only inputs...", flush=True)

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
    raise RuntimeError("Invalid graph tensor.")

if dict(Counter(splits)) != {
    "train": 80,
    "validation": 20,
    "test": 20,
}:
    raise RuntimeError("Unexpected spatial splits.")

by_region = defaultdict(dict)

for i, name in enumerate(regions):
    if splits[i] != "validation":
        continue

    if scanners[i] in by_region[name]:
        raise RuntimeError("Duplicate validation scanner.")

    by_region[name][scanners[i]] = i

if len(by_region) != 10:
    raise RuntimeError(
        "Expected exactly 10 validation regions."
    )

paired = []

for region, pair in sorted(by_region.items()):
    if set(pair) != {"AT2", "GT450"}:
        raise RuntimeError("Incomplete scanner pair.")

    a, b = pair["AT2"], pair["GT450"]

    if not np.array_equal(tile_ids[a], tile_ids[b]):
        raise RuntimeError("Node ordering mismatch.")

    paired.append(
        np.stack(
            (features[a], features[b]),
            axis=0,
        )
    )

validation = torch.as_tensor(
    np.stack(paired),
    dtype=torch.float32,
    device=DEVICE,
)

if validation.shape != (10, 2, 36, 384):
    raise RuntimeError("Validation shape mismatch.")

weights = neighbor_matrix(edges).to(DEVICE)

print("Validation region pairs:", len(validation))
print("Candidate positions:", 360)
print("Test regions: NOT EVALUATED")


# -------------------------------------------------
# 3. Evaluate checkpoint gate alternatives
# -------------------------------------------------

print("\n[3] Evaluating gate sensitivity...", flush=True)

rows = []

for seed in SEEDS:
    checkpoint = (
        CHECKPOINT_DIR
        / f"gated_residual_seed_{seed}.pt"
    )

    expected_hash = (
        model_report["checkpoint_hashes"][str(seed)]
    )

    if sha256(checkpoint) != expected_hash:
        raise RuntimeError(
            f"Checkpoint hash mismatch: {seed}"
        )

    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    if (
        int(payload["seed"]) != seed
        or payload["graph_archive_sha256"] != graph_hash
        or payload["protocol_sha256"] != protocol_hash
        or payload["architecture_sha256"] != arch_hash
        or payload["test_evaluated"] is not False
    ):
        raise RuntimeError(
            f"Checkpoint metadata mismatch: {seed}"
        )

    model = GatedResidualGraphSAGE(weights)
    model.load_state_dict(
        payload["model_state_dict"],
        strict=True,
    )

    model = model.to(DEVICE)
    model.eval()

    learned = float(
        torch.sigmoid(model.gate_logit).item()
    )

    # The checkpoint itself is never modified.
    conditions = [
        (f"fixed_{alpha:g}", alpha, False)
        for alpha in GATE_VALUES
    ]

    conditions += [
        ("checkpoint_learned", learned, False),
        ("graph_only", None, True),
    ]

    for name, alpha, graph_only in conditions:
        metrics = evaluate(
            model,
            validation,
            alpha,
            graph_only,
        )

        if not all(
            math.isfinite(float(v))
            for v in metrics.values()
        ):
            raise RuntimeError(
                "Non-finite diagnostic result."
            )

        rows.append({
            "seed": seed,
            "condition": name,
            "alpha": (
                "graph_only"
                if graph_only
                else float(alpha)
            ),
            **metrics,
        })

        print(
            f"  Seed {seed} | "
            f"{name:20s} | "
            f"Within {metrics['within_recall_at_1']:.4f} | "
            f"Global {metrics['global_recall_at_1']:.4f}",
            flush=True,
        )


# -------------------------------------------------
# 4. Aggregate and save without checkpoint changes
# -------------------------------------------------

print("\n[4] Aggregating diagnostic findings...", flush=True)

conditions = sorted({
    row["condition"] for row in rows
})

summary = {}

for condition in conditions:
    subset = [
        row for row in rows
        if row["condition"] == condition
    ]

    summary[condition] = {
        "runs": len(subset),
        "mean_within_recall_at_1": float(
            np.mean([
                row["within_recall_at_1"]
                for row in subset
            ])
        ),
        "mean_global_recall_at_1": float(
            np.mean([
                row["global_recall_at_1"]
                for row in subset
            ])
        ),
        "mean_at2_offdiagonal_cosine": float(
            np.mean([
                row["at2_offdiagonal_cosine"]
                for row in subset
            ])
        ),
        "mean_gt450_offdiagonal_cosine": float(
            np.mean([
                row["gt450_offdiagonal_cosine"]
                for row in subset
            ])
        ),
    }

save_csv_new(
    TABLE,
    rows,
    [
        "seed",
        "condition",
        "alpha",
        "within_recall_at_1",
        "within_mrr",
        "global_recall_at_1",
        "global_mrr",
        "global_positive_cosine",
        "at2_offdiagonal_cosine",
        "gt450_offdiagonal_cosine",
    ],
)

report = {
    "phase": "4.2",
    "step": "4.2.12",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "experiment": "posthoc_residual_gate_sensitivity",
    "graph_archive_sha256": graph_hash,
    "protocol_sha256": protocol_hash,
    "architecture_sha256": arch_hash,
    "source_model_report_sha256":
        sha256(MODEL_REPORT),
    "source_checkpoint_hashes":
        model_report["checkpoint_hashes"],
    "seeds": list(SEEDS),
    "validation_regions": 10,
    "candidate_count": 360,
    "training_performed": False,
    "checkpoint_selection_performed": False,
    "test_evaluated": False,
    "summary": summary,
    "table_sha256": sha256(TABLE),
    "limitations": [
        "Post-hoc gate interventions are exploratory",
        "Graph pathway weights are not retrained",
        "Alpha values above 1 exceed the original gate range",
        "No independent biological validation",
        "Validation data must not be treated as a new test set",
    ],
    "xai_planned": True,
}

save_json_new(REPORT, report)

print("\n" + "=" * 70)
print("PHASE 4.2.12 - GATE DIAGNOSTICS COMPLETED")
print("=" * 70)

for name, result in summary.items():
    print(
        f"{name:20s} | "
        f"Within R@1 "
        f"{result['mean_within_recall_at_1']:.4f} | "
        f"Global R@1 "
        f"{result['mean_global_recall_at_1']:.4f}"
    )

print("\nSaved:")
print(TABLE)
print(REPORT)

print("\nTraining: NO")
print("Checkpoints modified: NO")
print("Test partition evaluated: NO")
print("XAI: PLANNED")
print("\nSTEP 4.2.12 COMPLETED")
