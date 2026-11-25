
"""
PHASE 4.2.14 - REPRESENTATION STABILITY AUDIT

Compare:
    Frozen DINOv2
    MLP (3 saved seeds)
    Original GraphSAGE (3 saved seeds)
    Residual GraphSAGE init=0.5 (3 saved seeds)

Validation only:
    10 regions / 360 matched tissue positions

Metrics:
    Within-region Recall@1 / MRR
    Cross-scanner matched cosine
    Hardest-negative cosine margin
    Mean within-scanner off-diagonal cosine
    Effective node-feature rank
    Global 360-candidate Recall@1

No retraining or checkpoint selection.
No test evaluation.
Existing research artifacts are not modified.
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

REPORT_FILES = {
    "MLP": BASE / "outputs/reports/17_mlp_baseline_report.json",
    "GraphSAGE": BASE / "outputs/reports/19_graphsage_baseline_report.json",
    "ResidualGate05": BASE / "outputs/reports/28_gate_init05_report.json",
}

CHECKPOINT_FILES = {
    "MLP": (
        "17_mlp_baseline",
        "mlp_seed_{seed}.pt",
    ),
    "GraphSAGE": (
        "19_graphsage_baseline",
        "graphsage_seed_{seed}.pt",
    ),
    "ResidualGate05": (
        "28_residual_gate_init05",
        "gate_init05_seed_{seed}.pt",
    ),
}

TABLE = BASE / "outputs/tables/29_representation_region_metrics.csv"
SUMMARY = BASE / "outputs/tables/29_representation_stability_summary.csv"
REPORT = BASE / "outputs/reports/29_representation_stability_report.json"

SEEDS = (42, 123, 2026)
DEVICE = torch.device("cpu")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def save_csv_new(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_json_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)


def neighbor_matrix(edges):
    if edges.shape != (2, 120):
        raise RuntimeError("Unexpected edge shape.")

    matrix = torch.zeros(36, 36, dtype=torch.float32)
    seen = set()

    for source, target in edges.T.tolist():
        if (
            not 0 <= source < 36
            or not 0 <= target < 36
            or source == target
            or (source, target) in seen
        ):
            raise RuntimeError("Invalid edge.")

        seen.add((source, target))
        matrix[target, source] = 1.0

    if len(seen) != 120:
        raise RuntimeError("Unexpected edge count.")

    if any((v, u) not in seen for u, v in seen):
        raise RuntimeError("Missing reciprocal edge.")

    degrees = matrix.sum(-1, keepdim=True)
    if (degrees <= 0).any():
        raise RuntimeError("Isolated node.")

    return matrix / degrees


class SharedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Linear(256, 384),
        )

    def forward(self, x):
        return F.normalize(self.network(x), dim=-1)


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
            + self.neighbor_linear(torch.matmul(weights, x))
        )


class GraphSAGE(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer("neighbor_weights", weights.clone())
        self.conv1 = SAGEConv(384, 256)
        self.conv2 = SAGEConv(256, 384)

    def forward(self, x):
        h = F.gelu(self.conv1(x, self.neighbor_weights))
        return F.normalize(
            self.conv2(h, self.neighbor_weights),
            dim=-1,
        )


class ResidualGraphSAGE(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer("neighbor_weights", weights.clone())
        self.conv1 = SAGEConv(384, 256)
        self.conv2 = SAGEConv(256, 384)
        self.gate_logit = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        identity = F.normalize(x, dim=-1)
        h = F.gelu(self.conv1(x, self.neighbor_weights))
        graph = F.normalize(
            self.conv2(h, self.neighbor_weights),
            dim=-1,
        )
        gate = torch.sigmoid(self.gate_logit)
        return F.normalize(identity + gate * graph, dim=-1)


def effective_rank(z):
    """
    Effective rank based on eigenvalues of the
    centered 36x36 node-feature Gram matrix.

    Higher rank indicates broader spectral diversity,
    but is not by itself evidence of better retrieval.
    """

    array = z.detach().cpu().numpy().astype(np.float64)
    array = array - array.mean(axis=0, keepdims=True)

    gram = array @ array.T
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.maximum(eigenvalues, 0)

    total = float(eigenvalues.sum())
    if total <= 1e-12:
        return 0.0

    probabilities = eigenvalues / total
    probabilities = probabilities[probabilities > 1e-12]

    entropy = -np.sum(
        probabilities * np.log(probabilities)
    )

    return float(np.exp(entropy))


def off_diagonal_cosine(z):
    z = F.normalize(z, dim=-1)
    similarity = z @ z.T

    mask = ~torch.eye(
        len(z),
        dtype=torch.bool,
        device=z.device,
    )

    return float(similarity[mask].mean().item())


def retrieval_metrics(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)

    similarity = a @ b.T
    n = len(a)

    targets = torch.arange(n, device=a.device)
    positives = similarity.diagonal()

    ranks = 1 + (
        similarity > positives[:, None]
    ).sum(-1)

    negative_scores = similarity.clone()
    negative_scores.fill_diagonal_(float("-inf"))
    hardest_negative = negative_scores.max(dim=1).values

    margin = positives - hardest_negative

    return {
        "recall_at_1": float(
            (similarity.argmax(-1) == targets)
            .float().mean().item()
        ),
        "mrr": float(
            (1.0 / ranks.float()).mean().item()
        ),
        "matched_cosine": float(positives.mean().item()),
        "hard_negative_margin": float(margin.mean().item()),
        "positive_margin_fraction": float(
            (margin > 0).float().mean().item()
        ),
    }


print("=" * 70, flush=True)
print("PHASE 4.2.14 - REPRESENTATION STABILITY AUDIT", flush=True)
print("=" * 70, flush=True)

print("\n[1] Verifying provenance...", flush=True)

for path in (
    GRAPH, GRAPH_REPORT, PROTOCOL, ARCH,
    *REPORT_FILES.values(),
):
    if not path.is_file():
        raise RuntimeError(f"Missing source: {path}")

for path in (TABLE, SUMMARY, REPORT):
    if path.exists():
        raise RuntimeError(f"Refusing overwrite: {path}")

graph_hash = sha256(GRAPH)
protocol_hash = sha256(PROTOCOL)
arch_hash = sha256(ARCH)

if read_json(GRAPH_REPORT)["graph_archive_sha256"] != graph_hash:
    raise RuntimeError("Graph provenance mismatch.")

if read_json(PROTOCOL)["source_graph_sha256"] != graph_hash:
    raise RuntimeError("Protocol provenance mismatch.")

architecture = read_json(ARCH)

if (
    architecture["graph_archive_sha256"] != graph_hash
    or architecture["initial_protocol_sha256"] != protocol_hash
):
    raise RuntimeError("Architecture provenance mismatch.")

references = {
    name: read_json(path)
    for name, path in REPORT_FILES.items()
}

for name, reference in references.items():
    if (
        reference["graph_archive_sha256"] != graph_hash
        or reference["protocol_sha256"] != protocol_hash
    ):
        raise RuntimeError(
            f"Historical provenance mismatch: {name}"
        )

    if reference.get("test_evaluated", False):
        raise RuntimeError(
            f"Test evaluation detected: {name}"
        )

if (
    references["ResidualGate05"][
        "parent_architecture_sha256"
    ] != arch_hash
):
    raise RuntimeError("Residual architecture hash mismatch.")


print("\n[2] Preparing paired validation graphs...", flush=True)

with np.load(GRAPH, allow_pickle=False) as archive:
    features = archive["node_features"].copy()
    edges = archive["edge_index"].copy()
    regions = archive["region_id"].astype(str)
    splits = archive["spatial_split"].astype(str)
    scanners = archive["scanner"].astype(str)
    tiles = archive["tile_id"].astype(str)

if (
    features.shape != (120, 36, 384)
    or features.dtype != np.float32
    or not np.isfinite(features).all()
):
    raise RuntimeError("Invalid feature tensor.")

if dict(Counter(splits)) != {
    "train": 80, "validation": 20, "test": 20
}:
    raise RuntimeError("Unexpected graph partitions.")

by_region = defaultdict(dict)

for index, name in enumerate(regions):
    if splits[index] != "validation":
        continue

    scanner = scanners[index]

    if scanner in by_region[name]:
        raise RuntimeError("Duplicate scanner graph.")

    by_region[name][scanner] = index

region_names = sorted(by_region)

if len(region_names) != 10:
    raise RuntimeError("Expected 10 validation regions.")

pairs = []

for region in region_names:
    pair = by_region[region]

    if set(pair) != {"AT2", "GT450"}:
        raise RuntimeError("Incomplete paired region.")

    ai = pair["AT2"]
    bi = pair["GT450"]

    if not np.array_equal(tiles[ai], tiles[bi]):
        raise RuntimeError("Tile correspondence mismatch.")

    pairs.append(
        np.stack([features[ai], features[bi]], axis=0)
    )

validation = torch.from_numpy(
    np.stack(pairs).astype(np.float32)
).to(DEVICE)

if validation.shape != (10, 2, 36, 384):
    raise RuntimeError("Validation dimensions mismatch.")

weights = neighbor_matrix(edges).to(DEVICE)

print("Validation region pairs:", len(validation))
print("Matched tile positions:", len(validation) * 36)
print("Test partition: RESERVED")


print("\n[3] Evaluating frozen and trained models...", flush=True)

rows = []
global_rows = []


@torch.inference_mode()
def record_model(name, seed, model):
    if model is not None:
        model.eval()

    at2 = validation[:, 0]
    gt450 = validation[:, 1]

    if model is not None:
        at2 = model(at2)
        gt450 = model(gt450)

    at2 = F.normalize(at2, dim=-1)
    gt450 = F.normalize(gt450, dim=-1)

    for i, region in enumerate(region_names):
        matching = retrieval_metrics(at2[i], gt450[i])
        reverse = retrieval_metrics(gt450[i], at2[i])

        rows.append({
            "model": name,
            "seed": str(seed),
            "region_id": region,
            **matching,
            "reverse_recall_at_1": reverse["recall_at_1"],
            "at2_offdiagonal_cosine":
                off_diagonal_cosine(at2[i]),
            "gt450_offdiagonal_cosine":
                off_diagonal_cosine(gt450[i]),
            "at2_effective_rank": effective_rank(at2[i]),
            "gt450_effective_rank": effective_rank(gt450[i]),
        })

    a = at2.reshape(-1, 384)
    b = gt450.reshape(-1, 384)

    global_result = retrieval_metrics(a, b)

    global_rows.append({
        "model": name,
        "seed": str(seed),
        **global_result,
    })

    print(
        f"  {name:18s} seed={seed}: "
        f"Global R@1={global_result['recall_at_1']:.4f} "
        f"Matched cosine={global_result['matched_cosine']:.4f}",
        flush=True,
    )


record_model("Frozen DINOv2", "none", None)

for model_name, reference in references.items():
    folder, filename = CHECKPOINT_FILES[model_name]

    for seed in SEEDS:
        checkpoint = (
            BASE / "outputs/checkpoints"
            / folder / filename.format(seed=seed)
        )

        if not checkpoint.is_file():
            raise RuntimeError(f"Missing checkpoint: {checkpoint}")

        expected_hash = reference[
            "checkpoint_hashes"
        ][str(seed)]

        if sha256(checkpoint) != expected_hash:
            raise RuntimeError(
                f"Checkpoint integrity mismatch: {checkpoint}"
            )

        # Checkpoints come from our own completed
        # and hash-verified research scripts.
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        if (
            int(payload["seed"]) != seed
            or payload["graph_archive_sha256"] != graph_hash
            or payload["protocol_sha256"] != protocol_hash
            or payload["test_evaluated"] is not False
        ):
            raise RuntimeError("Checkpoint metadata mismatch.")

        if model_name == "MLP":
            model = SharedMLP()
        elif model_name == "GraphSAGE":
            model = GraphSAGE(weights)
        else:
            if (
                payload["parent_architecture_sha256"]
                != arch_hash
                or payload["initial_gate"] != 0.5
            ):
                raise RuntimeError(
                    "Residual checkpoint specification mismatch."
                )
            model = ResidualGraphSAGE(weights)

        model.load_state_dict(
            payload["model_state_dict"],
            strict=True,
        )

        model = model.to(DEVICE)
        record_model(model_name, seed, model)


print("\n[4] Aggregating stability and diversity metrics...", flush=True)

metric_names = [
    "recall_at_1",
    "mrr",
    "matched_cosine",
    "hard_negative_margin",
    "positive_margin_fraction",
    "reverse_recall_at_1",
    "at2_offdiagonal_cosine",
    "gt450_offdiagonal_cosine",
    "at2_effective_rank",
    "gt450_effective_rank",
]

summary_rows = []

for model_name in (
    "Frozen DINOv2",
    "MLP",
    "GraphSAGE",
    "ResidualGate05",
):
    subset = [
        row for row in rows
        if row["model"] == model_name
    ]

    if not subset:
        raise RuntimeError(
            f"Missing diagnostic model: {model_name}"
        )

    # Each region has equal node count (36),
    # so region-mean metrics have equal query weight.
    aggregated = {
        "model": model_name,
        "runs": len(set(row["seed"] for row in subset)),
        "validation_regions": len(region_names),
    }

    for metric in metric_names:
        aggregated[f"mean_{metric}"] = float(
            np.mean([float(r[metric]) for r in subset])
        )

    global_subset = [
        row for row in global_rows
        if row["model"] == model_name
    ]

    aggregated["mean_global_recall_at_1"] = float(
        np.mean([
            item["recall_at_1"]
            for item in global_subset
        ])
    )

    summary_rows.append(aggregated)

    print(
        f"  {model_name:18s} | "
        f"R@1 {aggregated['mean_recall_at_1']:.4f} | "
        f"Cosine {aggregated['mean_matched_cosine']:.4f} | "
        f"Margin {aggregated['mean_hard_negative_margin']:.4f} | "
        f"Rank AT2 {aggregated['mean_at2_effective_rank']:.2f}",
        flush=True,
    )


print("\n[5] Saving reproducible audit outputs...", flush=True)

save_csv_new(
    TABLE,
    rows,
    ["model", "seed", "region_id"] + metric_names,
)

summary_columns = (
    ["model", "runs", "validation_regions"]
    + [f"mean_{metric}" for metric in metric_names]
    + ["mean_global_recall_at_1"]
)

save_csv_new(
    SUMMARY,
    summary_rows,
    summary_columns,
)

report = {
    "phase": "4.2",
    "step": "4.2.14",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "experiment": "validation_representation_stability_audit",
    "graph_archive_sha256": graph_hash,
    "protocol_sha256": protocol_hash,
    "architecture_sha256": arch_hash,
    "validation_regions": 10,
    "validation_positions": 360,
    "test_evaluated": False,
    "training_performed": False,
    "checkpoint_selection_performed": False,
    "models": summary_rows,
    "region_table_sha256": sha256(TABLE),
    "summary_table_sha256": sha256(SUMMARY),
    "limitations": [
        "Validation-only exploratory comparison",
        "Three seeds are not independent tissue samples",
        "Matched cosine alone can reward collapsed representations",
        "Effective rank is descriptive, not a clinical metric",
        "No independently sampled biological test set",
        "The gate 0.5 choice followed previous validation analysis",
        "Existing test partition remains untouched",
    ],
    "xai_planned": True,
}

save_json_new(REPORT, report)

print("\n" + "=" * 70)
print("PHASE 4.2.14 - STABILITY AUDIT COMPLETED")
print("=" * 70)

print("Models evaluated:", len(summary_rows))
print("Validation regions:", len(region_names))
print("Training performed: NO")
print("Test evaluated: NO")

print("\nSaved:")
print(TABLE)
print(SUMMARY)
print(REPORT)

print("\nPrevious checkpoints: UNCHANGED")
print("Original images: UNCHANGED")
print("XAI: PLANNED")
print("\nSTEP 4.2.14 COMPLETED")
