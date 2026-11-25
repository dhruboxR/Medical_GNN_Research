
"""
PHASE 4.2.4 - SPATIAL GRAPHSAGE BASELINE

GraphSAGE with mean-neighbor aggregation:
    h' = W_self h + W_neighbor mean(neighbors)

Architecture:
    384 -> 256 -> 384
    GELU between layers
    L2-normalized output

Training:
    40 paired training regions
    10 paired validation regions
    10 paired test regions RESERVED
    Seeds: 42, 123, 2026

Use the same contrastive loss, optimizer, batch size,
early stopping and validation selection as the MLP
and GCN reference experiments.

No original images or graphs are modified.
"""

from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import hashlib
import json
import csv
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "phase4_main_research"

GRAPH = BASE / "outputs/features/15_stage_a_spatial_graphs.npz"
GRAPH_REPORT = BASE / "outputs/reports/15_graph_dataset_report.json"
PROTOCOL = BASE / "configs/16_initial_gnn_experimental_protocol.json"

MLP_REPORT = BASE / "outputs/reports/17_mlp_baseline_report.json"
GCN_REPORT = BASE / "outputs/reports/18_gcn_baseline_report.json"

CHECKPOINTS = BASE / "outputs/checkpoints/19_graphsage_baseline"
HISTORY = BASE / "outputs/tables/19_graphsage_training_history.csv"
SUMMARY = BASE / "outputs/tables/19_graphsage_validation_summary.csv"
REPORT = BASE / "outputs/reports/19_graphsage_baseline_report.json"

SEEDS = (42, 123, 2026)
TEMPERATURE = 0.07
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 20
BATCH_REGIONS = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def filehash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def neighbor_matrix(edge_index, nodes=36):
    """
    Row-normalized adjacency WITHOUT self-loops.

    GraphSAGE keeps self-node features separately
    through its own learnable transformation.
    """
    if edge_index.shape != (2, 120):
        raise RuntimeError("Invalid edge shape.")

    matrix = torch.zeros(nodes, nodes, dtype=torch.float32)
    seen = set()

    for source, target in edge_index.T.tolist():
        if source == target or not (
            0 <= source < nodes and 0 <= target < nodes
        ):
            raise RuntimeError("Invalid graph edge.")

        if (source, target) in seen:
            raise RuntimeError("Duplicate graph edge.")

        seen.add((source, target))
        matrix[target, source] = 1.0

    if len(seen) != 120:
        raise RuntimeError("Unexpected number of edges.")

    if any((b, a) not in seen for a, b in seen):
        raise RuntimeError("Asymmetric spatial edges.")

    degrees = matrix.sum(-1, keepdim=True)

    if torch.any(degrees == 0):
        raise RuntimeError("Isolated node.")

    return matrix / degrees


class SAGEConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neighbor_linear = nn.Linear(
            in_dim, out_dim, bias=False
        )

    def forward(self, x, neighbor_weights):
        neighbor_mean = torch.matmul(neighbor_weights, x)
        return (
            self.self_linear(x)
            + self.neighbor_linear(neighbor_mean)
        )


class SpatialGraphSAGE(nn.Module):
    def __init__(self, neighbor_weights):
        super().__init__()

        self.register_buffer(
            "neighbor_weights", neighbor_weights.clone()
        )
        self.conv1 = SAGEConv(384, 256)
        self.conv2 = SAGEConv(256, 384)

    def forward(self, x):
        hidden = F.gelu(
            self.conv1(x, self.neighbor_weights)
        )
        output = self.conv2(
            hidden, self.neighbor_weights
        )
        return F.normalize(output, p=2, dim=-1)


def loss_fn(a, b):
    logits = (
        a @ b.transpose(-1, -2)
    ) / TEMPERATURE

    batch, nodes, _ = logits.shape
    labels = torch.arange(
        nodes, device=logits.device
    ).expand(batch, nodes).reshape(-1)

    return 0.5 * (
        F.cross_entropy(logits.reshape(-1, nodes), labels)
        + F.cross_entropy(
            logits.transpose(-1, -2).reshape(-1, nodes),
            labels
        )
    )


def off_diagonal_cosine(z):
    similarity = z @ z.transpose(-1, -2)
    n = similarity.shape[-1]

    mask = ~torch.eye(
        n, device=z.device, dtype=torch.bool
    )

    return float(similarity[..., mask].mean().item())


@torch.inference_mode()
def evaluate(model, data):
    model.eval()

    tensor = torch.as_tensor(
        data, dtype=torch.float32, device=DEVICE
    )

    a = model(tensor[:, 0])
    b = model(tensor[:, 1])

    similarity = a @ b.transpose(-1, -2)
    reverse = similarity.transpose(-1, -2)

    batch, n, _ = similarity.shape
    target = torch.arange(
        n, device=DEVICE
    ).expand(batch, n)

    # Strict greater-than ranking, consistent
    # with the MLP and GCN reference scripts.
    positives = similarity.diagonal(dim1=-2, dim2=-1)

    rank_forward = 1 + (
        similarity > positives.unsqueeze(-1)
    ).sum(-1)

    rank_reverse = 1 + (
        reverse > positives.unsqueeze(-1)
    ).sum(-1)

    return {
        "loss": float(loss_fn(a, b).item()),
        "recall_at_1": float(
            (similarity.argmax(-1) == target)
            .float().mean().item()
        ),
        "mrr": float(
            (1.0 / rank_forward.float()).mean().item()
        ),
        "reverse_recall_at_1": float(
            (reverse.argmax(-1) == target)
            .float().mean().item()
        ),
        "reverse_mrr": float(
            (1.0 / rank_reverse.float()).mean().item()
        ),
        "median_positive_cosine": float(
            positives.median().item()
        ),
        "at2_offdiagonal_cosine": off_diagonal_cosine(a),
        "gt450_offdiagonal_cosine": off_diagonal_cosine(b),
    }


def better(candidate, best):
    if best is None:
        return True

    difference = (
        candidate["recall_at_1"] - best["recall_at_1"]
    )

    return (
        difference > 1e-10
        or (
            abs(difference) <= 1e-10
            and candidate["loss"] < best["loss"] - 1e-8
        )
    )


print("=" * 70, flush=True)
print("PHASE 4.2.4 - GRAPHSAGE BASELINE", flush=True)
print("=" * 70, flush=True)

print("\n[1] Validating sources...", flush=True)

for path in (
    GRAPH, GRAPH_REPORT, PROTOCOL,
    MLP_REPORT, GCN_REPORT
):
    if not path.is_file():
        raise RuntimeError(f"Missing input: {path}")

for path in (HISTORY, SUMMARY, REPORT):
    if path.exists():
        raise RuntimeError(f"Output already exists: {path}")

CHECKPOINTS.mkdir(parents=True, exist_ok=True)

for seed in SEEDS:
    if (CHECKPOINTS / f"graphsage_seed_{seed}.pt").exists():
        raise RuntimeError(
            f"Checkpoint already exists for seed {seed}."
        )

graph_hash = filehash(GRAPH)
protocol_hash = filehash(PROTOCOL)
protocol = load_json(PROTOCOL)

if graph_hash != load_json(GRAPH_REPORT)["graph_archive_sha256"]:
    raise RuntimeError("Graph provenance mismatch.")

if graph_hash != protocol["source_graph_sha256"]:
    raise RuntimeError("Protocol provenance mismatch.")

if "spatial_graphsage" not in protocol["initial_baselines"]:
    raise RuntimeError("GraphSAGE is missing from protocol.")

mlp = load_json(MLP_REPORT)
gcn = load_json(GCN_REPORT)

for reference in (mlp, gcn):
    if (
        reference["graph_archive_sha256"] != graph_hash
        or reference["protocol_sha256"] != protocol_hash
    ):
        raise RuntimeError("Baseline provenance mismatch.")

with np.load(GRAPH, allow_pickle=False) as z:
    features = z["node_features"].copy()
    edges = z["edge_index"].copy()
    regions = z["region_id"].astype(str)
    splits = z["spatial_split"].astype(str)
    scanners = z["scanner"].astype(str)
    tile_ids = z["tile_id"].astype(str)

if features.shape != (120, 36, 384):
    raise RuntimeError("Unexpected feature shape.")

if features.dtype != np.float32 or not np.isfinite(features).all():
    raise RuntimeError("Invalid features.")

neighbor_weights = neighbor_matrix(edges).to(DEVICE)

print("Graph tensor:", features.shape)
print("Spatial edges:", edges.shape)
print("Device:", DEVICE)


print("\n[2] Preparing paired spatial partitions...", flush=True)

by_region = defaultdict(dict)
region_split = {}

for i, region in enumerate(regions):
    scanner = scanners[i]
    split = splits[i]

    if scanner not in ("AT2", "GT450"):
        raise RuntimeError("Unexpected scanner.")

    if split not in ("train", "validation", "test"):
        raise RuntimeError("Unexpected split.")

    if scanner in by_region[region]:
        raise RuntimeError("Duplicate region/scanner graph.")

    by_region[region][scanner] = i

    if region in region_split and region_split[region] != split:
        raise RuntimeError("Region crosses partitions.")

    region_split[region] = split

if dict(Counter(region_split.values())) != {
    "train": 40,
    "validation": 10,
    "test": 10,
}:
    raise RuntimeError("Unexpected region split counts.")

paired = {"train": [], "validation": []}

for region in sorted(by_region):
    pair = by_region[region]

    if set(pair) != {"AT2", "GT450"}:
        raise RuntimeError("Incomplete scanner pair.")

    a, b = pair["AT2"], pair["GT450"]

    if not np.array_equal(tile_ids[a], tile_ids[b]):
        raise RuntimeError("Node-order mismatch.")

    split = region_split[region]

    if split == "test":
        continue

    paired[split].append(
        np.stack([features[a], features[b]], axis=0)
    )

train = np.stack(paired["train"]).astype(np.float32)
validation = np.stack(paired["validation"]).astype(np.float32)

if train.shape != (40, 2, 36, 384):
    raise RuntimeError("Training data shape mismatch.")

if validation.shape != (10, 2, 36, 384):
    raise RuntimeError("Validation data shape mismatch.")

print("Training pairs:", len(train))
print("Validation pairs:", len(validation))
print("Test pairs: 10, reserved")


print("\n[3] Training GraphSAGE...", flush=True)

history = []
summaries = []
checkpoint_hashes = {}

for seed in SEEDS:
    print(f"\nSeed: {seed}", flush=True)
    seed_all(seed)

    model = SpatialGraphSAGE(neighbor_weights).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    rng = np.random.default_rng(seed)
    best = None
    best_weights = None
    best_epoch = None
    stale = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        permutation = rng.permutation(len(train))

        total_loss = 0.0
        samples_seen = 0

        for start in range(0, len(train), BATCH_REGIONS):
            indices = permutation[start:start + BATCH_REGIONS]

            batch = torch.as_tensor(
                train[indices],
                dtype=torch.float32,
                device=DEVICE,
            )

            a = model(batch[:, 0])
            b = model(batch[:, 1])

            loss = loss_fn(a, b)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(indices)
            samples_seen += len(indices)

        train_loss = total_loss / samples_seen
        metrics = evaluate(model, validation)

        improved = better(metrics, best)

        if improved:
            best = dict(metrics)
            best_epoch = epoch
            best_weights = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1

        history.append({
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": metrics["loss"],
            "validation_recall_at_1": metrics["recall_at_1"],
            "validation_mrr": metrics["mrr"],
            "validation_reverse_recall_at_1":
                metrics["reverse_recall_at_1"],
            "validation_reverse_mrr": metrics["reverse_mrr"],
            "validation_at2_offdiagonal_cosine":
                metrics["at2_offdiagonal_cosine"],
            "validation_gt450_offdiagonal_cosine":
                metrics["gt450_offdiagonal_cosine"],
            "best_checkpoint": improved,
        })

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  Epoch {epoch:03d} | "
                f"Train loss {train_loss:.4f} | "
                f"Val R@1 {metrics['recall_at_1']:.4f} | "
                f"MRR {metrics['mrr']:.4f}",
                flush=True,
            )

        if stale >= PATIENCE:
            print(f"  Early stopping: epoch {epoch}", flush=True)
            break

    if best_weights is None:
        raise RuntimeError("No best checkpoint selected.")

    model.load_state_dict(best_weights)
    final = evaluate(model, validation)

    checkpoint = CHECKPOINTS / f"graphsage_seed_{seed}.pt"

    with checkpoint.open("xb") as f:
        torch.save({
            "model_state_dict": best_weights,
            "model_name": "spatial_graphsage",
            "architecture": "384-256-384",
            "aggregation": "mean_without_self_loops",
            "seed": seed,
            "best_epoch": best_epoch,
            "validation_metrics": final,
            "graph_archive_sha256": graph_hash,
            "protocol_sha256": protocol_hash,
            "test_evaluated": False,
        }, f)

    checkpoint_hashes[str(seed)] = filehash(checkpoint)

    summaries.append({
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_loss": final["loss"],
        "validation_recall_at_1": final["recall_at_1"],
        "validation_mrr": final["mrr"],
        "validation_reverse_recall_at_1":
            final["reverse_recall_at_1"],
        "validation_reverse_mrr": final["reverse_mrr"],
        "validation_at2_offdiagonal_cosine":
            final["at2_offdiagonal_cosine"],
        "validation_gt450_offdiagonal_cosine":
            final["gt450_offdiagonal_cosine"],
        "checkpoint_sha256": checkpoint_hashes[str(seed)],
    })

    print(
        f"  Best epoch {best_epoch} | "
        f"Validation R@1 {final['recall_at_1']:.4f}",
        flush=True,
    )


print("\n[4] Saving validation results...", flush=True)

history_columns = [
    "seed", "epoch", "train_loss", "validation_loss",
    "validation_recall_at_1", "validation_mrr",
    "validation_reverse_recall_at_1", "validation_reverse_mrr",
    "validation_at2_offdiagonal_cosine",
    "validation_gt450_offdiagonal_cosine", "best_checkpoint",
]

summary_columns = [
    "seed", "best_epoch", "validation_loss",
    "validation_recall_at_1", "validation_mrr",
    "validation_reverse_recall_at_1", "validation_reverse_mrr",
    "validation_at2_offdiagonal_cosine",
    "validation_gt450_offdiagonal_cosine", "checkpoint_sha256",
]

save_csv(HISTORY, history, history_columns)
save_csv(SUMMARY, summaries, summary_columns)

scores = np.array(
    [row["validation_recall_at_1"] for row in summaries]
)

sage_mean = float(scores.mean())
mlp_mean = float(mlp["mean_validation_recall_at_1"])
gcn_mean = float(gcn["mean_validation_recall_at_1"])

report = {
    "phase": "4.2",
    "step": "4.2.4",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "experiment": "spatial_graphsage_baseline",
    "architecture": "GraphSAGE_384_256_384_GELU",
    "aggregator": "mean_neighbors",
    "self_features": "separate_linear_transform",
    "graph_message_passing": True,
    "node_output_normalization": "L2",
    "objective": "bidirectional_within_region_InfoNCE",
    "temperature": TEMPERATURE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "batch_regions": BATCH_REGIONS,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
    "seeds": list(SEEDS),
    "graph_archive_sha256": graph_hash,
    "protocol_sha256": protocol_hash,
    "train_region_pairs": 40,
    "validation_region_pairs": 10,
    "test_region_pairs_reserved": 10,
    "test_evaluated": False,
    "checkpoint_selection":
        "validation_recall_at_1_then_loss",
    "seed_results": summaries,
    "mean_validation_recall_at_1": sage_mean,
    "std_validation_recall_at_1": float(
        scores.std(ddof=1)
    ),
    "mlp_mean_validation_recall_at_1": mlp_mean,
    "gcn_mean_validation_recall_at_1": gcn_mean,
    "sage_minus_mlp_percentage_points":
        (sage_mean - mlp_mean) * 100,
    "sage_minus_gcn_percentage_points":
        (sage_mean - gcn_mean) * 100,
    "history_sha256": filehash(HISTORY),
    "summary_sha256": filehash(SUMMARY),
    "checkpoint_hashes": checkpoint_hashes,
    "biological_independence_verified": False,
    "xai_planned": True,
    "limitations": [
        "Spatial splits are not biologically independent",
        "Only 10 validation regions",
        "36 candidates per retrieval query",
        "GraphSAGE has more parameters than the reference MLP",
        "Node cosine alone cannot establish oversmoothing",
        "No test evaluation performed",
    ],
}

save_json(REPORT, report)

print("\n" + "=" * 70)
print("PHASE 4.2.4 - GRAPHSAGE BASELINE COMPLETED")
print("=" * 70)

for row in summaries:
    print(
        f"Seed {row['seed']} | "
        f"Best epoch {row['best_epoch']} | "
        f"Val R@1 {row['validation_recall_at_1']:.4f} | "
        f"MRR {row['validation_mrr']:.4f}"
    )

print(f"\nGraphSAGE mean R@1: {sage_mean:.4f}")
print(f"GCN mean R@1: {gcn_mean:.4f}")
print(f"MLP mean R@1: {mlp_mean:.4f}")

print(
    f"GraphSAGE vs MLP: "
    f"{report['sage_minus_mlp_percentage_points']:+.2f} pp"
)
print(
    f"GraphSAGE vs GCN: "
    f"{report['sage_minus_gcn_percentage_points']:+.2f} pp"
)

print("\nSaved:")
print(HISTORY)
print(SUMMARY)
print(REPORT)

print("\nTest partition: NOT EVALUATED")
print("Original images and features: UNCHANGED")
print("Proposed stability-aware GNN: NOT YET TRAINED")
print("XAI: PLANNED")
print("\nSTEP 4.2.4 COMPLETED")
