
"""
PHASE 4.2.3 - SPATIAL GCN BASELINE

Comparison:
    Phase 4.2.2 shared MLP (no graph edges)
    versus
    Two-layer spatial GCN (uses graph edges)

Architecture:
    GCNConv(384, 256)
    GELU
    GCNConv(256, 384)
    L2-normalized output

GCN implementation:
    Symmetric degree-normalized adjacency
    Self-loops included
    Two message-passing layers
    No residual connections or batch normalization

Training:
    Shared model for AT2 and GT450
    Bidirectional within-region InfoNCE
    Seeds: 42, 123, 2026
    Training regions: 40
    Validation regions: 10
    Test regions: never evaluated

Outputs:
    One best checkpoint per seed
    Epoch history
    Validation summary
    Reproducibility report

This is a reference GCN, not the proposed method.
"""

from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "phase4_main_research"

GRAPH_PATH = (
    BASE / "outputs/features"
    / "15_stage_a_spatial_graphs.npz"
)

GRAPH_REPORT = (
    BASE / "outputs/reports"
    / "15_graph_dataset_report.json"
)

PROTOCOL_PATH = (
    BASE / "configs"
    / "16_initial_gnn_experimental_protocol.json"
)

MLP_REPORT = (
    BASE / "outputs/reports"
    / "17_mlp_baseline_report.json"
)

CHECKPOINT_DIR = (
    BASE / "outputs/checkpoints"
    / "18_gcn_baseline"
)

HISTORY_PATH = (
    BASE / "outputs/tables"
    / "18_gcn_training_history.csv"
)

SUMMARY_PATH = (
    BASE / "outputs/tables"
    / "18_gcn_validation_summary.csv"
)

REPORT_PATH = (
    BASE / "outputs/reports"
    / "18_gcn_baseline_report.json"
)

SEEDS = (42, 123, 2026)

INPUT_DIM = 384
HIDDEN_DIM = 256
OUTPUT_DIM = 384

TEMPERATURE = 0.07
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 100
PATIENCE = 20
BATCH_REGIONS = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def save_csv_new(path, rows, fields):
    with path.open(
        "x", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_json_new(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(
            True, warn_only=True
        )


def normalized_adjacency(edge_index, nodes=36):
    """
    Build A_hat = D^-1/2 (A + I) D^-1/2.

    Stored tensor convention:
        adjacency[target, source]

    The source graph already contains reciprocal
    directed edges, without self-loops.
    """

    if edge_index.shape != (2, 120):
        raise RuntimeError("Invalid edge-index shape.")

    adjacency = torch.zeros(
        (nodes, nodes), dtype=torch.float32
    )

    edges = set()

    for source, target in edge_index.T:
        source = int(source)
        target = int(target)

        if not (0 <= source < nodes and 0 <= target < nodes):
            raise RuntimeError("Invalid edge endpoint.")

        if source == target:
            raise RuntimeError("Unexpected source self-loop.")

        if (source, target) in edges:
            raise RuntimeError("Duplicate source edge.")

        edges.add((source, target))
        adjacency[target, source] = 1.0

    if len(edges) != 120:
        raise RuntimeError("Unexpected edge count.")

    if any(
        (target, source) not in edges
        for source, target in edges
    ):
        raise RuntimeError("Missing reciprocal edge.")

    adjacency += torch.eye(nodes)

    degree = adjacency.sum(dim=1)
    inverse_sqrt = degree.pow(-0.5)

    normalized = (
        inverse_sqrt[:, None]
        * adjacency
        * inverse_sqrt[None, :]
    )

    if not torch.isfinite(normalized).all():
        raise RuntimeError("Invalid adjacency normalization.")

    return normalized


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(
            input_dim,
            output_dim,
            bias=True,
        )

    def forward(self, x, adjacency):
        transformed = self.linear(x)

        return torch.matmul(
            adjacency,
            transformed,
        )


class SpatialGCN(nn.Module):
    def __init__(self, adjacency):
        super().__init__()

        self.register_buffer(
            "adjacency",
            adjacency.clone(),
        )

        self.layer1 = GCNLayer(
            INPUT_DIM,
            HIDDEN_DIM,
        )

        self.layer2 = GCNLayer(
            HIDDEN_DIM,
            OUTPUT_DIM,
        )

    def forward(self, x):
        hidden = self.layer1(
            x, self.adjacency
        )

        hidden = F.gelu(hidden)

        output = self.layer2(
            hidden, self.adjacency
        )

        return F.normalize(
            output,
            p=2,
            dim=-1,
        )


def contrastive_loss(a, b):
    logits = (
        a @ b.transpose(-1, -2)
    ) / TEMPERATURE

    batch_size, nodes, _ = logits.shape

    labels = torch.arange(
        nodes,
        device=logits.device,
    ).expand(batch_size, -1)

    forward = F.cross_entropy(
        logits.reshape(-1, nodes),
        labels.reshape(-1),
    )

    reverse = F.cross_entropy(
        logits.transpose(1, 2).reshape(-1, nodes),
        labels.reshape(-1),
    )

    return 0.5 * (forward + reverse)


def node_diversity(z):
    """
    Mean off-diagonal cosine similarity.

    Larger values indicate greater average node
    similarity, potentially associated with smoothing.

    This is descriptive and not sufficient alone
    to diagnose oversmoothing.
    """

    similarities = (
        z @ z.transpose(-1, -2)
    )

    nodes = similarities.size(-1)

    mask = ~torch.eye(
        nodes,
        dtype=torch.bool,
        device=z.device,
    )

    values = similarities[..., mask]

    return float(values.mean().item())


@torch.no_grad()
def evaluate(model, paired_data):
    model.eval()

    batch = torch.as_tensor(
        paired_data,
        dtype=torch.float32,
        device=DEVICE,
    )

    za = model(batch[:, 0])
    zb = model(batch[:, 1])

    logits = (
        za @ zb.transpose(-1, -2)
    ) / TEMPERATURE

    loss = contrastive_loss(
        za, zb
    )

    similarity = (
        za @ zb.transpose(-1, -2)
    )

    nodes = similarity.shape[-1]

    labels = torch.arange(
        nodes,
        device=DEVICE,
    ).expand(similarity.shape[0], -1)

    positive = similarity.diagonal(
        dim1=-2,
        dim2=-1,
    )

    rank_forward = 1 + (
        similarity > positive.unsqueeze(-1)
    ).sum(dim=-1)

    transposed = similarity.transpose(1, 2)

    rank_reverse = 1 + (
        transposed > positive.unsqueeze(-1)
    ).sum(dim=-1)

    forward_correct = (
        similarity.argmax(dim=-1) == labels
    )

    reverse_correct = (
        transposed.argmax(dim=-1) == labels
    )

    return {
        "loss": float(loss.item()),
        "recall_at_1": float(
            forward_correct.float().mean().item()
        ),
        "mrr": float(
            (1.0 / rank_forward.float()).mean().item()
        ),
        "reverse_recall_at_1": float(
            reverse_correct.float().mean().item()
        ),
        "reverse_mrr": float(
            (1.0 / rank_reverse.float()).mean().item()
        ),
        "median_positive_cosine": float(
            positive.median().item()
        ),
        "at2_offdiagonal_cosine": node_diversity(za),
        "gt450_offdiagonal_cosine": node_diversity(zb),
    }


def is_better(candidate, current_best):
    if current_best is None:
        return True

    if (
        candidate["recall_at_1"]
        > current_best["recall_at_1"] + 1e-10
    ):
        return True

    if (
        abs(
            candidate["recall_at_1"]
            - current_best["recall_at_1"]
        ) <= 1e-10
        and candidate["loss"] < current_best["loss"] - 1e-8
    ):
        return True

    return False


print("=" * 70, flush=True)
print("PHASE 4.2.3 - SPATIAL GCN BASELINE", flush=True)
print("=" * 70)


# --------------------------------------------------
# 1. Validate experimental provenance
# --------------------------------------------------

print("\n[1] Validating graph and protocol...", flush=True)

for path in (
    GRAPH_PATH,
    GRAPH_REPORT,
    PROTOCOL_PATH,
    MLP_REPORT,
):
    if not path.is_file():
        raise RuntimeError(f"Missing input: {path}")

graph_report = read_json(GRAPH_REPORT)
protocol = read_json(PROTOCOL_PATH)
mlp_report = read_json(MLP_REPORT)

graph_hash = sha256(GRAPH_PATH)
protocol_hash = sha256(PROTOCOL_PATH)

if graph_hash != graph_report["graph_archive_sha256"]:
    raise RuntimeError("Graph provenance mismatch.")

if graph_hash != protocol["source_graph_sha256"]:
    raise RuntimeError("Protocol graph mismatch.")

if "spatial_gcn" not in protocol["initial_baselines"]:
    raise RuntimeError("GCN baseline not in protocol.")

if (
    mlp_report["graph_archive_sha256"] != graph_hash
    or mlp_report["protocol_sha256"] != protocol_hash
):
    raise RuntimeError("MLP reference provenance mismatch.")

for path in (
    HISTORY_PATH,
    SUMMARY_PATH,
    REPORT_PATH,
):
    if path.exists():
        raise RuntimeError(
            f"Refusing to overwrite output: {path}"
        )

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

checkpoint_paths = {
    seed: CHECKPOINT_DIR / f"gcn_seed_{seed}.pt"
    for seed in SEEDS
}

if any(path.exists() for path in checkpoint_paths.values()):
    raise RuntimeError(
        "Existing GCN checkpoint found. "
        "Refusing to overwrite research results."
    )

with np.load(GRAPH_PATH, allow_pickle=False) as archive:
    features = archive["node_features"].copy()
    edges = archive["edge_index"].copy()

    regions = archive["region_id"].astype(str)
    splits = archive["spatial_split"].astype(str)
    scanners = archive["scanner"].astype(str)
    tile_ids = archive["tile_id"].astype(str)

if features.shape != (120, 36, 384):
    raise RuntimeError("Invalid graph feature tensor.")

if features.dtype != np.float32:
    raise RuntimeError("Features must be float32.")

if not np.isfinite(features).all():
    raise RuntimeError("Non-finite source features.")

adjacency = normalized_adjacency(edges).to(DEVICE)

print("Graphs:", len(features))
print("Feature shape:", features.shape)
print("Spatial directed edges:", edges.shape[1])
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# 2. Construct training and validation pairs
# --------------------------------------------------

print("\n[2] Preparing region-level partitions...", flush=True)

by_region = defaultdict(dict)
region_splits = {}

for index, region in enumerate(regions):
    scanner = scanners[index]
    split = splits[index]

    if scanner not in ("AT2", "GT450"):
        raise RuntimeError("Unexpected scanner.")

    if split not in ("train", "validation", "test"):
        raise RuntimeError("Unexpected partition.")

    if scanner in by_region[region]:
        raise RuntimeError("Duplicate scanner graph.")

    by_region[region][scanner] = index

    if region in region_splits and region_splits[region] != split:
        raise RuntimeError("Region crosses partitions.")

    region_splits[region] = split

if dict(Counter(region_splits.values())) != {
    "train": 40,
    "validation": 10,
    "test": 10,
}:
    raise RuntimeError("Unexpected region split counts.")

paired_data = {
    "train": [],
    "validation": [],
}

for region in sorted(by_region):
    pair = by_region[region]

    if set(pair) != {"AT2", "GT450"}:
        raise RuntimeError("Incomplete graph pair.")

    a = pair["AT2"]
    b = pair["GT450"]

    if not np.array_equal(tile_ids[a], tile_ids[b]):
        raise RuntimeError("Paired node ordering mismatch.")

    split = region_splits[region]

    if split == "test":
        continue

    paired_data[split].append(
        np.stack(
            (features[a], features[b]),
            axis=0,
        )
    )

train_data = np.stack(
    paired_data["train"]
).astype(np.float32)

validation_data = np.stack(
    paired_data["validation"]
).astype(np.float32)

if train_data.shape != (40, 2, 36, 384):
    raise RuntimeError("Invalid training data shape.")

if validation_data.shape != (10, 2, 36, 384):
    raise RuntimeError("Invalid validation data shape.")

print("Training graph pairs:", len(train_data))
print("Validation graph pairs:", len(validation_data))
print("Test graph pairs: 10 (reserved)")


# --------------------------------------------------
# 3. Train three independent seeds
# --------------------------------------------------

print("\n[3] Training spatial GCN...", flush=True)

history = []
summaries = []
checkpoint_hashes = {}

for seed in SEEDS:
    print(f"\nRandom seed: {seed}", flush=True)

    set_seed(seed)

    model = SpatialGCN(adjacency).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    rng = np.random.default_rng(seed)

    best_metrics = None
    best_weights = None
    best_epoch = 0
    stale = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        shuffled = rng.permutation(
            len(train_data)
        )

        total_loss = 0.0
        seen = 0

        for start in range(
            0,
            len(shuffled),
            BATCH_REGIONS,
        ):
            selected = shuffled[
                start:start + BATCH_REGIONS
            ]

            batch = torch.as_tensor(
                train_data[selected],
                dtype=torch.float32,
                device=DEVICE,
            )

            za = model(batch[:, 0])
            zb = model(batch[:, 1])

            loss = contrastive_loss(za, zb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += (
                float(loss.item()) * len(selected)
            )
            seen += len(selected)

        train_loss = total_loss / seen

        metrics = evaluate(
            model,
            validation_data,
        )

        improved = is_better(
            metrics,
            best_metrics,
        )

        if improved:
            best_metrics = dict(metrics)
            best_epoch = epoch

            best_weights = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
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
            "validation_reverse_mrr":
                metrics["reverse_mrr"],
            "validation_at2_offdiagonal_cosine":
                metrics["at2_offdiagonal_cosine"],
            "validation_gt450_offdiagonal_cosine":
                metrics["gt450_offdiagonal_cosine"],
            "best_checkpoint": improved,
        })

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  Epoch {epoch:03d} | "
                f"Loss {train_loss:.4f} | "
                f"Val R@1 {metrics['recall_at_1']:.4f} | "
                f"MRR {metrics['mrr']:.4f}",
                flush=True,
            )

        if stale >= PATIENCE:
            print(
                f"  Early stopping at epoch {epoch}",
                flush=True,
            )
            break

    if best_weights is None:
        raise RuntimeError("No valid GCN checkpoint.")

    model.load_state_dict(best_weights)

    final_validation = evaluate(
        model, validation_data
    )

    payload = {
        "model_state_dict": best_weights,
        "model_name": "spatial_gcn",
        "architecture": "384-256-384",
        "topology": "spatial_4_neighbor",
        "edge_normalization": "symmetric_with_self_loops",
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_metrics": final_validation,
        "graph_archive_sha256": graph_hash,
        "protocol_sha256": protocol_hash,
        "test_evaluated": False,
    }

    checkpoint = checkpoint_paths[seed]

    with checkpoint.open("xb") as stream:
        torch.save(payload, stream)

    checkpoint_hashes[str(seed)] = sha256(checkpoint)

    summaries.append({
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_loss": final_validation["loss"],
        "validation_recall_at_1":
            final_validation["recall_at_1"],
        "validation_mrr": final_validation["mrr"],
        "validation_reverse_recall_at_1":
            final_validation["reverse_recall_at_1"],
        "validation_reverse_mrr":
            final_validation["reverse_mrr"],
        "validation_at2_offdiagonal_cosine":
            final_validation["at2_offdiagonal_cosine"],
        "validation_gt450_offdiagonal_cosine":
            final_validation["gt450_offdiagonal_cosine"],
        "checkpoint_sha256": checkpoint_hashes[str(seed)],
    })

    print(
        f"  Best epoch: {best_epoch} | "
        f"Val R@1: {final_validation['recall_at_1']:.4f}",
        flush=True,
    )


# --------------------------------------------------
# 4. Save results and MLP comparison
# --------------------------------------------------

print("\n[4] Saving reproducible GCN results...", flush=True)

HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

save_csv_new(
    HISTORY_PATH,
    history,
    [
        "seed", "epoch", "train_loss",
        "validation_loss", "validation_recall_at_1",
        "validation_mrr",
        "validation_reverse_recall_at_1",
        "validation_reverse_mrr",
        "validation_at2_offdiagonal_cosine",
        "validation_gt450_offdiagonal_cosine",
        "best_checkpoint",
    ],
)

save_csv_new(
    SUMMARY_PATH,
    summaries,
    [
        "seed", "best_epoch",
        "validation_loss",
        "validation_recall_at_1",
        "validation_mrr",
        "validation_reverse_recall_at_1",
        "validation_reverse_mrr",
        "validation_at2_offdiagonal_cosine",
        "validation_gt450_offdiagonal_cosine",
        "checkpoint_sha256",
    ],
)

scores = np.array(
    [
        r["validation_recall_at_1"]
        for r in summaries
    ],
    dtype=np.float64,
)

mlp_mean = float(
    mlp_report["mean_validation_recall_at_1"]
)

gcn_mean = float(scores.mean())

report = {
    "phase": "4.2",
    "step": "4.2.3",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "experiment": "spatial_gcn_baseline",
    "architecture": "GCN_384_256_384_GELU",
    "graph_message_passing": True,
    "graph_topology": "spatial_4_neighbor",
    "normalization": "symmetric_degree_with_self_loops",
    "model_input": "frozen_raw_dinov2_cls",
    "output_normalization": "L2",
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
    "mean_validation_recall_at_1": gcn_mean,
    "std_validation_recall_at_1": float(
        scores.std(ddof=1)
    ),
    "mlp_mean_validation_recall_at_1": mlp_mean,
    "gcn_minus_mlp_percentage_points":
        (gcn_mean - mlp_mean) * 100.0,
    "history_sha256": sha256(HISTORY_PATH),
    "summary_sha256": sha256(SUMMARY_PATH),
    "checkpoint_hashes": checkpoint_hashes,
    "test_data_used_for_training": False,
    "biological_independence_verified": False,
    "xai_planned": True,
    "limitations": [
        "Single biological source context",
        "Small spatial validation set",
        "Only two scanners",
        "Graph propagation may smooth discriminative node features",
        "Mean off-diagonal cosine is only one smoothing diagnostic",
        "No final test evaluation performed",
    ],
}

save_json_new(REPORT_PATH, report)

print("\n" + "=" * 70)
print("PHASE 4.2.3 - GCN BASELINE COMPLETED")
print("=" * 70)

for item in summaries:
    print(
        f"Seed {item['seed']} | "
        f"Best epoch {item['best_epoch']} | "
        f"Val R@1 {item['validation_recall_at_1']:.4f} | "
        f"MRR {item['validation_mrr']:.4f}"
    )

print(
    "\nGCN mean validation Recall@1:",
    f"{gcn_mean:.4f}",
)

print(
    "MLP mean validation Recall@1:",
    f"{mlp_mean:.4f}",
)

print(
    "GCN - MLP:",
    f"{report['gcn_minus_mlp_percentage_points']:+.2f} pp",
)

print("\nSaved:")
print(HISTORY_PATH)
print(SUMMARY_PATH)
print(REPORT_PATH)

for path in checkpoint_paths.values():
    print(path)

print("\nTest partition: NOT EVALUATED")
print("Original dataset: UNCHANGED")
print("Proposed stability-aware GNN: NOT YET TRAINED")
print("XAI: PLANNED")
print("\nSTEP 4.2.3 COMPLETED")
