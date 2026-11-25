
"""
PHASE 4.2.7 - SELF-ONLY GRAPHSAGE CONTROL

Purpose:
    Determine whether training GraphSAGE without
    neighborhood messages retains its retrieval accuracy.

Architecture:
    Two SAGEConv layers: 384 -> 256 -> 384
    GELU and L2 normalization
    Neighbor matrix fixed to zero during training and eval

Important:
    Neighbor transformation parameters exist but receive
    no effective gradient when neighbor messages are zero.
    This is an architecture-path control, not a
    parameter-count-matched active-parameter control.

Training:
    Three seeds: 42, 123, 2026
    Train: 40 region pairs
    Validation: 10 region pairs
    Test: reserved, not evaluated
    Same Phase 4.2.4 hyperparameters

Outputs:
    22_self_only_training_history.csv
    22_self_only_validation_summary.csv
    22_self_only_graphsage_report.json
    Three checkpoint files

Never modifies previous artifacts.
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


BASE = (
    Path.home()
    / "MEDICAL_GNN_RESEARCH"
    / "phase4_main_research"
)

GRAPH = BASE / "outputs/features/15_stage_a_spatial_graphs.npz"
GRAPH_REPORT = BASE / "outputs/reports/15_graph_dataset_report.json"
PROTOCOL = BASE / "configs/16_initial_gnn_experimental_protocol.json"

SAGE_REPORT = BASE / "outputs/reports/19_graphsage_baseline_report.json"
PERM_REPORT = BASE / "outputs/reports/21_permuted_topology_report.json"

CHECKPOINT_DIR = BASE / "outputs/checkpoints/22_self_only_graphsage"
HISTORY = BASE / "outputs/tables/22_self_only_training_history.csv"
SUMMARY = BASE / "outputs/tables/22_self_only_validation_summary.csv"
REPORT = BASE / "outputs/reports/22_self_only_graphsage_report.json"

SEEDS = (42, 123, 2026)
TEMPERATURE = 0.07
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 20
BATCH_REGIONS = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(
        True, warn_only=True
    )


class SAGEConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neighbor_linear = nn.Linear(
            in_dim, out_dim, bias=False
        )

    def forward(self, x, neighbor_weights):
        return (
            self.self_linear(x)
            + self.neighbor_linear(
                torch.matmul(neighbor_weights, x)
            )
        )


class SelfOnlyGraphSAGE(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer(
            "neighbor_weights",
            torch.zeros(36, 36, dtype=torch.float32)
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


def contrastive_loss(a, b):
    logits = (
        a @ b.transpose(-1, -2)
    ) / TEMPERATURE

    batch, nodes, _ = logits.shape

    labels = torch.arange(
        nodes, device=logits.device
    ).expand(batch, nodes).reshape(-1)

    forward = F.cross_entropy(
        logits.reshape(-1, nodes), labels
    )

    reverse = F.cross_entropy(
        logits.transpose(-1, -2).reshape(-1, nodes),
        labels
    )

    return 0.5 * (forward + reverse)


@torch.inference_mode()
def evaluate(model, pairs):
    model.eval()

    data = torch.as_tensor(
        pairs, dtype=torch.float32, device=DEVICE
    )

    a = model(data[:, 0])
    b = model(data[:, 1])

    similarity = a @ b.transpose(-1, -2)
    reverse = similarity.transpose(-1, -2)

    batch, nodes, _ = similarity.shape
    labels = torch.arange(
        nodes, device=DEVICE
    ).expand(batch, nodes)

    positives = similarity.diagonal(
        dim1=-2, dim2=-1
    )

    forward_ranks = 1 + (
        similarity > positives.unsqueeze(-1)
    ).sum(-1)

    reverse_ranks = 1 + (
        reverse > positives.unsqueeze(-1)
    ).sum(-1)

    mask = ~torch.eye(
        nodes, dtype=torch.bool, device=DEVICE
    )

    a_similarity = a @ a.transpose(-1, -2)
    b_similarity = b @ b.transpose(-1, -2)

    return {
        "loss": float(contrastive_loss(a, b).item()),
        "recall_at_1": float(
            (similarity.argmax(-1) == labels)
            .float().mean().item()
        ),
        "mrr": float(
            (1.0 / forward_ranks.float()).mean().item()
        ),
        "reverse_recall_at_1": float(
            (reverse.argmax(-1) == labels)
            .float().mean().item()
        ),
        "reverse_mrr": float(
            (1.0 / reverse_ranks.float()).mean().item()
        ),
        "median_positive_cosine": float(
            positives.median().item()
        ),
        "at2_offdiagonal_cosine": float(
            a_similarity[..., mask].mean().item()
        ),
        "gt450_offdiagonal_cosine": float(
            b_similarity[..., mask].mean().item()
        ),
    }


def better(current, best):
    if best is None:
        return True

    delta = (
        current["recall_at_1"] - best["recall_at_1"]
    )

    return (
        delta > 1e-10
        or (
            abs(delta) <= 1e-10
            and current["loss"] < best["loss"] - 1e-8
        )
    )


print("=" * 70, flush=True)
print("PHASE 4.2.7 - SELF-ONLY GRAPHSAGE CONTROL", flush=True)
print("=" * 70, flush=True)


# ---------------------------------------------------
# 1. Verify experimental provenance
# ---------------------------------------------------

print("\n[1] Verifying source experiments...", flush=True)

for path in (
    GRAPH, GRAPH_REPORT, PROTOCOL,
    SAGE_REPORT, PERM_REPORT
):
    if not path.is_file():
        raise RuntimeError(f"Missing source: {path}")

for path in (HISTORY, SUMMARY, REPORT):
    if path.exists():
        raise RuntimeError(f"Refusing overwrite: {path}")

graph_hash = file_hash(GRAPH)
protocol_hash = file_hash(PROTOCOL)

if read_json(GRAPH_REPORT)["graph_archive_sha256"] != graph_hash:
    raise RuntimeError("Graph provenance mismatch.")

protocol = read_json(PROTOCOL)

if protocol["source_graph_sha256"] != graph_hash:
    raise RuntimeError("Protocol graph mismatch.")

if protocol["partition_policy"]["test_used_for_hyperparameter_selection"]:
    raise RuntimeError("Test tuning is not permitted.")

sage = read_json(SAGE_REPORT)
permuted = read_json(PERM_REPORT)

for reference in (sage, permuted):
    if (
        reference["graph_archive_sha256"] != graph_hash
        or reference["protocol_sha256"] != protocol_hash
    ):
        raise RuntimeError("Reference provenance mismatch.")

if tuple(sage["seeds"]) != SEEDS or tuple(permuted["seeds"]) != SEEDS:
    raise RuntimeError("Reference seed mismatch.")

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

checkpoints = {
    seed: CHECKPOINT_DIR / f"self_only_seed_{seed}.pt"
    for seed in SEEDS
}

if any(path.exists() for path in checkpoints.values()):
    raise RuntimeError("Existing checkpoint detected.")

with np.load(GRAPH, allow_pickle=False) as data:
    features = data["node_features"].copy()
    regions = data["region_id"].astype(str)
    splits = data["spatial_split"].astype(str)
    scanners = data["scanner"].astype(str)
    tiles = data["tile_id"].astype(str)

if features.shape != (120, 36, 384):
    raise RuntimeError("Unexpected graph tensor.")

if features.dtype != np.float32 or not np.isfinite(features).all():
    raise RuntimeError("Invalid feature values.")

print("Graph tensor:", features.shape)
print("Device:", DEVICE)


# ---------------------------------------------------
# 2. Prepare train and validation region pairs
# ---------------------------------------------------

print("\n[2] Preparing spatial partitions...", flush=True)

by_region = defaultdict(dict)
region_split = {}

for i, region in enumerate(regions):
    scanner = scanners[i]
    split = splits[i]

    if scanner not in ("AT2", "GT450"):
        raise RuntimeError("Unexpected scanner.")

    if split not in ("train", "validation", "test"):
        raise RuntimeError("Unexpected partition.")

    if scanner in by_region[region]:
        raise RuntimeError("Duplicate scanner/region.")

    by_region[region][scanner] = i

    if region in region_split and region_split[region] != split:
        raise RuntimeError("Region crosses partitions.")

    region_split[region] = split

if dict(Counter(region_split.values())) != {
    "train": 40,
    "validation": 10,
    "test": 10,
}:
    raise RuntimeError("Spatial split mismatch.")

paired = {"train": [], "validation": []}

for region, indices in sorted(by_region.items()):
    if set(indices) != {"AT2", "GT450"}:
        raise RuntimeError("Incomplete scanner pair.")

    a = indices["AT2"]
    b = indices["GT450"]

    if not np.array_equal(tiles[a], tiles[b]):
        raise RuntimeError("Paired node order mismatch.")

    split = region_split[region]

    if split == "test":
        continue

    paired[split].append(
        np.stack(
            [features[a], features[b]], axis=0
        )
    )

train = np.stack(paired["train"]).astype(np.float32)
validation = np.stack(paired["validation"]).astype(np.float32)

if train.shape != (40, 2, 36, 384):
    raise RuntimeError("Incorrect training shape.")

if validation.shape != (10, 2, 36, 384):
    raise RuntimeError("Incorrect validation shape.")

print("Training pairs:", len(train))
print("Validation pairs:", len(validation))
print("Test pairs: 10 (reserved)")


# ---------------------------------------------------
# 3. Train self-only GraphSAGE
# ---------------------------------------------------

print("\n[3] Training self-only controls...", flush=True)

history = []
summaries = []
hashes = {}

for seed in SEEDS:
    print(f"\nSeed: {seed}", flush=True)

    set_seed(seed)

    model = SelfOnlyGraphSAGE().to(DEVICE)

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
        order = rng.permutation(len(train))

        total_loss = 0.0
        seen = 0

        for start in range(
            0, len(train), BATCH_REGIONS
        ):
            selected = order[
                start:start + BATCH_REGIONS
            ]

            batch = torch.as_tensor(
                train[selected],
                dtype=torch.float32,
                device=DEVICE,
            )

            a = model(batch[:, 0])
            b = model(batch[:, 1])

            loss = contrastive_loss(a, b)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += (
                float(loss.item()) * len(selected)
            )
            seen += len(selected)

        train_loss = total_loss / seen

        validation_metrics = evaluate(
            model, validation
        )

        improved = better(validation_metrics, best)

        if improved:
            best = dict(validation_metrics)
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
            "validation_loss": validation_metrics["loss"],
            "validation_recall_at_1":
                validation_metrics["recall_at_1"],
            "validation_mrr": validation_metrics["mrr"],
            "best_checkpoint": improved,
        })

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  Epoch {epoch:03d} | "
                f"Train loss {train_loss:.4f} | "
                f"Val R@1 "
                f"{validation_metrics['recall_at_1']:.4f} | "
                f"MRR {validation_metrics['mrr']:.4f}",
                flush=True,
            )

        if stale >= PATIENCE:
            print(
                f"  Early stopping: epoch {epoch}",
                flush=True,
            )
            break

    if best_weights is None:
        raise RuntimeError("No checkpoint selected.")

    model.load_state_dict(best_weights)
    final = evaluate(model, validation)

    checkpoint = checkpoints[seed]

    with checkpoint.open("xb") as stream:
        torch.save({
            "model_state_dict": best_weights,
            "model_name": "self_only_graphsage",
            "architecture": "384_256_384",
            "neighbor_aggregation": "disabled_during_training",
            "seed": seed,
            "best_epoch": best_epoch,
            "validation_metrics": final,
            "graph_archive_sha256": graph_hash,
            "protocol_sha256": protocol_hash,
            "test_evaluated": False,
        }, stream)

    hashes[str(seed)] = file_hash(checkpoint)

    original_row = next(
        r for r in sage["seed_results"]
        if int(r["seed"]) == seed
    )

    permuted_row = next(
        r for r in permuted["seed_results"]
        if int(r["seed"]) == seed
    )

    summaries.append({
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_loss": final["loss"],
        "validation_recall_at_1": final["recall_at_1"],
        "validation_mrr": final["mrr"],
        "reverse_recall_at_1": final["reverse_recall_at_1"],
        "at2_offdiagonal_cosine": final["at2_offdiagonal_cosine"],
        "gt450_offdiagonal_cosine": final["gt450_offdiagonal_cosine"],
        "original_graphsage_recall_at_1":
            original_row["validation_recall_at_1"],
        "permuted_graphsage_recall_at_1":
            permuted_row["validation_recall_at_1"],
        "checkpoint_sha256": hashes[str(seed)],
    })

    print(
        f"  Best epoch: {best_epoch} | "
        f"Self-only R@1: {final['recall_at_1']:.4f}",
        flush=True,
    )


# ---------------------------------------------------
# 4. Save results
# ---------------------------------------------------

print("\n[4] Saving experiment results...", flush=True)

save_csv(
    HISTORY, history,
    [
        "seed", "epoch", "train_loss",
        "validation_loss", "validation_recall_at_1",
        "validation_mrr", "best_checkpoint",
    ],
)

save_csv(
    SUMMARY, summaries,
    [
        "seed", "best_epoch", "validation_loss",
        "validation_recall_at_1", "validation_mrr",
        "reverse_recall_at_1",
        "at2_offdiagonal_cosine",
        "gt450_offdiagonal_cosine",
        "original_graphsage_recall_at_1",
        "permuted_graphsage_recall_at_1",
        "checkpoint_sha256",
    ],
)

self_scores = np.array([
    row["validation_recall_at_1"]
    for row in summaries
], dtype=np.float64)

original_scores = np.array([
    row["original_graphsage_recall_at_1"]
    for row in summaries
], dtype=np.float64)

permuted_scores = np.array([
    row["permuted_graphsage_recall_at_1"]
    for row in summaries
], dtype=np.float64)

report = {
    "phase": "4.2",
    "step": "4.2.7",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "experiment": "self_only_graphsage_control",
    "architecture": "GraphSAGE_384_256_384_GELU",
    "neighbor_messages": "zero_during_training_and_evaluation",
    "optimization": "AdamW",
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "temperature": TEMPERATURE,
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
    "mean_self_only_recall_at_1": float(self_scores.mean()),
    "std_self_only_recall_at_1": float(
        self_scores.std(ddof=1)
    ),
    "mean_original_graphsage_recall_at_1": float(
        original_scores.mean()
    ),
    "mean_permuted_graphsage_recall_at_1": float(
        permuted_scores.mean()
    ),
    "original_minus_self_only_pp": float(
        (original_scores - self_scores).mean() * 100
    ),
    "permuted_minus_self_only_pp": float(
        (permuted_scores - self_scores).mean() * 100
    ),
    "seed_results": summaries,
    "history_sha256": file_hash(HISTORY),
    "summary_sha256": file_hash(SUMMARY),
    "checkpoint_hashes": hashes,
    "test_evaluated_for_selection": False,
    "biological_independence_verified": False,
    "xai_planned": True,
    "limitations": [
        "Self-only neighbor parameter branch has no active gradients",
        "Three initialization seeds",
        "Ten spatial validation regions",
        "Near-ceiling retrieval performance for graph baselines",
        "No independent biological validation",
        "No final test evaluation",
    ],
}

save_json(REPORT, report)

print("\n" + "=" * 70)
print("PHASE 4.2.7 - SELF-ONLY CONTROL COMPLETED")
print("=" * 70)

for row in summaries:
    print(
        f"Seed {row['seed']} | "
        f"Self-only {row['validation_recall_at_1']:.4f} | "
        f"Original {row['original_graphsage_recall_at_1']:.4f} | "
        f"Permuted {row['permuted_graphsage_recall_at_1']:.4f}"
    )

print(
    "\nSelf-only mean R@1:",
    f"{self_scores.mean():.4f}"
)
print(
    "Original mean R@1:",
    f"{original_scores.mean():.4f}"
)
print(
    "Permuted mean R@1:",
    f"{permuted_scores.mean():.4f}"
)
print(
    "Original - self-only:",
    f"{report['original_minus_self_only_pp']:+.2f} pp"
)
print(
    "Permuted - self-only:",
    f"{report['permuted_minus_self_only_pp']:+.2f} pp"
)

print("\nSaved:")
print(HISTORY)
print(SUMMARY)
print(REPORT)

for checkpoint in checkpoints.values():
    print(checkpoint)

print("\nTest partition: NOT EVALUATED")
print("Previous experiments: UNCHANGED")
print("Proposed stability-aware GNN: NOT TRAINED")
print("XAI: PLANNED")
print("\nSTEP 4.2.7 COMPLETED")
