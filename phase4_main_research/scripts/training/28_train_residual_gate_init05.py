
"""
PHASE 4.2.13 - RESIDUAL GATE INITIALIZATION CONTROL

Only intentional architecture change relative to 4.2.11:
    Initial gate: 0.10 -> 0.50

The gate remains learnable with sigmoid parameterization.
This is an exploratory, validation-only follow-up.

Training:
    Seeds: 42, 123, 2026
    Train: 40 paired regions
    Validation: 10 paired regions
    Test: completely reserved

Artifacts are new and previous results are never overwritten.
"""

from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
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
ARCH = BASE / "configs/25_stability_aware_gnn_architecture_v1.json"
ARCH_REPORT = BASE / "outputs/reports/25_architecture_protocol_report.json"

PREVIOUS = BASE / "outputs/reports/26_gated_residual_graphsage_report.json"
DIAGNOSTIC = BASE / "outputs/reports/27_residual_gate_diagnostics.json"

CHECKPOINT_DIR = BASE / "outputs/checkpoints/28_residual_gate_init05"
HISTORY = BASE / "outputs/tables/28_gate_init05_training_history.csv"
SUMMARY = BASE / "outputs/tables/28_gate_init05_validation_summary.csv"
REPORT = BASE / "outputs/reports/28_gate_init05_report.json"

SEEDS = (42, 123, 2026)
INITIAL_GATE = 0.5

TEMPERATURE = 0.07
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_REGIONS = 8
MAX_EPOCHS = 100
PATIENCE = 20

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


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


def neighbor_matrix(edges):
    if edges.shape != (2, 120):
        raise RuntimeError("Unexpected edge-index shape.")

    a = torch.zeros((36, 36), dtype=torch.float32)
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
        a[target, source] = 1.0

    if len(seen) != 120:
        raise RuntimeError("Incorrect edge count.")

    if any((v, u) not in seen for u, v in seen):
        raise RuntimeError("Nonreciprocal adjacency.")

    degree = a.sum(dim=1, keepdim=True)

    if (degree <= 0).any():
        raise RuntimeError("Isolated node.")

    return a / degree


class SAGEConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.neighbor_linear = nn.Linear(
            input_dim, output_dim, bias=False
        )

    def forward(self, x, neighbors):
        return (
            self.self_linear(x)
            + self.neighbor_linear(
                torch.matmul(neighbors, x)
            )
        )


class GatedResidualGraphSAGE(nn.Module):
    def __init__(self, neighbors):
        super().__init__()

        self.register_buffer(
            "neighbor_weights", neighbors.clone()
        )

        self.conv1 = SAGEConv(384, 256)
        self.conv2 = SAGEConv(256, 384)

        logit = math.log(
            INITIAL_GATE / (1.0 - INITIAL_GATE)
        )

        self.gate_logit = nn.Parameter(
            torch.tensor(logit, dtype=torch.float32)
        )

    def gate(self):
        return torch.sigmoid(self.gate_logit)

    def forward(self, x):
        identity = F.normalize(x, p=2, dim=-1)

        hidden = F.gelu(
            self.conv1(x, self.neighbor_weights)
        )

        graph = F.normalize(
            self.conv2(hidden, self.neighbor_weights),
            p=2,
            dim=-1,
        )

        return F.normalize(
            identity + self.gate() * graph,
            p=2,
            dim=-1,
        )


def contrastive_loss(a, b):
    logits = (
        a @ b.transpose(-1, -2)
    ) / TEMPERATURE

    batch, nodes, _ = logits.shape

    targets = torch.arange(
        nodes, device=logits.device
    ).expand(batch, nodes).reshape(-1)

    return 0.5 * (
        F.cross_entropy(
            logits.reshape(-1, nodes), targets
        )
        + F.cross_entropy(
            logits.transpose(-1, -2).reshape(-1, nodes),
            targets
        )
    )


def retrieval(a, b):
    """
    Query and candidate rows have corresponding order.
    Handles a batch of 36-candidate retrieval problems.
    """
    similarity = a @ b.transpose(-1, -2)

    batch, nodes, _ = similarity.shape

    targets = torch.arange(
        nodes, device=similarity.device
    ).expand(batch, nodes)

    positive = similarity.diagonal(
        dim1=-2, dim2=-1
    )

    ranks = 1 + (
        similarity > positive.unsqueeze(-1)
    ).sum(-1)

    return {
        "recall_at_1": float(
            (similarity.argmax(-1) == targets)
            .float().mean().item()
        ),
        "mrr": float(
            (1.0 / ranks.float()).mean().item()
        ),
        "reverse_recall_at_1": float(
            (
                similarity.transpose(-1, -2).argmax(-1)
                == targets
            ).float().mean().item()
        ),
        "median_positive_cosine": float(
            positive.median().item()
        ),
    }


def offdiagonal_cosine(z):
    scores = z @ z.transpose(-1, -2)

    mask = ~torch.eye(
        36, dtype=torch.bool, device=z.device
    )

    return float(scores[..., mask].mean().item())


@torch.inference_mode()
def evaluate(model, data, include_global=False):
    model.eval()

    batch = torch.as_tensor(
        data, dtype=torch.float32, device=DEVICE
    )

    a = model(batch[:, 0])
    b = model(batch[:, 1])

    result = retrieval(a, b)

    result.update({
        "loss": float(contrastive_loss(a, b).item()),
        "gate": float(model.gate().item()),
        "at2_offdiagonal_cosine": offdiagonal_cosine(a),
        "gt450_offdiagonal_cosine": offdiagonal_cosine(b),
    })

    if include_global:
        flattened_a = a.reshape(-1, 384)
        flattened_b = b.reshape(-1, 384)

        # Reuse retrieval() with one batch of
        # 360 candidate nodes.
        global_metrics = retrieval(
            flattened_a.unsqueeze(0),
            flattened_b.unsqueeze(0),
        )

        result["global_recall_at_1"] = (
            global_metrics["recall_at_1"]
        )
        result["global_mrr"] = global_metrics["mrr"]

    return result


def better(current, best):
    if best is None:
        return True

    delta = (
        current["recall_at_1"]
        - best["recall_at_1"]
    )

    return (
        delta > 1e-10
        or (
            abs(delta) <= 1e-10
            and current["loss"] < best["loss"] - 1e-8
        )
    )


print("=" * 70, flush=True)
print("PHASE 4.2.13 - RESIDUAL GATE INITIALIZATION 0.5", flush=True)
print("=" * 70, flush=True)


# ---------------------------------------------------
# 1. Verify research provenance
# ---------------------------------------------------

print("\n[1] Verifying source experiments...", flush=True)

sources = (
    GRAPH, GRAPH_REPORT, PROTOCOL, ARCH,
    ARCH_REPORT, PREVIOUS, DIAGNOSTIC
)

for path in sources:
    if not path.is_file():
        raise RuntimeError(f"Missing input: {path}")

for path in (HISTORY, SUMMARY, REPORT):
    if path.exists():
        raise RuntimeError(
            f"Refusing overwrite: {path}"
        )

graph_hash = sha256(GRAPH)
protocol_hash = sha256(PROTOCOL)
arch_hash = sha256(ARCH)

if load_json(GRAPH_REPORT)["graph_archive_sha256"] != graph_hash:
    raise RuntimeError("Graph provenance mismatch.")

protocol = load_json(PROTOCOL)
architecture = load_json(ARCH)
arch_report = load_json(ARCH_REPORT)
previous = load_json(PREVIOUS)
diagnostic = load_json(DIAGNOSTIC)

if protocol["source_graph_sha256"] != graph_hash:
    raise RuntimeError("Protocol provenance mismatch.")

if (
    architecture["graph_archive_sha256"] != graph_hash
    or architecture["initial_protocol_sha256"] != protocol_hash
    or arch_report["architecture_sha256"] != arch_hash
):
    raise RuntimeError("Architecture provenance mismatch.")

if (
    previous["graph_archive_sha256"] != graph_hash
    or previous["protocol_sha256"] != protocol_hash
    or previous["architecture_sha256"] != arch_hash
):
    raise RuntimeError("Previous model provenance mismatch.")

if (
    diagnostic["graph_archive_sha256"] != graph_hash
    or diagnostic["source_model_report_sha256"] != sha256(PREVIOUS)
):
    raise RuntimeError("Gate diagnostic provenance mismatch.")

if (
    previous["test_evaluated"]
    or diagnostic["test_evaluated"]
):
    raise RuntimeError("Unexpected prior test evaluation.")

spec = architecture["initial_training"]

required_hyperparameters = {
    "temperature": TEMPERATURE,
    "learning_rate": LR,
    "weight_decay": WEIGHT_DECAY,
    "batch_regions": BATCH_REGIONS,
    "maximum_epochs": MAX_EPOCHS,
    "early_stopping_patience": PATIENCE,
}

for name, expected in required_hyperparameters.items():
    if spec[name] != expected:
        raise RuntimeError(
            f"Protocol mismatch: {name}"
        )

if tuple(spec["seeds"]) != SEEDS:
    raise RuntimeError("Seed mismatch.")

if architecture["model"]["gate_initial_value"] != 0.1:
    raise RuntimeError(
        "Expected original gate initialization of 0.1."
    )

if architecture["model"]["gate_type"] != "learned_global_scalar":
    raise RuntimeError("Unexpected gate architecture.")

if architecture["model"]["dimensions"] != [384, 256, 384]:
    raise RuntimeError("Architecture dimensions changed.")

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

checkpoint_paths = {
    seed: CHECKPOINT_DIR / f"gate_init05_seed_{seed}.pt"
    for seed in SEEDS
}

if any(path.exists() for path in checkpoint_paths.values()):
    raise RuntimeError("Checkpoint already exists.")

with np.load(GRAPH, allow_pickle=False) as z:
    features = z["node_features"].copy()
    edges = z["edge_index"].copy()
    regions = z["region_id"].astype(str)
    splits = z["spatial_split"].astype(str)
    scanners = z["scanner"].astype(str)
    tiles = z["tile_id"].astype(str)

if (
    features.shape != (120, 36, 384)
    or features.dtype != np.float32
    or not np.isfinite(features).all()
):
    raise RuntimeError("Invalid feature archive.")

neighbors = neighbor_matrix(edges).to(DEVICE)

print("Graph tensor:", features.shape)
print("Device:", DEVICE)
print("Original initial gate: 0.1")
print("New initial gate:", INITIAL_GATE)


# ---------------------------------------------------
# 2. Prepare matched regions without test evaluation
# ---------------------------------------------------

print("\n[2] Preparing train/validation partitions...", flush=True)

by_region = defaultdict(dict)
region_splits = {}

for i, name in enumerate(regions):
    scanner = scanners[i]
    split = splits[i]

    if scanner not in ("AT2", "GT450"):
        raise RuntimeError("Unexpected scanner.")

    if split not in ("train", "validation", "test"):
        raise RuntimeError("Unexpected partition.")

    if scanner in by_region[name]:
        raise RuntimeError("Duplicate scanner graph.")

    by_region[name][scanner] = i

    if name in region_splits and region_splits[name] != split:
        raise RuntimeError("Region split conflict.")

    region_splits[name] = split

if dict(Counter(region_splits.values())) != {
    "train": 40, "validation": 10, "test": 10
}:
    raise RuntimeError("Incorrect spatial partitions.")

paired = {"train": [], "validation": []}

for region in sorted(by_region):
    pair = by_region[region]

    if set(pair) != {"AT2", "GT450"}:
        raise RuntimeError("Unpaired region.")

    ai, bi = pair["AT2"], pair["GT450"]

    if not np.array_equal(tiles[ai], tiles[bi]):
        raise RuntimeError("Tile ordering mismatch.")

    split = region_splits[region]

    if split == "test":
        continue

    paired[split].append(
        np.stack(
            [features[ai], features[bi]],
            axis=0,
        )
    )

train = np.stack(paired["train"]).astype(np.float32)
validation = np.stack(paired["validation"]).astype(np.float32)

if train.shape != (40, 2, 36, 384):
    raise RuntimeError("Training shape mismatch.")

if validation.shape != (10, 2, 36, 384):
    raise RuntimeError("Validation shape mismatch.")

print("Training pairs:", len(train))
print("Validation pairs:", len(validation))
print("Test pairs: 10 (reserved)")


# ---------------------------------------------------
# 3. Train the gate-initialization control
# ---------------------------------------------------

print("\n[3] Training independent seeds...", flush=True)

history = []
summaries = []
checkpoint_hashes = {}

for seed in SEEDS:
    print(f"\nSeed: {seed}", flush=True)

    set_seed(seed)

    model = GatedResidualGraphSAGE(
        neighbors
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    rng = np.random.default_rng(seed)

    best_metrics = None
    best_weights = None
    best_epoch = None
    stale = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        permutation = rng.permutation(len(train))

        total_loss = 0.0
        seen = 0

        for start in range(
            0, len(train), BATCH_REGIONS
        ):
            indices = permutation[
                start:start + BATCH_REGIONS
            ]

            batch = torch.as_tensor(
                train[indices],
                dtype=torch.float32,
                device=DEVICE,
            )

            a = model(batch[:, 0])
            b = model(batch[:, 1])

            loss = contrastive_loss(a, b)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss.")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(indices)
            seen += len(indices)

        train_loss = total_loss / seen
        metrics = evaluate(model, validation)

        if not all(
            math.isfinite(float(v))
            for v in metrics.values()
        ):
            raise RuntimeError(
                "Non-finite validation metrics."
            )

        improved = better(metrics, best_metrics)

        if improved:
            best_metrics = dict(metrics)
            best_epoch = epoch

            best_weights = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
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
            "learned_gate": metrics["gate"],
            "best_checkpoint": improved,
        })

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  Epoch {epoch:03d} | "
                f"Train loss {train_loss:.4f} | "
                f"Val R@1 {metrics['recall_at_1']:.4f} | "
                f"MRR {metrics['mrr']:.4f} | "
                f"Gate {metrics['gate']:.4f}",
                flush=True,
            )

        if stale >= PATIENCE:
            print(
                f"  Early stopping at epoch {epoch}",
                flush=True,
            )
            break

    if best_weights is None:
        raise RuntimeError("No checkpoint selected.")

    model.load_state_dict(
        best_weights, strict=True
    )

    final = evaluate(
        model, validation, include_global=True
    )

    if abs(
        final["recall_at_1"]
        - best_metrics["recall_at_1"]
    ) > 1e-7:
        raise RuntimeError(
            "Best checkpoint restoration failed."
        )

    checkpoint = checkpoint_paths[seed]

    with checkpoint.open("xb") as stream:
        torch.save({
            "model_state_dict": best_weights,
            "model_name": "GatedResidualGraphSAGE",
            "experiment": "gate_initialization_0.5",
            "seed": seed,
            "initial_gate": INITIAL_GATE,
            "best_epoch": best_epoch,
            "validation_metrics": final,
            "graph_archive_sha256": graph_hash,
            "protocol_sha256": protocol_hash,
            "parent_architecture_sha256": arch_hash,
            "test_evaluated": False,
        }, stream)

    checkpoint_hashes[str(seed)] = sha256(
        checkpoint
    )

    original = next(
        r for r in previous["seed_results"]
        if int(r["seed"]) == seed
    )

    summaries.append({
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_loss": final["loss"],
        "validation_recall_at_1": final["recall_at_1"],
        "validation_mrr": final["mrr"],
        "global_recall_at_1": final["global_recall_at_1"],
        "global_mrr": final["global_mrr"],
        "reverse_recall_at_1":
            final["reverse_recall_at_1"],
        "learned_gate": final["gate"],
        "at2_offdiagonal_cosine":
            final["at2_offdiagonal_cosine"],
        "gt450_offdiagonal_cosine":
            final["gt450_offdiagonal_cosine"],
        "original_gate01_recall_at_1":
            original["validation_recall_at_1"],
        "checkpoint_sha256": checkpoint_hashes[str(seed)],
    })

    print(
        f"  Best epoch {best_epoch} | "
        f"Val R@1 {final['recall_at_1']:.4f} | "
        f"Global R@1 {final['global_recall_at_1']:.4f} | "
        f"Gate {final['gate']:.4f}",
        flush=True,
    )


# ---------------------------------------------------
# 4. Save reports, without replacing old experiments
# ---------------------------------------------------

print("\n[4] Saving controlled experiment...", flush=True)

save_csv(
    HISTORY,
    history,
    [
        "seed", "epoch", "train_loss",
        "validation_loss", "validation_recall_at_1",
        "validation_mrr", "learned_gate",
        "best_checkpoint",
    ],
)

save_csv(
    SUMMARY,
    summaries,
    [
        "seed", "best_epoch", "validation_loss",
        "validation_recall_at_1", "validation_mrr",
        "global_recall_at_1", "global_mrr",
        "reverse_recall_at_1", "learned_gate",
        "at2_offdiagonal_cosine",
        "gt450_offdiagonal_cosine",
        "original_gate01_recall_at_1",
        "checkpoint_sha256",
    ],
)

new_scores = np.array(
    [r["validation_recall_at_1"] for r in summaries],
    dtype=np.float64,
)

old_scores = np.array(
    [r["original_gate01_recall_at_1"] for r in summaries],
    dtype=np.float64,
)

graphsage_reference = float(
    load_json(
        BASE / "outputs/reports/19_graphsage_baseline_report.json"
    )["mean_validation_recall_at_1"]
)

report = {
    "phase": "4.2",
    "step": "4.2.13",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "experiment": "residual_gate_initialization_control",
    "model": "GatedResidualGraphSAGE",
    "parent_architecture_sha256": arch_hash,
    "changed_variable": "initial_gate_only",
    "original_gate_initialization": 0.1,
    "new_gate_initialization": INITIAL_GATE,
    "gate_trainable": True,
    "explicit_stability_loss": False,
    "diversity_regularization": False,
    "seeds": list(SEEDS),
    "temperature": TEMPERATURE,
    "learning_rate": LR,
    "weight_decay": WEIGHT_DECAY,
    "batch_regions": BATCH_REGIONS,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
    "graph_archive_sha256": graph_hash,
    "protocol_sha256": protocol_hash,
    "train_region_pairs": 40,
    "validation_region_pairs": 10,
    "test_region_pairs_reserved": 10,
    "test_evaluated": False,
    "checkpoint_selection":
        "within_region_validation_recall_at_1_then_loss",
    "new_mean_validation_recall_at_1":
        float(new_scores.mean()),
    "new_std_validation_recall_at_1":
        float(new_scores.std(ddof=1)),
    "original_mean_validation_recall_at_1":
        float(old_scores.mean()),
    "new_minus_original_pp":
        float((new_scores - old_scores).mean() * 100),
    "reference_graphsage_mean_recall_at_1":
        graphsage_reference,
    "new_minus_graphsage_pp":
        float((new_scores.mean() - graphsage_reference) * 100),
    "seed_results": summaries,
    "checkpoint_hashes": checkpoint_hashes,
    "history_sha256": sha256(HISTORY),
    "summary_sha256": sha256(SUMMARY),
    "validation_reuse_notice": (
        "The new initialization was motivated by the earlier "
        "post-hoc validation gate sweep. This is exploratory "
        "model development, not independent confirmation."
    ),
    "biological_independence_verified": False,
    "xai_planned": True,
}

save_json(REPORT, report)

print("\n" + "=" * 70)
print("PHASE 4.2.13 - GATE INIT 0.5 COMPLETED")
print("=" * 70)

for row in summaries:
    print(
        f"Seed {row['seed']} | "
        f"Epoch {row['best_epoch']} | "
        f"Val R@1 {row['validation_recall_at_1']:.4f} | "
        f"Global R@1 {row['global_recall_at_1']:.4f} | "
        f"Gate {row['learned_gate']:.4f}"
    )

print(
    "\nNew mean validation R@1:",
    f"{new_scores.mean():.4f}"
)

print(
    "Original gate 0.1 mean:",
    f"{old_scores.mean():.4f}"
)

print(
    "Improvement:",
    f"{report['new_minus_original_pp']:+.2f} pp"
)

print(
    "Difference vs original GraphSAGE:",
    f"{report['new_minus_graphsage_pp']:+.2f} pp"
)

print("\nSaved:")
print(HISTORY)
print(SUMMARY)
print(REPORT)

for checkpoint in checkpoint_paths.values():
    print(checkpoint)

print("\nPrevious experiments: UNCHANGED")
print("Test partition: NOT EVALUATED")
print("XAI: PLANNED")
print("\nSTEP 4.2.13 COMPLETED")

