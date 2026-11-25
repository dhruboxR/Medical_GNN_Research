
"""
PHASE 4.2.2 - TRAINABLE MLP BASELINE

Research question:
Can a shared MLP improve cross-scanner node matching
without graph message passing?

Input:
    Phase 4.1.15 verified graph archive
    Phase 4.2.1 experimental protocol

Method:
    Shared 384 -> 256 -> 384 MLP
    Bidirectional within-region contrastive loss
    Cosine similarity, temperature 0.07

Training:
    40 paired regions
    10 validation paired regions
    Test partition NOT evaluated
    Three random seeds: 42, 123, 2026
    Early stopping using validation metrics

Output:
    One best checkpoint per seed
    Epoch history CSV
    Per-seed validation summary CSV
    JSON experimental report

No graph edges used.
No original feature or image files modified.
"""

from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
from copy import deepcopy
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

CHECKPOINT_DIR = (
    BASE / "outputs/checkpoints"
    / "17_mlp_baseline"
)

HISTORY_PATH = (
    BASE / "outputs/tables"
    / "17_mlp_training_history.csv"
)

SUMMARY_PATH = (
    BASE / "outputs/tables"
    / "17_mlp_validation_summary.csv"
)

REPORT_PATH = (
    BASE / "outputs/reports"
    / "17_mlp_baseline_report.json"
)

SEEDS = (42, 123, 2026)

FEATURE_DIM = 384
HIDDEN_DIM = 256

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
    h = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def write_csv_new(path, rows, columns):
    with path.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=columns,
        )
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SharedMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(FEATURE_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, FEATURE_DIM),
        )

    def forward(self, x):
        z = self.network(x)
        return F.normalize(z, dim=-1)


def contrastive_logits(a, b):
    return (
        a @ b.transpose(-1, -2)
    ) / TEMPERATURE


def contrastive_loss(a, b):
    """
    One correct matching GT450 node among 36
    candidates for each AT2 node, and vice versa.
    """

    logits = contrastive_logits(a, b)

    batch_size, nodes, _ = logits.shape

    labels = torch.arange(
        nodes,
        device=logits.device,
    ).expand(batch_size, -1)

    forward_loss = F.cross_entropy(
        logits.reshape(-1, nodes),
        labels.reshape(-1),
    )

    reverse_loss = F.cross_entropy(
        logits.transpose(1, 2).reshape(-1, nodes),
        labels.reshape(-1),
    )

    return 0.5 * (
        forward_loss + reverse_loss
    )


@torch.no_grad()
def evaluate(model, pairs):
    """
    Evaluate a collection of paired regions.

    Primary metric:
        AT2 -> GT450 Recall@1.

    Secondary:
        MRR, reverse retrieval, contrastive loss.

    No gradients or parameter updates.
    """

    model.eval()

    a = torch.as_tensor(
        pairs[:, 0],
        dtype=torch.float32,
        device=DEVICE,
    )

    b = torch.as_tensor(
        pairs[:, 1],
        dtype=torch.float32,
        device=DEVICE,
    )

    za = model(a)
    zb = model(b)

    logits = contrastive_logits(za, zb)
    loss = float(
        contrastive_loss(za, zb).item()
    )

    similarity = (
        za @ zb.transpose(-1, -2)
    )

    nodes = similarity.size(-1)

    diagonal = similarity.diagonal(
        dim1=-2,
        dim2=-1,
    )

    rank_forward = 1 + (
        similarity > diagonal.unsqueeze(-1)
    ).sum(-1)

    rank_reverse = 1 + (
        similarity.transpose(1, 2)
        > diagonal.unsqueeze(-1)
    ).sum(-1)

    predicted_forward = similarity.argmax(-1)

    labels = torch.arange(
        nodes,
        device=DEVICE,
    ).expand(similarity.size(0), -1)

    correct_forward = (
        predicted_forward == labels
    )

    predicted_reverse = similarity.transpose(
        1, 2
    ).argmax(-1)

    correct_reverse = (
        predicted_reverse == labels
    )

    return {
        "loss": loss,
        "recall_at_1": float(
            correct_forward.float().mean().item()
        ),
        "mrr": float(
            (1.0 / rank_forward.float())
            .mean()
            .item()
        ),
        "reverse_recall_at_1": float(
            correct_reverse.float().mean().item()
        ),
        "reverse_mrr": float(
            (1.0 / rank_reverse.float())
            .mean()
            .item()
        ),
        "median_positive_cosine": float(
            diagonal.median().item()
        ),
    }


def best_is_better(candidate, best):
    """
    Select checkpoints by validation Recall@1.
    Break ties with lower validation loss.
    """

    if best is None:
        return True

    if candidate["recall_at_1"] > best["recall_at_1"] + 1e-10:
        return True

    if (
        abs(
            candidate["recall_at_1"]
            - best["recall_at_1"]
        ) <= 1e-10
        and candidate["loss"] < best["loss"] - 1e-8
    ):
        return True

    return False


print("=" * 70, flush=True)
print("PHASE 4.2.2 - MLP BASELINE TRAINING", flush=True)
print("=" * 70, flush=True)


# ---------------------------------------------------------
# 1. Validate the frozen experimental inputs
# ---------------------------------------------------------

print("\n[1] Validating protocol and graph provenance...", flush=True)

for path in (GRAPH_PATH, GRAPH_REPORT, PROTOCOL_PATH):
    if not path.is_file():
        raise RuntimeError(
            f"Missing input: {path}"
        )

with GRAPH_REPORT.open(encoding="utf-8") as stream:
    graph_report = json.load(stream)

with PROTOCOL_PATH.open(encoding="utf-8") as stream:
    protocol = json.load(stream)

graph_hash = sha256(GRAPH_PATH)
protocol_hash = sha256(PROTOCOL_PATH)

if graph_hash != graph_report["graph_archive_sha256"]:
    raise RuntimeError(
        "Graph archive provenance mismatch."
    )

if graph_hash != protocol["source_graph_sha256"]:
    raise RuntimeError(
        "Experimental protocol graph mismatch."
    )

if protocol["partition_policy"]["test_used_for_hyperparameter_selection"]:
    raise RuntimeError(
        "Protocol incorrectly permits test tuning."
    )

if "mlp_projection" not in protocol["initial_baselines"]:
    raise RuntimeError(
        "MLP is not listed as an approved baseline."
    )

for output in (HISTORY_PATH, SUMMARY_PATH, REPORT_PATH):
    if output.exists():
        raise RuntimeError(
            f"Refusing to overwrite: {output}"
        )

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

checkpoint_paths = {
    seed: CHECKPOINT_DIR / f"mlp_seed_{seed}.pt"
    for seed in SEEDS
}

if any(path.exists() for path in checkpoint_paths.values()):
    raise RuntimeError(
        "Existing MLP checkpoint found. "
        "Refusing to overwrite experiment results."
    )

with np.load(
    GRAPH_PATH,
    allow_pickle=False,
) as archive:
    features = archive["node_features"].copy()
    regions = archive["region_id"].astype(str)
    splits = archive["spatial_split"].astype(str)
    scanners = archive["scanner"].astype(str)
    tile_ids = archive["tile_id"].astype(str)

if features.shape != (120, 36, 384):
    raise RuntimeError(
        "Unexpected graph feature shape."
    )

if features.dtype != np.float32:
    raise RuntimeError(
        "Unexpected graph feature datatype."
    )

if not np.isfinite(features).all():
    raise RuntimeError(
        "Non-finite graph features."
    )

print("Graphs:", len(features))
print("Graph feature dimensions:", features.shape)
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
        flush=True,
    )


# ---------------------------------------------------------
# 2. Assemble paired regions, without evaluating test
# ---------------------------------------------------------

print("\n[2] Preparing training and validation pairs...", flush=True)

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
        raise RuntimeError("Duplicate region/scanner.")

    by_region[region][scanner] = i

    if region in region_split and region_split[region] != split:
        raise RuntimeError("Region appears in multiple splits.")

    region_split[region] = split

paired_data = {
    "train": [],
    "validation": [],
}

for region in sorted(by_region):
    pair = by_region[region]

    if set(pair) != {"AT2", "GT450"}:
        raise RuntimeError(
            f"Incomplete scanner pair: {region}"
        )

    first = pair["AT2"]
    second = pair["GT450"]

    if not np.array_equal(
        tile_ids[first],
        tile_ids[second],
    ):
        raise RuntimeError(
            f"Node correspondence mismatch: {region}"
        )

    split = region_split[region]

    # Test features are deliberately never copied
    # into the training or evaluation collections.
    if split == "test":
        continue

    paired_data[split].append(
        np.stack(
            [features[first], features[second]],
            axis=0,
        )
    )

train_pairs = np.stack(
    paired_data["train"],
).astype(np.float32)

validation_pairs = np.stack(
    paired_data["validation"],
).astype(np.float32)

if train_pairs.shape != (40, 2, 36, 384):
    raise RuntimeError(
        "Training pairs have unexpected shape."
    )

if validation_pairs.shape != (10, 2, 36, 384):
    raise RuntimeError(
        "Validation pairs have unexpected shape."
    )

if dict(Counter(region_split.values())) != {
    "train": 40,
    "validation": 10,
    "test": 10,
}:
    raise RuntimeError(
        "Spatial partition count mismatch."
    )

print("Training region pairs:", len(train_pairs))
print("Validation region pairs:", len(validation_pairs))
print("Test region pairs: 10 (not evaluated)")


# ---------------------------------------------------------
# 3. Train independent random seeds
# ---------------------------------------------------------

print("\n[3] Training shared MLP projection...", flush=True)

history_rows = []
summary_rows = []
checkpoint_hashes = {}

for seed in SEEDS:
    print(
        f"\nRandom seed: {seed}",
        flush=True,
    )

    set_seed(seed)

    model = SharedMLP().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_metrics = None
    best_state = None
    best_epoch = None
    stale_epochs = 0

    rng = np.random.default_rng(seed)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        indices = rng.permutation(
            len(train_pairs)
        )

        accumulated_loss = 0.0
        examples_seen = 0

        for start in range(
            0,
            len(indices),
            BATCH_REGIONS,
        ):
            selected = indices[
                start:start + BATCH_REGIONS
            ]

            batch = torch.as_tensor(
                train_pairs[selected],
                dtype=torch.float32,
                device=DEVICE,
            )

            a = model(batch[:, 0])
            b = model(batch[:, 1])

            loss = contrastive_loss(a, b)

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()
            optimizer.step()

            accumulated_loss += (
                float(loss.item()) * len(selected)
            )

            examples_seen += len(selected)

        train_loss = (
            accumulated_loss / examples_seen
        )

        validation = evaluate(
            model,
            validation_pairs,
        )

        improved = best_is_better(
            validation,
            best_metrics,
        )

        if improved:
            best_metrics = dict(validation)
            best_epoch = epoch

            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

            stale_epochs = 0

        else:
            stale_epochs += 1

        history_rows.append({
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation["loss"],
            "validation_recall_at_1":
                validation["recall_at_1"],
            "validation_mrr": validation["mrr"],
            "validation_reverse_recall_at_1":
                validation["reverse_recall_at_1"],
            "validation_reverse_mrr":
                validation["reverse_mrr"],
            "best_checkpoint": improved,
        })

        if (
            epoch == 1
            or epoch % 10 == 0
            or improved and epoch <= 5
        ):
            print(
                f"  Epoch {epoch:03d} | "
                f"Train loss {train_loss:.4f} | "
                f"Val R@1 {validation['recall_at_1']:.4f} | "
                f"Val MRR {validation['mrr']:.4f}",
                flush=True,
            )

        if stale_epochs >= PATIENCE:
            print(
                f"  Early stopping at epoch {epoch}",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError(
            f"No valid checkpoint for seed {seed}."
        )

    model.load_state_dict(best_state)

    best_validation = evaluate(
        model,
        validation_pairs,
    )

    # Save only the best validation-selected checkpoint.
    payload = {
        "model_state_dict": best_state,
        "model_name": "shared_mlp",
        "input_dim": FEATURE_DIM,
        "hidden_dim": HIDDEN_DIM,
        "output_dim": FEATURE_DIM,
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_metrics": best_validation,
        "graph_archive_sha256": graph_hash,
        "protocol_sha256": protocol_hash,
        "training_region_count": 40,
        "validation_region_count": 10,
        "test_evaluated": False,
    }

    with checkpoint_paths[seed].open("xb") as stream:
        torch.save(payload, stream)

    checkpoint_hashes[str(seed)] = sha256(
        checkpoint_paths[seed]
    )

    summary_rows.append({
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_loss": best_validation["loss"],
        "validation_recall_at_1":
            best_validation["recall_at_1"],
        "validation_mrr": best_validation["mrr"],
        "validation_reverse_recall_at_1":
            best_validation["reverse_recall_at_1"],
        "validation_reverse_mrr":
            best_validation["reverse_mrr"],
        "median_positive_cosine":
            best_validation["median_positive_cosine"],
        "checkpoint_sha256": checkpoint_hashes[str(seed)],
    })

    print(
        f"  Best epoch: {best_epoch} | "
        f"Best validation Recall@1: "
        f"{best_validation['recall_at_1']:.4f}",
        flush=True,
    )


# ---------------------------------------------------------
# 4. Save training results
# ---------------------------------------------------------

print("\n[4] Saving experimental results...", flush=True)

HISTORY_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

write_csv_new(
    HISTORY_PATH,
    history_rows,
    [
        "seed",
        "epoch",
        "train_loss",
        "validation_loss",
        "validation_recall_at_1",
        "validation_mrr",
        "validation_reverse_recall_at_1",
        "validation_reverse_mrr",
        "best_checkpoint",
    ],
)

write_csv_new(
    SUMMARY_PATH,
    summary_rows,
    [
        "seed",
        "best_epoch",
        "validation_loss",
        "validation_recall_at_1",
        "validation_mrr",
        "validation_reverse_recall_at_1",
        "validation_reverse_mrr",
        "median_positive_cosine",
        "checkpoint_sha256",
    ],
)

validation_scores = np.array(
    [
        r["validation_recall_at_1"]
        for r in summary_rows
    ],
    dtype=np.float64,
)

report = {
    "phase": "4.2",
    "step": "4.2.2",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "experiment": "shared_mlp_baseline",
    "architecture": "384-256-384_GELU",
    "graph_message_passing": False,
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
    "checkpoint_selection_metric":
        "validation_recall_at_1_then_loss",
    "seed_results": summary_rows,
    "mean_validation_recall_at_1": float(
        validation_scores.mean()
    ),
    "std_validation_recall_at_1": float(
        validation_scores.std(ddof=1)
    ),
    "history_sha256": sha256(HISTORY_PATH),
    "summary_sha256": sha256(SUMMARY_PATH),
    "checkpoint_hashes": checkpoint_hashes,
    "biological_independence_verified": False,
    "pixel_perfect_registration_assumed": False,
    "xai_planned": True,
    "limitations": [
        "Small number of training regions",
        "Training pairs share original biological material",
        "Within-region retrieval has only 36 candidates",
        "Validation regions are spatial but not biologically independent",
        "Only two scanners included",
        "Test data intentionally not evaluated",
    ],
}

with REPORT_PATH.open(
    "x",
    encoding="utf-8",
) as stream:
    json.dump(
        report,
        stream,
        indent=2,
    )

print("\n" + "=" * 70)
print("PHASE 4.2.2 - MLP BASELINE COMPLETED")
print("=" * 70)

for row in summary_rows:
    print(
        f"Seed {row['seed']} | "
        f"Best epoch {row['best_epoch']} | "
        f"Validation Recall@1 "
        f"{row['validation_recall_at_1']:.4f} | "
        f"MRR {row['validation_mrr']:.4f}"
    )

print(
    "\nMean validation Recall@1:",
    f"{validation_scores.mean():.4f}",
)

print(
    "Standard deviation:",
    f"{validation_scores.std(ddof=1):.4f}",
)

print("\nSaved checkpoints:")
for path in checkpoint_paths.values():
    print(path)

print("\nSaved:")
print(HISTORY_PATH)
print(SUMMARY_PATH)
print(REPORT_PATH)

print("\nTest partition: NOT EVALUATED")
print("Graph edges: NOT USED")
print("Original images and features: UNCHANGED")
print("XAI evaluation: PLANNED")
print("\nSTEP 4.2.2 COMPLETED")
