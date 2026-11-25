
"""
PHASE 4.2.6 - SPATIAL TOPOLOGY PERMUTATION CONTROL

Question:
Does anatomically correct spatial connectivity provide
an advantage over a degree-preserving scrambled graph?

Model:
    Identical GraphSAGE architecture to Phase 4.2.4
    384 -> 256 -> 384, GELU, L2 normalization

Control:
    Fixed random permutation of the node labels
    applied to the spatial adjacency matrix.

    The model still processes features in their
    original tile order.

Training protocol:
    Seeds: 42, 123, 2026
    Train: 40 paired regions
    Validation: 10 paired regions
    Test: excluded
    Same optimizer, objective, epochs, patience,
    batch size and selection criterion as GraphSAGE.

This is a topology-specific negative control.
It is not proof of biological generalization.
"""

from pathlib import Path
from collections import Counter, defaultdict
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

GRAPH_PATH = BASE / "outputs/features/15_stage_a_spatial_graphs.npz"
GRAPH_REPORT = BASE / "outputs/reports/15_graph_dataset_report.json"
PROTOCOL_PATH = BASE / "configs/16_initial_gnn_experimental_protocol.json"
SAGE_REPORT = BASE / "outputs/reports/19_graphsage_baseline_report.json"

CHECKPOINT_DIR = BASE / "outputs/checkpoints/21_permuted_topology"
HISTORY_PATH = BASE / "outputs/tables/21_permuted_training_history.csv"
SUMMARY_PATH = BASE / "outputs/tables/21_permuted_validation_summary.csv"
REPORT_PATH = BASE / "outputs/reports/21_permuted_topology_report.json"

SEEDS = (42, 123, 2026)

TEMPERATURE = 0.07
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 20
BATCH_REGIONS = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_csv_new(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_json_new(path, data):
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


def build_neighbor_matrix(edges):
    matrix = torch.zeros((36, 36), dtype=torch.float32)
    seen = set()

    if edges.shape != (2, 120):
        raise RuntimeError("Incorrect edge index shape.")

    for source, target in edges.T.tolist():
        if not (0 <= source < 36 and 0 <= target < 36):
            raise RuntimeError("Invalid edge endpoint.")

        if source == target or (source, target) in seen:
            raise RuntimeError("Invalid or duplicate edge.")

        seen.add((source, target))
        matrix[target, source] = 1.0

    if len(seen) != 120:
        raise RuntimeError("Expected 120 directed edges.")

    if any((v, u) not in seen for u, v in seen):
        raise RuntimeError("Nonreciprocal edge.")

    degree = matrix.sum(-1, keepdim=True)

    if torch.any(degree == 0):
        raise RuntimeError("Unexpected isolated node.")

    return matrix / degree


def permute_topology(matrix, seed):
    """
    Conjugate the original adjacency by a permutation.

    Node features are NOT permuted.

    This preserves the original graph's degree
    multiset and number of edges, but changes
    which positions are considered neighbors.
    """
    rng = np.random.default_rng(seed + 100_000)

    original_degree = (matrix > 0).sum(-1)

    for _ in range(100):
        permutation = rng.permutation(36)

        if np.array_equal(permutation, np.arange(36)):
            continue

        candidate = matrix[
            permutation, :
        ][:, permutation]

        if torch.equal(candidate, matrix):
            continue

        new_degree = (candidate > 0).sum(-1)

        if not torch.equal(
            torch.sort(original_degree).values,
            torch.sort(new_degree).values,
        ):
            raise RuntimeError("Degree distribution changed.")

        if (candidate > 0).sum().item() != 120:
            raise RuntimeError("Edge count changed.")

        return (
            candidate.clone(),
            permutation.astype(np.int64),
        )

    raise RuntimeError("Cannot produce different topology.")


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


class SpatialGraphSAGE(nn.Module):
    def __init__(self, weights):
        super().__init__()

        self.register_buffer(
            "neighbor_weights",
            weights.clone(),
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

    batch_size, nodes, _ = logits.shape

    labels = torch.arange(
        nodes,
        device=logits.device,
    ).expand(batch_size, nodes).reshape(-1)

    forward = F.cross_entropy(
        logits.reshape(-1, nodes),
        labels,
    )

    reverse = F.cross_entropy(
        logits.transpose(-1, -2).reshape(-1, nodes),
        labels,
    )

    return 0.5 * (forward + reverse)


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

    batch_size, nodes, _ = similarity.shape

    targets = torch.arange(
        nodes,
        device=DEVICE,
    ).expand(batch_size, nodes)

    positive = similarity.diagonal(
        dim1=-2, dim2=-1
    )

    ranks = 1 + (
        similarity > positive.unsqueeze(-1)
    ).sum(-1)

    reverse_ranks = 1 + (
        reverse > positive.unsqueeze(-1)
    ).sum(-1)

    return {
        "loss": float(contrastive_loss(a, b).item()),
        "recall_at_1": float(
            (similarity.argmax(-1) == targets)
            .float().mean().item()
        ),
        "mrr": float(
            (1.0 / ranks.float()).mean().item()
        ),
        "reverse_recall_at_1": float(
            (reverse.argmax(-1) == targets)
            .float().mean().item()
        ),
        "reverse_mrr": float(
            (1.0 / reverse_ranks.float()).mean().item()
        ),
        "median_positive_cosine": float(
            positive.median().item()
        ),
    }


def better(current, best):
    if best is None:
        return True

    difference = (
        current["recall_at_1"]
        - best["recall_at_1"]
    )

    return (
        difference > 1e-10
        or (
            abs(difference) <= 1e-10
            and current["loss"] < best["loss"] - 1e-8
        )
    )


print("=" * 70, flush=True)
print("PHASE 4.2.6 - PERMUTED TOPOLOGY CONTROL", flush=True)
print("=" * 70, flush=True)


# ------------------------------------------------------
# 1. Validate sources
# ------------------------------------------------------

print("\n[1] Validating sources...", flush=True)

for path in (
    GRAPH_PATH, GRAPH_REPORT,
    PROTOCOL_PATH, SAGE_REPORT,
):
    if not path.is_file():
        raise RuntimeError(f"Missing input: {path}")

for path in (HISTORY_PATH, SUMMARY_PATH, REPORT_PATH):
    if path.exists():
        raise RuntimeError(f"Output exists: {path}")

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

checkpoint_paths = {
    seed: CHECKPOINT_DIR / f"permuted_seed_{seed}.pt"
    for seed in SEEDS
}

if any(path.exists() for path in checkpoint_paths.values()):
    raise RuntimeError("Existing checkpoint; refusing overwrite.")

graph_hash = sha256(GRAPH_PATH)
protocol_hash = sha256(PROTOCOL_PATH)

graph_report = read_json(GRAPH_REPORT)
protocol = read_json(PROTOCOL_PATH)
sage_report = read_json(SAGE_REPORT)

if graph_hash != graph_report["graph_archive_sha256"]:
    raise RuntimeError("Graph SHA256 mismatch.")

if graph_hash != protocol["source_graph_sha256"]:
    raise RuntimeError("Protocol provenance mismatch.")

if (
    sage_report["graph_archive_sha256"] != graph_hash
    or sage_report["protocol_sha256"] != protocol_hash
):
    raise RuntimeError("GraphSAGE provenance mismatch.")

with np.load(GRAPH_PATH, allow_pickle=False) as z:
    features = z["node_features"].copy()
    edges = z["edge_index"].copy()
    regions = z["region_id"].astype(str)
    splits = z["spatial_split"].astype(str)
    scanners = z["scanner"].astype(str)
    tiles = z["tile_id"].astype(str)

if features.shape != (120, 36, 384):
    raise RuntimeError("Invalid graph features.")

if features.dtype != np.float32:
    raise RuntimeError("Unexpected datatype.")

if not np.isfinite(features).all():
    raise RuntimeError("Non-finite graph features.")

spatial_weights = build_neighbor_matrix(edges)

print("Graphs:", features.shape[0])
print("Original directed edges:", edges.shape[1])


# ------------------------------------------------------
# 2. Prepare train and validation only
# ------------------------------------------------------

print("\n[2] Preparing paired partitions...", flush=True)

by_region = defaultdict(dict)
region_splits = {}

for i, region in enumerate(regions):
    scanner = scanners[i]
    split = splits[i]

    if scanner not in ("AT2", "GT450"):
        raise RuntimeError("Unexpected scanner.")

    if split not in ("train", "validation", "test"):
        raise RuntimeError("Unexpected split.")

    if scanner in by_region[region]:
        raise RuntimeError("Duplicate scanner graph.")

    by_region[region][scanner] = i

    if region in region_splits and region_splits[region] != split:
        raise RuntimeError("Region crossed partitions.")

    region_splits[region] = split

if dict(Counter(region_splits.values())) != {
    "train": 40, "validation": 10, "test": 10
}:
    raise RuntimeError("Invalid region splits.")

paired = {"train": [], "validation": []}

for region in sorted(by_region):
    pair = by_region[region]

    if set(pair) != {"AT2", "GT450"}:
        raise RuntimeError("Incomplete paired region.")

    ai, bi = pair["AT2"], pair["GT450"]

    if not np.array_equal(tiles[ai], tiles[bi]):
        raise RuntimeError("Different paired node orders.")

    split = region_splits[region]

    if split == "test":
        continue

    paired[split].append(
        np.stack([features[ai], features[bi]], axis=0)
    )

train = np.stack(paired["train"]).astype(np.float32)
validation = np.stack(paired["validation"]).astype(np.float32)

if train.shape != (40, 2, 36, 384):
    raise RuntimeError("Incorrect training shape.")

if validation.shape != (10, 2, 36, 384):
    raise RuntimeError("Incorrect validation shape.")

print("Train regions:", len(train))
print("Validation regions:", len(validation))
print("Test regions: 10, reserved")


# ------------------------------------------------------
# 3. Train using permuted topology
# ------------------------------------------------------

print("\n[3] Training permuted-topology GraphSAGE...", flush=True)

history = []
summaries = []
checkpoint_hashes = {}

for seed in SEEDS:
    print(f"\nSeed: {seed}", flush=True)

    set_seed(seed)

    scrambled_weights, permutation = permute_topology(
        spatial_weights, seed
    )

    model = SpatialGraphSAGE(
        scrambled_weights.to(DEVICE)
    ).to(DEVICE)

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

        loss_total = 0.0
        seen = 0

        for start in range(
            0, len(train), BATCH_REGIONS
        ):
            indices = order[start:start + BATCH_REGIONS]

            batch = torch.as_tensor(
                train[indices],
                dtype=torch.float32,
                device=DEVICE,
            )

            a = model(batch[:, 0])
            b = model(batch[:, 1])

            loss = contrastive_loss(a, b)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_total += float(loss.item()) * len(indices)
            seen += len(indices)

        train_loss = loss_total / seen

        metrics = evaluate(model, validation)

        improved = better(metrics, best)

        if improved:
            best = dict(metrics)
            best_epoch = epoch
            best_weights = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
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
            print(
                f"  Early stopping: epoch {epoch}",
                flush=True,
            )
            break

    if best_weights is None:
        raise RuntimeError("No best checkpoint.")

    model.load_state_dict(best_weights)
    final = evaluate(model, validation)

    checkpoint = checkpoint_paths[seed]

    with checkpoint.open("xb") as f:
        torch.save({
            "model_state_dict": best_weights,
            "model_name": "permuted_graphsage",
            "seed": seed,
            "best_epoch": best_epoch,
            "permutation": permutation.tolist(),
            "validation_metrics": final,
            "graph_archive_sha256": graph_hash,
            "protocol_sha256": protocol_hash,
            "test_evaluated": False,
        }, f)

    checkpoint_hashes[str(seed)] = sha256(checkpoint)

    original_seed_result = next(
        r for r in sage_report["seed_results"]
        if int(r["seed"]) == seed
    )

    original_recall = float(
        original_seed_result["validation_recall_at_1"]
    )

    summaries.append({
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_loss": final["loss"],
        "validation_recall_at_1": final["recall_at_1"],
        "validation_mrr": final["mrr"],
        "original_graphsage_recall_at_1": original_recall,
        "original_minus_permuted_pp": (
            original_recall - final["recall_at_1"]
        ) * 100,
        "checkpoint_sha256": checkpoint_hashes[str(seed)],
    })

    print(
        f"  Best epoch: {best_epoch} | "
        f"Permuted R@1: {final['recall_at_1']:.4f} | "
        f"Original R@1: {original_recall:.4f}",
        flush=True,
    )


# ------------------------------------------------------
# 4. Save comparison
# ------------------------------------------------------

print("\n[4] Saving topology-control results...", flush=True)

save_csv_new(
    HISTORY_PATH,
    history,
    [
        "seed", "epoch", "train_loss",
        "validation_loss", "validation_recall_at_1",
        "validation_mrr", "best_checkpoint",
    ],
)

save_csv_new(
    SUMMARY_PATH,
    summaries,
    [
        "seed", "best_epoch", "validation_loss",
        "validation_recall_at_1", "validation_mrr",
        "original_graphsage_recall_at_1",
        "original_minus_permuted_pp",
        "checkpoint_sha256",
    ],
)

permuted_scores = np.array([
    r["validation_recall_at_1"]
    for r in summaries
])

original_scores = np.array([
    r["original_graphsage_recall_at_1"]
    for r in summaries
])

report = {
    "phase": "4.2",
    "step": "4.2.6",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "experiment": "degree_preserving_topology_permutation",
    "model": "GraphSAGE_384_256_384",
    "permutation_policy": (
        "fixed_independent_per_seed_same_train_validation_topology"
    ),
    "node_features_permuted": False,
    "degree_multiset_preserved": True,
    "edge_count_preserved": True,
    "model_hyperparameters_matched": True,
    "seeds": list(SEEDS),
    "train_regions": 40,
    "validation_regions": 10,
    "test_regions_reserved": 10,
    "graph_archive_sha256": graph_hash,
    "protocol_sha256": protocol_hash,
    "mean_original_recall_at_1": float(
        original_scores.mean()
    ),
    "mean_permuted_recall_at_1": float(
        permuted_scores.mean()
    ),
    "original_minus_permuted_pp": float(
        (original_scores - permuted_scores).mean() * 100
    ),
    "seed_results": summaries,
    "checkpoint_hashes": checkpoint_hashes,
    "history_sha256": sha256(HISTORY_PATH),
    "summary_sha256": sha256(SUMMARY_PATH),
    "test_evaluated": False,
    "biological_independence_verified": False,
    "xai_planned": True,
    "limitations": [
        "Only one permuted graph per random seed",
        "Permutation may create nonlocal edges",
        "Small spatial validation set",
        "Results do not establish biological generalization",
        "Original and permuted models are selected on validation data",
    ],
}

save_json_new(REPORT_PATH, report)

print("\n" + "=" * 70)
print("PHASE 4.2.6 - TOPOLOGY CONTROL COMPLETED")
print("=" * 70)

for row in summaries:
    print(
        f"Seed {row['seed']} | "
        f"Original {row['original_graphsage_recall_at_1']:.4f} | "
        f"Permuted {row['validation_recall_at_1']:.4f} | "
        f"Difference {row['original_minus_permuted_pp']:+.2f} pp"
    )

print(
    "\nMean original GraphSAGE R@1:",
    f"{original_scores.mean():.4f}"
)

print(
    "Mean permuted GraphSAGE R@1:",
    f"{permuted_scores.mean():.4f}"
)

print(
    "Mean original - permuted:",
    f"{report['original_minus_permuted_pp']:+.2f} pp"
)

print("\nSaved:")
print(HISTORY_PATH)
print(SUMMARY_PATH)
print(REPORT_PATH)

print("\nTest partition: NOT EVALUATED")
print("Original GraphSAGE checkpoints: UNCHANGED")
print("Original graph archive: UNCHANGED")
print("Proposed GNN: NOT YET TRAINED")
print("XAI: PLANNED")
print("\nSTEP 4.2.6 COMPLETED")
