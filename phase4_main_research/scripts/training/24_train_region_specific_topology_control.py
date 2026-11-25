
"""
PHASE 4.2.9 - REGION-SPECIFIC TOPOLOGY CONTROL

Question:
Can GraphSAGE maintain cross-scanner matching performance
when graph connectivity is randomized independently
for every spatial region?

Control:
- Each region gets a deterministic random permutation
  of the original 36-node spatial adjacency.
- AT2 and GT450 share their region's topology.
- Different regions get independently sampled topologies.
- Features and node correspondence remain unchanged.
- Undirected edge count and degree multiset are preserved.

Training:
- Same GraphSAGE architecture as Phase 4.2.4
- Same contrastive objective and hyperparameters
- Seeds: 42, 123, 2026
- Train: 40 paired regions
- Validation: 10 paired regions
- Test: reserved

No modification of previous research artifacts.
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
SELF_REPORT = BASE / "outputs/reports/22_self_only_graphsage_report.json"

CHECKPOINT_DIR = BASE / "outputs/checkpoints/24_region_topology"
HISTORY = BASE / "outputs/tables/24_region_topology_history.csv"
SUMMARY = BASE / "outputs/tables/24_region_topology_summary.csv"
REPORT = BASE / "outputs/reports/24_region_topology_report.json"

SEEDS = (42, 123, 2026)
TEMPERATURE = 0.07
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_REGIONS = 8
MAX_EPOCHS = 100
PATIENCE = 20

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            digest.update(block)
    return digest.hexdigest()


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


def original_neighbor_matrix(edges):
    if edges.shape != (2, 120):
        raise RuntimeError("Unexpected edge-index shape.")

    adjacency = np.zeros((36, 36), dtype=np.float32)
    seen = set()

    for source, target in edges.T.tolist():
        if (
            source == target
            or not 0 <= source < 36
            or not 0 <= target < 36
        ):
            raise RuntimeError("Invalid graph edge.")

        if (source, target) in seen:
            raise RuntimeError("Duplicate graph edge.")

        seen.add((source, target))
        adjacency[target, source] = 1.0

    if len(seen) != 120 or any(
        (b, a) not in seen for a, b in seen
    ):
        raise RuntimeError("Invalid reciprocal topology.")

    return adjacency


def region_topology(adjacency, seed, region):
    """
    Derive a stable region-specific permutation.

    The same region receives the same topology for
    both scanner views, including during validation.
    """

    key = f"phase429|{seed}|{region}".encode("utf-8")
    seed_bytes = hashlib.sha256(key).digest()

    local_seed = int.from_bytes(
        seed_bytes[:8], byteorder="little"
    )

    rng = np.random.default_rng(local_seed)

    for _ in range(200):
        permutation = rng.permutation(36)

        scrambled = adjacency[
            np.ix_(permutation, permutation)
        ]

        if np.array_equal(scrambled, adjacency):
            continue

        old_degree = np.sort(adjacency.sum(axis=1))
        new_degree = np.sort(scrambled.sum(axis=1))

        if not np.array_equal(old_degree, new_degree):
            raise RuntimeError("Degree multiset changed.")

        if int(scrambled.sum()) != 120:
            raise RuntimeError("Edge count changed.")

        if not np.array_equal(scrambled, scrambled.T):
            raise RuntimeError("Adjacency lost symmetry.")

        degree = scrambled.sum(axis=1, keepdims=True)

        return (
            scrambled / degree,
            permutation.astype(np.int64),
        )

    raise RuntimeError(
        f"Cannot generate topology for {region}"
    )


class SAGEConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neighbor_linear = nn.Linear(
            in_dim, out_dim, bias=False
        )

    def forward(self, x, weights):
        neighbors = torch.bmm(weights, x)
        return (
            self.self_linear(x)
            + self.neighbor_linear(neighbors)
        )


class GraphSAGE(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = SAGEConv(384, 256)
        self.conv2 = SAGEConv(256, 384)

    def forward(self, x, adjacency):
        h = F.gelu(self.conv1(x, adjacency))
        z = self.conv2(h, adjacency)

        return F.normalize(z, p=2, dim=-1)


def loss_fn(a, b):
    logits = (
        a @ b.transpose(-1, -2)
    ) / TEMPERATURE

    batch, nodes, _ = logits.shape

    labels = torch.arange(
        nodes, device=logits.device
    ).expand(batch, nodes).reshape(-1)

    return 0.5 * (
        F.cross_entropy(
            logits.reshape(-1, nodes), labels
        )
        + F.cross_entropy(
            logits.transpose(-1, -2).reshape(-1, nodes),
            labels
        )
    )


@torch.inference_mode()
def evaluate(model, data, adjacencies):
    model.eval()

    tensor = torch.as_tensor(
        data, dtype=torch.float32, device=DEVICE
    )
    weights = torch.as_tensor(
        adjacencies, dtype=torch.float32, device=DEVICE
    )

    a = model(tensor[:, 0], weights)
    b = model(tensor[:, 1], weights)

    similarity = a @ b.transpose(-1, -2)
    reverse = similarity.transpose(-1, -2)

    batch, nodes, _ = similarity.shape
    labels = torch.arange(
        nodes, device=DEVICE
    ).expand(batch, nodes)

    positives = similarity.diagonal(
        dim1=-2, dim2=-1
    )

    rank_forward = 1 + (
        similarity > positives.unsqueeze(-1)
    ).sum(-1)

    rank_reverse = 1 + (
        reverse > positives.unsqueeze(-1)
    ).sum(-1)

    return {
        "loss": float(loss_fn(a, b).item()),
        "recall_at_1": float(
            (similarity.argmax(-1) == labels)
            .float().mean().item()
        ),
        "mrr": float(
            (1.0 / rank_forward.float()).mean().item()
        ),
        "reverse_recall_at_1": float(
            (reverse.argmax(-1) == labels)
            .float().mean().item()
        ),
        "reverse_mrr": float(
            (1.0 / rank_reverse.float()).mean().item()
        ),
        "median_positive_cosine": float(
            positives.median().item()
        ),
    }


def better(current, best):
    if best is None:
        return True

    difference = (
        current["recall_at_1"] - best["recall_at_1"]
    )

    return (
        difference > 1e-10
        or (
            abs(difference) <= 1e-10
            and current["loss"] < best["loss"] - 1e-8
        )
    )


print("=" * 70, flush=True)
print("PHASE 4.2.9 - REGION-SPECIFIC TOPOLOGY CONTROL", flush=True)
print("=" * 70, flush=True)


# --------------------------------------------------
# 1. Verify provenance
# --------------------------------------------------

print("\n[1] Verifying source experiments...", flush=True)

for path in (
    GRAPH, GRAPH_REPORT, PROTOCOL,
    SAGE_REPORT, PERM_REPORT, SELF_REPORT
):
    if not path.is_file():
        raise RuntimeError(f"Missing source: {path}")

for path in (HISTORY, SUMMARY, REPORT):
    if path.exists():
        raise RuntimeError(
            f"Refusing to overwrite output: {path}"
        )

graph_hash = file_hash(GRAPH)
protocol_hash = file_hash(PROTOCOL)

if read_json(GRAPH_REPORT)["graph_archive_sha256"] != graph_hash:
    raise RuntimeError("Graph provenance mismatch.")

if read_json(PROTOCOL)["source_graph_sha256"] != graph_hash:
    raise RuntimeError("Protocol provenance mismatch.")

sage = read_json(SAGE_REPORT)
permuted = read_json(PERM_REPORT)
self_only = read_json(SELF_REPORT)

for previous in (sage, permuted, self_only):
    if (
        previous["graph_archive_sha256"] != graph_hash
        or previous["protocol_sha256"] != protocol_hash
    ):
        raise RuntimeError(
            "Baseline provenance mismatch."
        )

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

checkpoint_paths = {
    seed: CHECKPOINT_DIR / f"region_topology_seed_{seed}.pt"
    for seed in SEEDS
}

if any(p.exists() for p in checkpoint_paths.values()):
    raise RuntimeError("Existing checkpoint detected.")

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

adjacency = original_neighbor_matrix(edges)

print("Graph tensor:", features.shape)
print("Device:", DEVICE)


# --------------------------------------------------
# 2. Assemble region pairs
# --------------------------------------------------

print("\n[2] Preparing training and validation regions...", flush=True)

by_region = defaultdict(dict)
region_splits = {}

for index, region in enumerate(regions):
    scanner = scanners[index]
    split = splits[index]

    if scanner not in ("AT2", "GT450"):
        raise RuntimeError("Unknown scanner.")

    if split not in ("train", "validation", "test"):
        raise RuntimeError("Unknown partition.")

    if scanner in by_region[region]:
        raise RuntimeError("Duplicate scanner graph.")

    by_region[region][scanner] = index

    if region in region_splits and region_splits[region] != split:
        raise RuntimeError("Region crosses partitions.")

    region_splits[region] = split

if dict(Counter(region_splits.values())) != {
    "train": 40, "validation": 10, "test": 10
}:
    raise RuntimeError("Unexpected region partition.")

paired = {"train": [], "validation": []}
region_names = {"train": [], "validation": []}

for region in sorted(by_region):
    pair = by_region[region]

    if set(pair) != {"AT2", "GT450"}:
        raise RuntimeError("Incomplete scanner pair.")

    a, b = pair["AT2"], pair["GT450"]

    if not np.array_equal(tiles[a], tiles[b]):
        raise RuntimeError("Tile ordering mismatch.")

    split = region_splits[region]

    if split == "test":
        continue

    paired[split].append(
        np.stack([features[a], features[b]], axis=0)
    )

    region_names[split].append(region)

train = np.stack(paired["train"]).astype(np.float32)
validation = np.stack(paired["validation"]).astype(np.float32)

if train.shape != (40, 2, 36, 384):
    raise RuntimeError("Training tensor shape mismatch.")

if validation.shape != (10, 2, 36, 384):
    raise RuntimeError("Validation tensor shape mismatch.")

print("Training pairs:", len(train))
print("Validation pairs:", len(validation))
print("Test pairs: 10, reserved")


# --------------------------------------------------
# 3. Train models with per-region topology
# --------------------------------------------------

print("\n[3] Training region-specific GraphSAGE...", flush=True)

history = []
summaries = []
checkpoint_hashes = {}

for seed in SEEDS:
    print(f"\nSeed: {seed}", flush=True)
    set_seed(seed)

    all_regions = (
        region_names["train"]
        + region_names["validation"]
    )

    topology_map = {}
    permutation_map = {}

    for region in all_regions:
        weights, permutation = region_topology(
            adjacency, seed, region
        )

        topology_map[region] = weights
        permutation_map[region] = permutation.tolist()

    train_weights = np.stack([
        topology_map[r]
        for r in region_names["train"]
    ]).astype(np.float32)

    val_weights = np.stack([
        topology_map[r]
        for r in region_names["validation"]
    ]).astype(np.float32)

    if train_weights.shape != (40, 36, 36):
        raise RuntimeError("Train topology tensor mismatch.")

    if val_weights.shape != (10, 36, 36):
        raise RuntimeError("Validation topology tensor mismatch.")

    model = GraphSAGE().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    rng = np.random.default_rng(seed)

    best = None
    best_state = None
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

            weights = torch.as_tensor(
                train_weights[indices],
                dtype=torch.float32,
                device=DEVICE,
            )

            a = model(batch[:, 0], weights)
            b = model(batch[:, 1], weights)

            loss = loss_fn(a, b)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_total += float(loss.item()) * len(indices)
            seen += len(indices)

        train_loss = loss_total / seen
        metrics = evaluate(model, validation, val_weights)

        improved = better(metrics, best)

        if improved:
            best = dict(metrics)
            best_epoch = epoch
            best_state = {
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

    if best_state is None:
        raise RuntimeError("No best checkpoint.")

    model.load_state_dict(best_state)
    final = evaluate(model, validation, val_weights)

    checkpoint = checkpoint_paths[seed]

    with checkpoint.open("xb") as stream:
        torch.save({
            "model_state_dict": best_state,
            "model_name": "region_specific_topology_graphsage",
            "seed": seed,
            "best_epoch": best_epoch,
            "validation_metrics": final,
            "region_permutations": permutation_map,
            "graph_archive_sha256": graph_hash,
            "protocol_sha256": protocol_hash,
            "test_evaluated": False,
        }, stream)

    checkpoint_hashes[str(seed)] = file_hash(checkpoint)

    original = next(
        row for row in sage["seed_results"]
        if int(row["seed"]) == seed
    )["validation_recall_at_1"]

    fixed_perm = next(
        row for row in permuted["seed_results"]
        if int(row["seed"]) == seed
    )["validation_recall_at_1"]

    self_score = next(
        row for row in self_only["seed_results"]
        if int(row["seed"]) == seed
    )["validation_recall_at_1"]

    summaries.append({
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_loss": final["loss"],
        "validation_recall_at_1": final["recall_at_1"],
        "validation_mrr": final["mrr"],
        "original_graphsage_recall_at_1": original,
        "fixed_permutation_recall_at_1": fixed_perm,
        "self_only_recall_at_1": self_score,
        "checkpoint_sha256": checkpoint_hashes[str(seed)],
    })

    print(
        f"  Best epoch {best_epoch} | "
        f"Region-specific R@1 {final['recall_at_1']:.4f}",
        flush=True,
    )


# --------------------------------------------------
# 4. Save results
# --------------------------------------------------

print("\n[4] Saving controlled experiment...", flush=True)

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
        "original_graphsage_recall_at_1",
        "fixed_permutation_recall_at_1",
        "self_only_recall_at_1",
        "checkpoint_sha256",
    ],
)

new_scores = np.array([
    row["validation_recall_at_1"]
    for row in summaries
], dtype=np.float64)

original_scores = np.array([
    row["original_graphsage_recall_at_1"]
    for row in summaries
], dtype=np.float64)

fixed_scores = np.array([
    row["fixed_permutation_recall_at_1"]
    for row in summaries
], dtype=np.float64)

report = {
    "phase": "4.2",
    "step": "4.2.9",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "experiment": "region_specific_topology_control",
    "architecture": "GraphSAGE_384_256_384_GELU",
    "randomization": (
        "independent_per_region_and_seed; "
        "shared_between_scanners"
    ),
    "degree_multiset_preserved": True,
    "graph_edges_per_region": 120,
    "input_features_unchanged": True,
    "model_hyperparameters_matched": True,
    "seeds": list(SEEDS),
    "train_region_pairs": 40,
    "validation_region_pairs": 10,
    "test_region_pairs_reserved": 10,
    "test_evaluated": False,
    "graph_archive_sha256": graph_hash,
    "protocol_sha256": protocol_hash,
    "mean_region_specific_recall_at_1": float(
        new_scores.mean()
    ),
    "std_region_specific_recall_at_1": float(
        new_scores.std(ddof=1)
    ),
    "mean_original_graphsage_recall_at_1": float(
        original_scores.mean()
    ),
    "mean_fixed_permutation_recall_at_1": float(
        fixed_scores.mean()
    ),
    "original_minus_region_specific_pp": float(
        (original_scores - new_scores).mean() * 100
    ),
    "fixed_minus_region_specific_pp": float(
        (fixed_scores - new_scores).mean() * 100
    ),
    "seed_results": summaries,
    "checkpoint_hashes": checkpoint_hashes,
    "history_sha256": file_hash(HISTORY),
    "summary_sha256": file_hash(SUMMARY),
    "limitations": [
        "Only three seed-specific randomization families",
        "Paired scanners share each region's random topology",
        "Degree-preserving permutation does not preserve "
        "each original node's degree",
        "Validation selection uses 36-candidate Recall@1",
        "Validation retrieval is near saturation",
        "No independent biological validation",
        "Test partition untouched",
    ],
    "xai_planned": True,
    "biological_independence_verified": False,
}

save_json(REPORT, report)

print("\n" + "=" * 70)
print("PHASE 4.2.9 - REGION TOPOLOGY CONTROL COMPLETED")
print("=" * 70)

for row in summaries:
    print(
        f"Seed {row['seed']} | "
        f"Original {row['original_graphsage_recall_at_1']:.4f} | "
        f"Fixed permuted {row['fixed_permutation_recall_at_1']:.4f} | "
        f"Region-specific {row['validation_recall_at_1']:.4f}"
    )

print(
    "\nOriginal mean R@1:",
    f"{original_scores.mean():.4f}"
)
print(
    "Fixed-permuted mean R@1:",
    f"{fixed_scores.mean():.4f}"
)
print(
    "Region-specific mean R@1:",
    f"{new_scores.mean():.4f}"
)
print(
    "Original minus region-specific:",
    f"{report['original_minus_region_specific_pp']:+.2f} pp"
)

print("\nSaved:")
print(HISTORY)
print(SUMMARY)
print(REPORT)

print("\nTest partition: NOT EVALUATED")
print("Previous experiments: UNCHANGED")
print("Proposed model: NOT TRAINED")
print("XAI: PLANNED")
print("\nSTEP 4.2.9 COMPLETED")
