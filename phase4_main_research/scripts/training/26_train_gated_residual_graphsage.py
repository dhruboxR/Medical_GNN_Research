
"""
PHASE 4.2.11 - GATED RESIDUAL GRAPHSAGE V1

Architecture:
    Frozen DINOv2 CLS features (384D)
    Identity pathway: normalize(original feature)
    Graph pathway: two-layer GraphSAGE, 384-256-384
    Learned scalar gate, initialized at 0.1

    z = normalize(
        normalize(x) + alpha * normalize(GraphSAGE(x,A))
    )

    alpha = sigmoid(gate_logit), constrained to (0,1).

Objective:
    Bidirectional within-region contrastive loss,
    temperature 0.07.

Training protocol:
    40 train region pairs
    10 validation region pairs
    10 test region pairs reserved
    Seeds: 42, 123, 2026
    AdamW, lr=1e-3, weight_decay=1e-4
    Batch size 8 regions
    Maximum 100 epochs, patience 20

Validation:
    Primary: within-region AT2->GT450 Recall@1
    Secondary: MRR, reverse retrieval, node diversity,
               learned gate, 360-candidate retrieval.

No test evaluation, existing files unchanged.
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

PROTOCOL = (
    BASE / "configs/16_initial_gnn_experimental_protocol.json"
)

ARCH_CONFIG = (
    BASE / "configs/25_stability_aware_gnn_architecture_v1.json"
)

ARCH_REPORT = (
    BASE / "outputs/reports/25_architecture_protocol_report.json"
)

REFERENCE_REPORTS = {
    "MLP": BASE / "outputs/reports/17_mlp_baseline_report.json",
    "GCN": BASE / "outputs/reports/18_gcn_baseline_report.json",
    "GraphSAGE": BASE / "outputs/reports/19_graphsage_baseline_report.json",
    "FixedRandom": BASE / "outputs/reports/21_permuted_topology_report.json",
    "SelfOnly": BASE / "outputs/reports/22_self_only_graphsage_report.json",
    "RegionRandom": BASE / "outputs/reports/24_region_topology_report.json",
}

CHECKPOINT_DIR = (
    BASE / "outputs/checkpoints/26_gated_residual_graphsage"
)

HISTORY = (
    BASE / "outputs/tables/26_gated_residual_training_history.csv"
)

SUMMARY = (
    BASE / "outputs/tables/26_gated_residual_validation_summary.csv"
)

REGION_TABLE = (
    BASE / "outputs/tables/26_gated_residual_region_metrics.csv"
)

REPORT = (
    BASE / "outputs/reports/26_gated_residual_graphsage_report.json"
)

SEEDS = (42, 123, 2026)

DIM = 384
HIDDEN = 256
GATE_INITIAL = 0.1

TEMPERATURE = 0.07
LR = 1e-3
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


def load_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def save_csv_new(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "x", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_json_new(path, content):
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


def make_neighbor_matrix(edge_index):
    """
    Mean aggregation over four-neighbor spatial edges.
    Self information travels through separate SAGE
    transformations and the explicit residual pathway.
    """

    if edge_index.shape != (2, 120):
        raise RuntimeError("Unexpected graph edge shape.")

    adjacency = torch.zeros(
        (36, 36), dtype=torch.float32
    )

    seen = set()

    for source, target in edge_index.T.tolist():
        if not (
            0 <= source < 36
            and 0 <= target < 36
        ):
            raise RuntimeError("Invalid graph endpoint.")

        if source == target:
            raise RuntimeError("Unexpected self-loop.")

        if (source, target) in seen:
            raise RuntimeError("Duplicate spatial edge.")

        seen.add((source, target))
        adjacency[target, source] = 1.0

    if len(seen) != 120:
        raise RuntimeError("Unexpected edge count.")

    if any(
        (target, source) not in seen
        for source, target in seen
    ):
        raise RuntimeError("Nonreciprocal edge.")

    degree = adjacency.sum(
        dim=-1, keepdim=True
    )

    if (degree <= 0).any():
        raise RuntimeError("Isolated graph node.")

    return adjacency / degree


class SAGEConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.self_linear = nn.Linear(
            input_dim, output_dim
        )

        self.neighbor_linear = nn.Linear(
            input_dim, output_dim, bias=False
        )

    def forward(self, x, neighbor_weights):
        neighbors = torch.matmul(
            neighbor_weights, x
        )

        return (
            self.self_linear(x)
            + self.neighbor_linear(neighbors)
        )


class GatedResidualGraphSAGE(nn.Module):
    def __init__(self, neighbor_weights):
        super().__init__()

        self.register_buffer(
            "neighbor_weights",
            neighbor_weights.clone()
        )

        self.conv1 = SAGEConv(DIM, HIDDEN)
        self.conv2 = SAGEConv(HIDDEN, DIM)

        initial_logit = math.log(
            GATE_INITIAL / (1.0 - GATE_INITIAL)
        )

        self.gate_logit = nn.Parameter(
            torch.tensor(
                initial_logit,
                dtype=torch.float32
            )
        )

    def gate(self):
        return torch.sigmoid(self.gate_logit)

    def forward(self, x):
        identity = F.normalize(
            x, p=2, dim=-1
        )

        hidden = F.gelu(
            self.conv1(
                x, self.neighbor_weights
            )
        )

        graph_features = self.conv2(
            hidden, self.neighbor_weights
        )

        graph_features = F.normalize(
            graph_features, p=2, dim=-1
        )

        combined = (
            identity
            + self.gate() * graph_features
        )

        return F.normalize(
            combined, p=2, dim=-1
        )


def contrastive_loss(a, b):
    logits = (
        a @ b.transpose(-1, -2)
    ) / TEMPERATURE

    batch, nodes, _ = logits.shape

    targets = torch.arange(
        nodes, device=logits.device
    ).expand(batch, nodes).reshape(-1)

    forward = F.cross_entropy(
        logits.reshape(-1, nodes),
        targets
    )

    reverse = F.cross_entropy(
        logits.transpose(-1, -2).reshape(-1, nodes),
        targets
    )

    return 0.5 * (forward + reverse)


def retrieval_scores(a, b):
    """
    Both inputs contain matching nodes in the same order.

    Supports either 36 nodes or the concatenated
    360 validation candidates.
    """

    similarity = a @ b.transpose(-1, -2)

    targets = torch.arange(
        similarity.shape[-1],
        device=similarity.device
    )

    positives = similarity.diagonal(
        dim1=-2, dim2=-1
    )

    ranks = 1 + (
        similarity > positives.unsqueeze(-1)
    ).sum(-1)

    predictions = similarity.argmax(-1)

    recall1 = (
        predictions == targets
    ).float().mean()

    recall5 = (
        ranks <= 5
    ).float().mean()

    mrr = (
        1.0 / ranks.float()
    ).mean()

    return {
        "recall_at_1": float(recall1.item()),
        "recall_at_5": float(recall5.item()),
        "mrr": float(mrr.item()),
        "median_positive_cosine": float(
            positives.median().item()
        ),
    }


def node_diversity(z):
    """
    Mean cosine between different nodes in each graph.

    High cosine alone does not establish oversmoothing.
    """

    sim = z @ z.transpose(-1, -2)
    nodes = sim.size(-1)

    mask = ~torch.eye(
        nodes, dtype=torch.bool, device=z.device
    )

    return float(
        sim[..., mask].mean().item()
    )


@torch.inference_mode()
def evaluate(model, data, region_names=None):
    model.eval()

    tensor = torch.as_tensor(
        data, dtype=torch.float32, device=DEVICE
    )

    at2 = model(tensor[:, 0])
    gt450 = model(tensor[:, 1])

    within = retrieval_scores(at2, gt450)
    reverse = retrieval_scores(gt450, at2)

    result = {
        "loss": float(
            contrastive_loss(at2, gt450).item()
        ),
        "recall_at_1": within["recall_at_1"],
        "mrr": within["mrr"],
        "reverse_recall_at_1":
            reverse["recall_at_1"],
        "reverse_mrr": reverse["mrr"],
        "median_positive_cosine":
            within["median_positive_cosine"],
        "at2_offdiagonal_cosine":
            node_diversity(at2),
        "gt450_offdiagonal_cosine":
            node_diversity(gt450),
        "gate": float(model.gate().item()),
    }

    if region_names is None:
        return result, []

    if len(region_names) != len(data):
        raise RuntimeError(
            "Region names do not match evaluation."
        )

    flattened_at2 = at2.reshape(-1, DIM)
    flattened_gt450 = gt450.reshape(-1, DIM)

    global_forward = retrieval_scores(
        flattened_at2, flattened_gt450
    )

    global_reverse = retrieval_scores(
        flattened_gt450, flattened_at2
    )

    result.update({
        "global_recall_at_1":
            global_forward["recall_at_1"],
        "global_recall_at_5":
            global_forward["recall_at_5"],
        "global_mrr":
            global_forward["mrr"],
        "reverse_global_recall_at_1":
            global_reverse["recall_at_1"],
        "reverse_global_mrr":
            global_reverse["mrr"],
    })

    region_rows = []

    for idx, name in enumerate(region_names):
        forward = retrieval_scores(
            at2[idx], gt450[idx]
        )

        backward = retrieval_scores(
            gt450[idx], at2[idx]
        )

        region_rows.append({
            "region_id": name,
            "recall_at_1": forward["recall_at_1"],
            "mrr": forward["mrr"],
            "reverse_recall_at_1":
                backward["recall_at_1"],
            "median_positive_cosine":
                forward["median_positive_cosine"],
            "at2_offdiagonal_cosine":
                node_diversity(at2[idx:idx + 1]),
            "gt450_offdiagonal_cosine":
                node_diversity(gt450[idx:idx + 1]),
        })

    return result, region_rows


def is_better(current, best):
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
print("PHASE 4.2.11 - GATED RESIDUAL GRAPHSAGE", flush=True)
print("=" * 70, flush=True)


# --------------------------------------------------
# 1. Verify provenance
# --------------------------------------------------

print("\n[1] Verifying frozen experimental inputs...", flush=True)

required = [
    GRAPH,
    GRAPH_REPORT,
    PROTOCOL,
    ARCH_CONFIG,
    ARCH_REPORT,
    *REFERENCE_REPORTS.values(),
]

for path in required:
    if not path.is_file():
        raise RuntimeError(
            f"Missing required source: {path}"
        )

for path in (HISTORY, SUMMARY, REGION_TABLE, REPORT):
    if path.exists():
        raise RuntimeError(
            f"Refusing to overwrite: {path}"
        )

graph_hash = file_hash(GRAPH)
protocol_hash = file_hash(PROTOCOL)
arch_hash = file_hash(ARCH_CONFIG)

if load_json(GRAPH_REPORT)["graph_archive_sha256"] != graph_hash:
    raise RuntimeError("Graph provenance mismatch.")

protocol = load_json(PROTOCOL)
architecture = load_json(ARCH_CONFIG)
arch_report = load_json(ARCH_REPORT)

if protocol["source_graph_sha256"] != graph_hash:
    raise RuntimeError("Protocol provenance mismatch.")

if (
    architecture["graph_archive_sha256"] != graph_hash
    or architecture["initial_protocol_sha256"] != protocol_hash
):
    raise RuntimeError("Architecture provenance mismatch.")

if arch_report["architecture_sha256"] != arch_hash:
    raise RuntimeError("Architecture SHA256 mismatch.")

model_spec = architecture["model"]
training_spec = architecture["initial_training"]

if model_spec["working_name"] != "GatedResidualGraphSAGE":
    raise RuntimeError("Unexpected architecture.")

if model_spec["dimensions"] != [384, 256, 384]:
    raise RuntimeError("Model dimension mismatch.")

if model_spec["gate_initial_value"] != GATE_INITIAL:
    raise RuntimeError("Initial gate mismatch.")

if tuple(training_spec["seeds"]) != SEEDS:
    raise RuntimeError("Seed protocol mismatch.")

expected_training = {
    "temperature": TEMPERATURE,
    "learning_rate": LR,
    "weight_decay": WEIGHT_DECAY,
    "batch_regions": BATCH_REGIONS,
    "maximum_epochs": MAX_EPOCHS,
    "early_stopping_patience": PATIENCE,
}

for key, expected in expected_training.items():
    if training_spec[key] != expected:
        raise RuntimeError(
            f"Training configuration mismatch: {key}"
        )

if training_spec["test_used_for_selection"]:
    raise RuntimeError("Test selection is prohibited.")

if training_spec["diversity_regularization_initial_weight"] != 0.0:
    raise RuntimeError(
        "First model must omit diversity regularization."
    )

references = {
    name: load_json(path)
    for name, path in REFERENCE_REPORTS.items()
}

for name, result in references.items():
    if (
        result["graph_archive_sha256"] != graph_hash
        or result["protocol_sha256"] != protocol_hash
    ):
        raise RuntimeError(
            f"Historical provenance mismatch: {name}"
        )

CHECKPOINT_DIR.mkdir(
    parents=True, exist_ok=True
)

checkpoint_paths = {
    seed: CHECKPOINT_DIR / f"gated_residual_seed_{seed}.pt"
    for seed in SEEDS
}

if any(
    path.exists()
    for path in checkpoint_paths.values()
):
    raise RuntimeError(
        "Existing experiment checkpoint detected."
    )

with np.load(GRAPH, allow_pickle=False) as archive:
    features = archive["node_features"].copy()
    edges = archive["edge_index"].copy()
    regions = archive["region_id"].astype(str)
    splits = archive["spatial_split"].astype(str)
    scanners = archive["scanner"].astype(str)
    tiles = archive["tile_id"].astype(str)

if features.shape != (120, 36, DIM):
    raise RuntimeError("Invalid graph features.")

if (
    features.dtype != np.float32
    or not np.isfinite(features).all()
):
    raise RuntimeError("Invalid feature values.")

neighbor_weights = make_neighbor_matrix(
    edges
).to(DEVICE)

print("Graph tensor:", features.shape)
print("Device:", DEVICE)
print("Initial gate:", GATE_INITIAL)

if DEVICE.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
        flush=True
    )


# --------------------------------------------------
# 2. Assemble train and validation pairs
# --------------------------------------------------

print("\n[2] Preparing spatial partitions...", flush=True)

by_region = defaultdict(dict)
region_splits = {}

for index, name in enumerate(regions):
    scanner = scanners[index]
    split = splits[index]

    if scanner not in ("AT2", "GT450"):
        raise RuntimeError("Unexpected scanner.")

    if split not in ("train", "validation", "test"):
        raise RuntimeError("Unexpected split.")

    if scanner in by_region[name]:
        raise RuntimeError(
            "Duplicate scanner/region graph."
        )

    by_region[name][scanner] = index

    if (
        name in region_splits
        and region_splits[name] != split
    ):
        raise RuntimeError(
            "Region appears across partitions."
        )

    region_splits[name] = split

if dict(Counter(region_splits.values())) != {
    "train": 40,
    "validation": 10,
    "test": 10,
}:
    raise RuntimeError(
        "Spatial partition counts incorrect."
    )

paired = {
    "train": [],
    "validation": [],
}

region_names = {
    "train": [],
    "validation": [],
}

for name in sorted(by_region):
    pair = by_region[name]

    if set(pair) != {"AT2", "GT450"}:
        raise RuntimeError("Missing scanner pair.")

    ai = pair["AT2"]
    bi = pair["GT450"]

    if not np.array_equal(
        tiles[ai], tiles[bi]
    ):
        raise RuntimeError(
            "Scanner node alignment mismatch."
        )

    split = region_splits[name]

    if split == "test":
        continue

    paired[split].append(
        np.stack(
            (features[ai], features[bi]),
            axis=0
        )
    )

    region_names[split].append(name)

train = np.stack(
    paired["train"]
).astype(np.float32)

validation = np.stack(
    paired["validation"]
).astype(np.float32)

if train.shape != (40, 2, 36, DIM):
    raise RuntimeError(
        "Training tensor mismatch."
    )

if validation.shape != (10, 2, 36, DIM):
    raise RuntimeError(
        "Validation tensor mismatch."
    )

print("Training pairs:", len(train))
print("Validation pairs:", len(validation))
print("Test pairs: 10 (reserved)")


# --------------------------------------------------
# 3. Train independent seeds
# --------------------------------------------------

print("\n[3] Training gated residual models...", flush=True)

history = []
summaries = []
region_results = []
checkpoint_hashes = {}

for seed in SEEDS:
    print(f"\nSeed: {seed}", flush=True)

    set_seed(seed)

    model = GatedResidualGraphSAGE(
        neighbor_weights
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    rng = np.random.default_rng(seed)

    best = None
    best_weights = None
    best_epoch = None
    stale = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        order = rng.permutation(
            len(train)
        )

        total_loss = 0.0
        observed = 0

        for start in range(
            0,
            len(order),
            BATCH_REGIONS
        ):
            indices = order[
                start:start + BATCH_REGIONS
            ]

            batch = torch.as_tensor(
                train[indices],
                dtype=torch.float32,
                device=DEVICE
            )

            a = model(batch[:, 0])
            b = model(batch[:, 1])

            loss = contrastive_loss(a, b)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite training loss."
                )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()
            optimizer.step()

            total_loss += (
                float(loss.item()) * len(indices)
            )

            observed += len(indices)

        train_loss = total_loss / observed

        metrics, _ = evaluate(
            model, validation
        )

        if not all(
            math.isfinite(float(value))
            for value in metrics.values()
        ):
            raise RuntimeError(
                "Non-finite validation metrics."
            )

        improved = is_better(
            metrics, best
        )

        if improved:
            best = dict(metrics)
            best_epoch = epoch

            best_weights = {
                key: value.detach().cpu().clone()
                for key, value in
                model.state_dict().items()
            }

            stale = 0
        else:
            stale += 1

        history.append({
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": metrics["loss"],
            "validation_recall_at_1":
                metrics["recall_at_1"],
            "validation_mrr":
                metrics["mrr"],
            "validation_reverse_recall_at_1":
                metrics["reverse_recall_at_1"],
            "at2_offdiagonal_cosine":
                metrics["at2_offdiagonal_cosine"],
            "gt450_offdiagonal_cosine":
                metrics["gt450_offdiagonal_cosine"],
            "learned_gate":
                metrics["gate"],
            "best_checkpoint": improved,
        })

        if (
            epoch == 1
            or epoch % 10 == 0
        ):
            print(
                f"  Epoch {epoch:03d} | "
                f"Train loss {train_loss:.4f} | "
                f"Val R@1 {metrics['recall_at_1']:.4f} | "
                f"MRR {metrics['mrr']:.4f} | "
                f"Gate {metrics['gate']:.4f}",
                flush=True
            )

        if stale >= PATIENCE:
            print(
                f"  Early stopping: epoch {epoch}",
                flush=True
            )
            break

    if best_weights is None:
        raise RuntimeError(
            "No selected checkpoint."
        )

    model.load_state_dict(
        best_weights, strict=True
    )

    final, per_region = evaluate(
        model,
        validation,
        region_names["validation"]
    )

    if abs(
        final["recall_at_1"]
        - best["recall_at_1"]
    ) > 1e-7:
        raise RuntimeError(
            "Restored best checkpoint mismatch."
        )

    checkpoint = checkpoint_paths[seed]

    with checkpoint.open("xb") as stream:
        torch.save({
            "model_state_dict": best_weights,
            "model_name":
                "GatedResidualGraphSAGE",
            "architecture_version":
                "candidate_v1",
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_metrics": final,
            "graph_archive_sha256": graph_hash,
            "protocol_sha256": protocol_hash,
            "architecture_sha256": arch_hash,
            "test_evaluated": False,
        }, stream)

    checkpoint_hashes[str(seed)] = file_hash(
        checkpoint
    )

    summaries.append({
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_loss": final["loss"],
        "validation_recall_at_1":
            final["recall_at_1"],
        "validation_mrr": final["mrr"],
        "reverse_recall_at_1":
            final["reverse_recall_at_1"],
        "global_recall_at_1":
            final["global_recall_at_1"],
        "global_recall_at_5":
            final["global_recall_at_5"],
        "global_mrr":
            final["global_mrr"],
        "reverse_global_recall_at_1":
            final["reverse_global_recall_at_1"],
        "median_positive_cosine":
            final["median_positive_cosine"],
        "at2_offdiagonal_cosine":
            final["at2_offdiagonal_cosine"],
        "gt450_offdiagonal_cosine":
            final["gt450_offdiagonal_cosine"],
        "learned_gate": final["gate"],
        "checkpoint_sha256":
            checkpoint_hashes[str(seed)],
    })

    for item in per_region:
        region_results.append({
            "seed": seed,
            **item,
        })

    print(
        f"  Best epoch: {best_epoch} | "
        f"Val R@1 {final['recall_at_1']:.4f} | "
        f"Global R@1 {final['global_recall_at_1']:.4f} | "
        f"Gate {final['gate']:.4f}",
        flush=True
    )


# --------------------------------------------------
# 4. Save new research artifacts
# --------------------------------------------------

print("\n[4] Saving model results...", flush=True)

save_csv_new(
    HISTORY,
    history,
    [
        "seed", "epoch", "train_loss",
        "validation_loss",
        "validation_recall_at_1",
        "validation_mrr",
        "validation_reverse_recall_at_1",
        "at2_offdiagonal_cosine",
        "gt450_offdiagonal_cosine",
        "learned_gate",
        "best_checkpoint",
    ],
)

save_csv_new(
    SUMMARY,
    summaries,
    [
        "seed", "best_epoch",
        "validation_loss",
        "validation_recall_at_1",
        "validation_mrr",
        "reverse_recall_at_1",
        "global_recall_at_1",
        "global_recall_at_5",
        "global_mrr",
        "reverse_global_recall_at_1",
        "median_positive_cosine",
        "at2_offdiagonal_cosine",
        "gt450_offdiagonal_cosine",
        "learned_gate",
        "checkpoint_sha256",
    ],
)

save_csv_new(
    REGION_TABLE,
    region_results,
    [
        "seed", "region_id",
        "recall_at_1",
        "mrr",
        "reverse_recall_at_1",
        "median_positive_cosine",
        "at2_offdiagonal_cosine",
        "gt450_offdiagonal_cosine",
    ],
)

scores = np.array(
    [row["validation_recall_at_1"]
     for row in summaries],
    dtype=np.float64
)

global_scores = np.array(
    [row["global_recall_at_1"]
     for row in summaries],
    dtype=np.float64
)

gates = np.array(
    [row["learned_gate"]
     for row in summaries],
    dtype=np.float64
)

baseline_scores = {
    "MLP":
        references["MLP"]["mean_validation_recall_at_1"],
    "GCN":
        references["GCN"]["mean_validation_recall_at_1"],
    "GraphSAGE":
        references["GraphSAGE"]["mean_validation_recall_at_1"],
    "FixedRandom":
        references["FixedRandom"]["mean_permuted_recall_at_1"],
    "SelfOnly":
        references["SelfOnly"]["mean_self_only_recall_at_1"],
    "RegionRandom":
        references["RegionRandom"]["mean_region_specific_recall_at_1"],
}

report = {
    "phase": "4.2",
    "step": "4.2.11",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "experiment":
        "gated_residual_graphsage_v1",
    "architecture":
        "384_256_384_residual_gated_graphsage",
    "input": "frozen_raw_dinov2_cls",
    "backbone": "mean_aggregation_graphsage",
    "topology": "original_spatial_four_neighbor",
    "residual_identity": True,
    "gate": {
        "parameterization": "sigmoid_scalar",
        "initial_value": GATE_INITIAL,
        "mean_best_checkpoint_value":
            float(gates.mean()),
        "seed_values": gates.tolist(),
    },
    "objective":
        "bidirectional_within_region_InfoNCE",
    "diversity_regularization_weight": 0.0,
    "temperature": TEMPERATURE,
    "learning_rate": LR,
    "weight_decay": WEIGHT_DECAY,
    "batch_regions": BATCH_REGIONS,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
    "seeds": list(SEEDS),
    "graph_archive_sha256": graph_hash,
    "protocol_sha256": protocol_hash,
    "architecture_sha256": arch_hash,
    "training_region_pairs": 40,
    "validation_region_pairs": 10,
    "test_region_pairs_reserved": 10,
    "test_evaluated": False,
    "primary_checkpoint_selection":
        "validation_within_region_recall_at_1_then_loss",
    "mean_validation_recall_at_1":
        float(scores.mean()),
    "std_validation_recall_at_1":
        float(scores.std(ddof=1)),
    "mean_validation_global_recall_at_1":
        float(global_scores.mean()),
    "global_retrieval_status":
        "exploratory_not_checkpoint_selection",
    "baseline_validation_recall_at_1":
        baseline_scores,
    "difference_vs_original_graphsage_pp":
        float(
            (scores.mean() - baseline_scores["GraphSAGE"]) * 100
        ),
    "difference_vs_region_random_pp":
        float(
            (scores.mean() - baseline_scores["RegionRandom"]) * 100
        ),
    "seed_results": summaries,
    "checkpoint_hashes": checkpoint_hashes,
    "history_sha256": file_hash(HISTORY),
    "summary_sha256": file_hash(SUMMARY),
    "region_metrics_sha256": file_hash(REGION_TABLE),
    "biological_independence_verified": False,
    "pixel_perfect_registration_assumed": False,
    "xai_planned": True,
    "limitations": [
        "Single biological source context",
        "Only 10 spatial validation regions",
        "High baseline retrieval performance",
        "Only two scanners in Stage A",
        "Scalar gate is shared across all nodes and regions",
        "No explicit stability or diversity loss yet",
        "Validation remains exploratory",
        "No held-out test evaluation",
    ],
}

save_json_new(REPORT, report)

print("\n" + "=" * 70)
print("PHASE 4.2.11 - GATED RESIDUAL TRAINING COMPLETED")
print("=" * 70)

for row in summaries:
    print(
        f"Seed {row['seed']} | "
        f"Best epoch {row['best_epoch']} | "
        f"Val R@1 {row['validation_recall_at_1']:.4f} | "
        f"Global R@1 {row['global_recall_at_1']:.4f} | "
        f"Gate {row['learned_gate']:.4f}"
    )

print(
    "\nMean validation R@1:",
    f"{scores.mean():.4f}"
)

print(
    "Mean global validation R@1:",
    f"{global_scores.mean():.4f}"
)

print(
    "Original GraphSAGE R@1:",
    f"{baseline_scores['GraphSAGE']:.4f}"
)

print(
    "Region-random GraphSAGE R@1:",
    f"{baseline_scores['RegionRandom']:.4f}"
)

print(
    "Difference vs original GraphSAGE:",
    f"{report['difference_vs_original_graphsage_pp']:+.2f} pp"
)

print(
    "Mean learned gate:",
    f"{gates.mean():.4f}"
)

print("\nSaved:")
print(HISTORY)
print(SUMMARY)
print(REGION_TABLE)
print(REPORT)

print("\nCheckpoints:")
for path in checkpoint_paths.values():
    print(path)

print("\nTest partition: NOT EVALUATED")
print("Historical experiments: UNCHANGED")
print("Explicit stability loss: NOT YET ADDED")
print("XAI: PLANNED")
print("\nSTEP 4.2.11 COMPLETED")

