
"""
PHASE 4.2.16 - TOPOLOGY-CONSISTENT RESIDUAL GRAPHSAGE

Changes relative to Phase 4.2.13:
    Adds a topology-consistency regularization loss.

Architecture:
    Gated residual GraphSAGE
    384 -> 256 -> 384
    Initial gate = 0.5

Training objective:
    Bidirectional within-region InfoNCE
    + 1.0 * topology consistency

Consistency:
    Same tissue features under original and
    independently randomized adjacency matrices.

Checkpoint selection:
    Original-topology validation Recall@1,
    followed by original-topology validation loss.

Evaluation:
    Original-topology validation retrieval.
    Scanner-discordant random topology robustness,
    using the Phase 4.2.15 randomization protocol.

Test partition stays untouched.
Historical checkpoints are not modified.
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

BASELINE_REPORT = (
    BASE / "outputs/reports/28_gate_init05_report.json"
)
ROBUSTNESS_REPORT = (
    BASE / "outputs/reports/30_topology_robustness_report.json"
)

CHECKPOINT_DIR = (
    BASE / "outputs/checkpoints/31_topology_consistent_residual"
)

HISTORY = (
    BASE / "outputs/tables/31_topology_consistent_history.csv"
)
SUMMARY = (
    BASE / "outputs/tables/31_topology_consistent_summary.csv"
)
ROBUSTNESS_TABLE = (
    BASE / "outputs/tables/31_topology_consistent_robustness.csv"
)
REPORT = (
    BASE / "outputs/reports/31_topology_consistent_report.json"
)

SEEDS = (42, 123, 2026)
INITIAL_GATE = 0.5

LAMBDA_TOPOLOGY = 1.0
TEMPERATURE = 0.07
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_REGIONS = 8
MAX_EPOCHS = 100
PATIENCE = 20
ROBUSTNESS_REPETITIONS = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv_new(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json_new(path, content):
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


def stable_rng(*values):
    key = "|".join(map(str, values)).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    seed = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(seed)


def original_adjacency(edges):
    if edges.shape != (2, 120):
        raise RuntimeError("Unexpected graph edges.")

    a = np.zeros((36, 36), dtype=np.float32)
    seen = set()

    for source, target in edges.T.tolist():
        if (
            source == target
            or not 0 <= source < 36
            or not 0 <= target < 36
            or (source, target) in seen
        ):
            raise RuntimeError("Invalid graph edge.")

        seen.add((source, target))
        a[target, source] = 1.0

    if len(seen) != 120:
        raise RuntimeError("Wrong number of edges.")

    if any((v, u) not in seen for u, v in seen):
        raise RuntimeError("Nonreciprocal graph.")

    if not np.array_equal(a, a.T):
        raise RuntimeError("Asymmetric adjacency.")

    return a


def normalize_adjacency(a):
    degree = a.sum(axis=1, keepdims=True)
    if np.any(degree <= 0):
        raise RuntimeError("Isolated graph node.")
    return (a / degree).astype(np.float32)


def random_adjacency(base, *key):
    """
    Degree-multiset-preserving adjacency permutation.
    Feature order never changes.

    Same randomization method as Phase 4.2.15,
    given the same key components.
    """

    rng = stable_rng("phase4215", *key)

    for _ in range(200):
        p = rng.permutation(36)
        candidate = base[np.ix_(p, p)]

        if np.array_equal(candidate, base):
            continue

        if int(candidate.sum()) != 120:
            raise RuntimeError("Edge count changed.")

        if not np.array_equal(
            np.sort(candidate.sum(axis=1)),
            np.sort(base.sum(axis=1)),
        ):
            raise RuntimeError("Degree multiset changed.")

        if not np.array_equal(candidate, candidate.T):
            raise RuntimeError("Lost adjacency symmetry.")

        return normalize_adjacency(candidate)

    raise RuntimeError("Random adjacency generation failed.")


class SAGEConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.neighbor_linear = nn.Linear(
            input_dim, output_dim, bias=False
        )

    def forward(self, x, adjacency):
        return (
            self.self_linear(x)
            + self.neighbor_linear(
                torch.matmul(adjacency, x)
            )
        )


class GatedResidualGraphSAGE(nn.Module):
    def __init__(self, adjacency):
        super().__init__()

        self.register_buffer(
            "neighbor_weights",
            adjacency.clone()
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

    def forward(self, x, override=None):
        adjacency = (
            self.neighbor_weights
            if override is None else override
        )

        identity = F.normalize(x, dim=-1)

        h = F.gelu(self.conv1(x, adjacency))
        graph = F.normalize(
            self.conv2(h, adjacency),
            dim=-1,
        )

        return F.normalize(
            identity + self.gate() * graph,
            dim=-1,
        )


def contrastive_loss(a, b):
    logits = (a @ b.transpose(-1, -2)) / TEMPERATURE

    batch, nodes, _ = logits.shape
    targets = torch.arange(
        nodes, device=logits.device
    ).expand(batch, nodes).reshape(-1)

    forward = F.cross_entropy(
        logits.reshape(-1, nodes), targets
    )

    reverse = F.cross_entropy(
        logits.transpose(-1, -2).reshape(-1, nodes),
        targets
    )

    return 0.5 * (forward + reverse)


def consistency_loss(original, perturbed):
    """
    Outputs are unit normalized.

    This is a same-node, same-scanner consistency
    objective under changes to adjacency.
    """
    return (
        1.0 - F.cosine_similarity(
            original, perturbed, dim=-1
        )
    ).mean()


def retrieval(a, b):
    similarity = a @ b.transpose(-1, -2)

    batch, nodes, _ = similarity.shape
    targets = torch.arange(
        nodes, device=similarity.device
    ).expand(batch, nodes)

    positives = similarity.diagonal(
        dim1=-2, dim2=-1
    )

    ranks = 1 + (
        similarity > positives.unsqueeze(-1)
    ).sum(-1)

    return {
        "recall_at_1": float(
            (similarity.argmax(-1) == targets)
            .float().mean().item()
        ),
        "mrr": float(
            (1.0 / ranks.float()).mean().item()
        ),
        "matched_cosine": float(
            positives.mean().item()
        ),
    }


@torch.inference_mode()
def evaluate_original(model, validation):
    model.eval()

    batch = torch.as_tensor(
        validation, dtype=torch.float32, device=DEVICE
    )

    a = model(batch[:, 0])
    b = model(batch[:, 1])

    forward = retrieval(a, b)
    reverse = retrieval(b, a)

    global_metrics = retrieval(
        a.reshape(1, -1, 384),
        b.reshape(1, -1, 384),
    )

    return {
        "loss": float(contrastive_loss(a, b).item()),
        "recall_at_1": forward["recall_at_1"],
        "mrr": forward["mrr"],
        "reverse_recall_at_1": reverse["recall_at_1"],
        "global_recall_at_1": global_metrics["recall_at_1"],
        "global_mrr": global_metrics["mrr"],
        "matched_cosine": forward["matched_cosine"],
        "gate": float(model.gate().item()),
    }


def is_better(current, best):
    if best is None:
        return True

    delta = current["recall_at_1"] - best["recall_at_1"]

    return (
        delta > 1e-10
        or (
            abs(delta) <= 1e-10
            and current["loss"] < best["loss"] - 1e-8
        )
    )


@torch.inference_mode()
def evaluate_robustness(
    model, validation, region_names, base, seed
):
    """
    Reproduce the Phase 4.2.15 perturbation
    keys and the same five repetitions.

    Robustness is evaluated after the checkpoint
    has already been selected using normal
    validation retrieval.
    """
    model.eval()
    output_rows = []

    for repetition in range(ROBUSTNESS_REPETITIONS):
        for index, region in enumerate(region_names):
            a = torch.as_tensor(
                validation[index, 0],
                dtype=torch.float32, device=DEVICE
            )
            b = torch.as_tensor(
                validation[index, 1],
                dtype=torch.float32, device=DEVICE
            )

            wa = random_adjacency(
                base, seed, repetition, region, "AT2"
            )
            wb = random_adjacency(
                base, seed, repetition, region, "GT450"
            )

            za = model(
                a, override=torch.from_numpy(wa).to(DEVICE)
            )
            zb = model(
                b, override=torch.from_numpy(wb).to(DEVICE)
            )

            metrics = retrieval(
                za.unsqueeze(0),
                zb.unsqueeze(0),
            )

            output_rows.append({
                "seed": seed,
                "repetition": repetition,
                "region_id": region,
                "recall_at_1": metrics["recall_at_1"],
                "mrr": metrics["mrr"],
                "matched_cosine": metrics["matched_cosine"],
            })

    return output_rows


print("=" * 70, flush=True)
print("PHASE 4.2.16 - TOPOLOGY-CONSISTENT RESIDUAL GNN", flush=True)
print("=" * 70, flush=True)


# ----------------------------------------------------
# 1. Verify prior sources and checkpoint protocol
# ----------------------------------------------------

print("\n[1] Verifying experiment provenance...", flush=True)

required = [
    GRAPH, GRAPH_REPORT, PROTOCOL, ARCH,
    BASELINE_REPORT, ROBUSTNESS_REPORT,
]

for path in required:
    if not path.is_file():
        raise RuntimeError(f"Missing input: {path}")

for path in (HISTORY, SUMMARY, ROBUSTNESS_TABLE, REPORT):
    if path.exists():
        raise RuntimeError(f"Output already exists: {path}")

graph_hash = sha256(GRAPH)
protocol_hash = sha256(PROTOCOL)
arch_hash = sha256(ARCH)

graph_report = read_json(GRAPH_REPORT)
protocol = read_json(PROTOCOL)
architecture = read_json(ARCH)
baseline = read_json(BASELINE_REPORT)
robustness = read_json(ROBUSTNESS_REPORT)

if graph_report["graph_archive_sha256"] != graph_hash:
    raise RuntimeError("Graph hash mismatch.")

if protocol["source_graph_sha256"] != graph_hash:
    raise RuntimeError("Protocol graph mismatch.")

if (
    architecture["graph_archive_sha256"] != graph_hash
    or architecture["initial_protocol_sha256"] != protocol_hash
):
    raise RuntimeError("Architecture mismatch.")

if (
    baseline["graph_archive_sha256"] != graph_hash
    or baseline["protocol_sha256"] != protocol_hash
    or baseline["parent_architecture_sha256"] != arch_hash
):
    raise RuntimeError("Baseline model mismatch.")

if (
    robustness["graph_archive_sha256"] != graph_hash
    or robustness["protocol_sha256"] != protocol_hash
):
    raise RuntimeError("Robustness source mismatch.")

if baseline["test_evaluated"] or robustness["test_evaluated"]:
    raise RuntimeError("Unexpected test evaluation.")

training_spec = architecture["initial_training"]

expected = {
    "temperature": TEMPERATURE,
    "learning_rate": LR,
    "weight_decay": WEIGHT_DECAY,
    "batch_regions": BATCH_REGIONS,
    "maximum_epochs": MAX_EPOCHS,
    "early_stopping_patience": PATIENCE,
}

for key, value in expected.items():
    if training_spec[key] != value:
        raise RuntimeError(f"Protocol mismatch: {key}")

if tuple(training_spec["seeds"]) != SEEDS:
    raise RuntimeError("Seed mismatch.")

if (
    architecture["model"]["dimensions"] != [384, 256, 384]
    or architecture["model"]["gate_type"] != "learned_global_scalar"
):
    raise RuntimeError("Architecture is not compatible.")

if (
    robustness["repetitions"] != ROBUSTNESS_REPETITIONS
    or "scanner_discordant_random"
    not in robustness["conditions"]
):
    raise RuntimeError("Robustness protocol mismatch.")

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

checkpoint_paths = {
    seed: CHECKPOINT_DIR / f"topology_consistent_seed_{seed}.pt"
    for seed in SEEDS
}

if any(path.exists() for path in checkpoint_paths.values()):
    raise RuntimeError("Existing checkpoint detected.")


# ----------------------------------------------------
# 2. Prepare original training and validation data
# ----------------------------------------------------

print("\n[2] Preparing spatial graph pairs...", flush=True)

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
    raise RuntimeError("Invalid feature tensor.")

by_region = defaultdict(dict)
region_split = {}

for i, region in enumerate(regions):
    scanner, split = scanners[i], splits[i]

    if scanner not in ("AT2", "GT450"):
        raise RuntimeError("Unexpected scanner.")

    if split not in ("train", "validation", "test"):
        raise RuntimeError("Unexpected partition.")

    if scanner in by_region[region]:
        raise RuntimeError("Duplicate scanner graph.")

    by_region[region][scanner] = i

    if region in region_split and region_split[region] != split:
        raise RuntimeError("Region split conflict.")

    region_split[region] = split

if dict(Counter(region_split.values())) != {
    "train": 40, "validation": 10, "test": 10
}:
    raise RuntimeError("Incorrect spatial partitions.")

paired = {"train": [], "validation": []}
names = {"train": [], "validation": []}

for region in sorted(by_region):
    indices = by_region[region]

    if set(indices) != {"AT2", "GT450"}:
        raise RuntimeError("Incomplete scanner pair.")

    a, b = indices["AT2"], indices["GT450"]

    if not np.array_equal(tiles[a], tiles[b]):
        raise RuntimeError("Tile correspondence mismatch.")

    split = region_split[region]
    if split == "test":
        continue

    paired[split].append(
        np.stack([features[a], features[b]], axis=0)
    )
    names[split].append(region)

train = np.stack(paired["train"]).astype(np.float32)
validation = np.stack(paired["validation"]).astype(np.float32)

if train.shape != (40, 2, 36, 384):
    raise RuntimeError("Incorrect train shape.")

if validation.shape != (10, 2, 36, 384):
    raise RuntimeError("Incorrect validation shape.")

base = original_adjacency(edges)
weights = torch.from_numpy(
    normalize_adjacency(base)
).to(DEVICE)

print("Training pairs:", len(train))
print("Validation pairs:", len(validation))
print("Test pairs: 10, reserved")
print("Device:", DEVICE)
print("Topology loss weight:", LAMBDA_TOPOLOGY)


# ----------------------------------------------------
# 3. Train each random seed
# ----------------------------------------------------

print("\n[3] Training topology-consistent models...", flush=True)

history = []
summaries = []
robustness_rows = []
checkpoint_hashes = {}

for seed in SEEDS:
    print(f"\nSeed: {seed}", flush=True)
    set_seed(seed)

    model = GatedResidualGraphSAGE(weights).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
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

        total_loss = 0.0
        total_contrastive = 0.0
        total_consistency = 0.0
        seen = 0

        for start in range(0, len(train), BATCH_REGIONS):
            indices = order[start:start + BATCH_REGIONS]

            batch = torch.as_tensor(
                train[indices],
                dtype=torch.float32,
                device=DEVICE,
            )

            # Independent training-only random graphs
            # for each region, scanner, and epoch.
            wa = np.stack([
                random_adjacency(
                    base, "train", seed, epoch,
                    names["train"][i], "AT2"
                )
                for i in indices
            ])

            wb = np.stack([
                random_adjacency(
                    base, "train", seed, epoch,
                    names["train"][i], "GT450"
                )
                for i in indices
            ])

            wa = torch.from_numpy(wa).to(DEVICE)
            wb = torch.from_numpy(wb).to(DEVICE)

            original_a = model(batch[:, 0])
            original_b = model(batch[:, 1])

            perturbed_a = model(
                batch[:, 0], override=wa
            )
            perturbed_b = model(
                batch[:, 1], override=wb
            )

            loss_contrastive = contrastive_loss(
                original_a, original_b
            )

            loss_consistency = 0.5 * (
                consistency_loss(
                    original_a, perturbed_a
                )
                + consistency_loss(
                    original_b, perturbed_b
                )
            )

            loss = (
                loss_contrastive
                + LAMBDA_TOPOLOGY * loss_consistency
            )

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss.")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            count = len(indices)
            total_loss += float(loss.item()) * count
            total_contrastive += (
                float(loss_contrastive.item()) * count
            )
            total_consistency += (
                float(loss_consistency.item()) * count
            )
            seen += count

        normal_metrics = evaluate_original(
            model, validation
        )

        if not all(
            math.isfinite(float(v))
            for v in normal_metrics.values()
        ):
            raise RuntimeError("Invalid validation metrics.")

        improved = is_better(normal_metrics, best)

        if improved:
            best = dict(normal_metrics)
            best_epoch = epoch
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1

        history.append({
            "seed": seed,
            "epoch": epoch,
            "train_total_loss": total_loss / seen,
            "train_contrastive_loss":
                total_contrastive / seen,
            "train_topology_loss":
                total_consistency / seen,
            "validation_loss": normal_metrics["loss"],
            "validation_recall_at_1":
                normal_metrics["recall_at_1"],
            "validation_mrr": normal_metrics["mrr"],
            "learned_gate": normal_metrics["gate"],
            "best_checkpoint": improved,
        })

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  Epoch {epoch:03d} | "
                f"Train total {total_loss / seen:.4f} | "
                f"Topology {total_consistency / seen:.4f} | "
                f"Val R@1 {normal_metrics['recall_at_1']:.4f} | "
                f"Gate {normal_metrics['gate']:.4f}",
                flush=True,
            )

        if stale >= PATIENCE:
            print(
                f"  Early stopping at epoch {epoch}",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError("No checkpoint selected.")

    model.load_state_dict(best_state, strict=True)
    final = evaluate_original(model, validation)

    if abs(
        final["recall_at_1"] - best["recall_at_1"]
    ) > 1e-7:
        raise RuntimeError("Checkpoint restoration mismatch.")

    # Secondary robustness evaluation happens only
    # after selection by the original-topology metric.
    seed_robustness = evaluate_robustness(
        model, validation, names["validation"],
        base, seed,
    )

    robust_recall = float(np.mean([
        r["recall_at_1"] for r in seed_robustness
    ]))

    robustness_rows.extend(seed_robustness)

    checkpoint = checkpoint_paths[seed]

    with checkpoint.open("xb") as stream:
        torch.save({
            "model_state_dict": best_state,
            "model_name": "TopologyConsistentResidualGraphSAGE",
            "seed": seed,
            "best_epoch": best_epoch,
            "initial_gate": INITIAL_GATE,
            "lambda_topology": LAMBDA_TOPOLOGY,
            "validation_metrics": final,
            "scanner_discordant_validation_recall_at_1":
                robust_recall,
            "graph_archive_sha256": graph_hash,
            "protocol_sha256": protocol_hash,
            "parent_architecture_sha256": arch_hash,
            "test_evaluated": False,
        }, stream)

    checkpoint_hashes[str(seed)] = sha256(checkpoint)

    baseline_seed = next(
        r for r in baseline["seed_results"]
        if int(r["seed"]) == seed
    )

    summaries.append({
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_recall_at_1": final["recall_at_1"],
        "validation_mrr": final["mrr"],
        "global_recall_at_1":
            final["global_recall_at_1"],
        "scanner_discordant_recall_at_1": robust_recall,
        "robustness_drop_pp":
            (final["recall_at_1"] - robust_recall) * 100,
        "learned_gate": final["gate"],
        "baseline_gate05_recall_at_1":
            baseline_seed["validation_recall_at_1"],
        "checkpoint_sha256": checkpoint_hashes[str(seed)],
    })

    print(
        f"  Best epoch {best_epoch} | "
        f"Normal R@1 {final['recall_at_1']:.4f} | "
        f"Discordant R@1 {robust_recall:.4f} | "
        f"Gate {final['gate']:.4f}",
        flush=True,
    )


# ----------------------------------------------------
# 4. Compare against historical robustness control
# ----------------------------------------------------

print("\n[4] Comparing stability and retrieval...", flush=True)

original_scores = np.array([
    row["validation_recall_at_1"]
    for row in summaries
], dtype=np.float64)

discordant_scores = np.array([
    row["scanner_discordant_recall_at_1"]
    for row in summaries
], dtype=np.float64)

baseline_normal = np.array([
    row["baseline_gate05_recall_at_1"]
    for row in summaries
], dtype=np.float64)

prior_robustness_rows = [
    row for row in robustness["models"]
    if (
        row["model"] == "ResidualGate05"
        and row["condition"] == "scanner_discordant_random"
    )
]

if len(prior_robustness_rows) != 1:
    raise RuntimeError("Missing historical robustness result.")

prior_robustness = float(
    prior_robustness_rows[0]["mean_recall_at_1"]
)

print(
    "Previous residual normal R@1:",
    f"{baseline_normal.mean():.4f}"
)

print(
    "New topology-consistent normal R@1:",
    f"{original_scores.mean():.4f}"
)

print(
    "Previous scanner-discordant R@1:",
    f"{prior_robustness:.4f}"
)

print(
    "New scanner-discordant R@1:",
    f"{discordant_scores.mean():.4f}"
)


# ----------------------------------------------------
# 5. Save new research results
# ----------------------------------------------------

print("\n[5] Saving experiment artifacts...", flush=True)

write_csv_new(
    HISTORY,
    history,
    [
        "seed", "epoch",
        "train_total_loss",
        "train_contrastive_loss",
        "train_topology_loss",
        "validation_loss",
        "validation_recall_at_1",
        "validation_mrr",
        "learned_gate",
        "best_checkpoint",
    ],
)

write_csv_new(
    SUMMARY,
    summaries,
    [
        "seed", "best_epoch",
        "validation_recall_at_1",
        "validation_mrr",
        "global_recall_at_1",
        "scanner_discordant_recall_at_1",
        "robustness_drop_pp",
        "learned_gate",
        "baseline_gate05_recall_at_1",
        "checkpoint_sha256",
    ],
)

write_csv_new(
    ROBUSTNESS_TABLE,
    robustness_rows,
    [
        "seed", "repetition", "region_id",
        "recall_at_1", "mrr", "matched_cosine",
    ],
)

report = {
    "phase": "4.2",
    "step": "4.2.16",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "experiment": "topology_consistency_regularized_residual",
    "model": "GatedResidualGraphSAGE",
    "initial_gate": INITIAL_GATE,
    "lambda_topology": LAMBDA_TOPOLOGY,
    "topology_training_perturbation":
        "independent_per_region_scanner_epoch",
    "topology_consistency":
        "same_scanner_same_node_original_vs_perturbed",
    "original_topology_contrastive_training": True,
    "seeds": list(SEEDS),
    "training_pairs": 40,
    "validation_pairs": 10,
    "test_pairs_reserved": 10,
    "graph_archive_sha256": graph_hash,
    "protocol_sha256": protocol_hash,
    "parent_architecture_sha256": arch_hash,
    "source_baseline_report_sha256":
        sha256(BASELINE_REPORT),
    "source_robustness_report_sha256":
        sha256(ROBUSTNESS_REPORT),
    "checkpoint_selection":
        "original_topology_validation_recall_at_1_then_loss",
    "robustness_used_for_checkpoint_selection": False,
    "mean_original_recall_at_1":
        float(original_scores.mean()),
    "mean_scanner_discordant_recall_at_1":
        float(discordant_scores.mean()),
    "mean_robustness_drop_pp":
        float(
            (original_scores - discordant_scores).mean() * 100
        ),
    "baseline_normal_recall_at_1":
        float(baseline_normal.mean()),
    "baseline_scanner_discordant_recall_at_1":
        prior_robustness,
    "normal_gain_vs_baseline_pp":
        float(
            (original_scores.mean() - baseline_normal.mean())
            * 100
        ),
    "discordant_gain_vs_baseline_pp":
        float(
            (discordant_scores.mean() - prior_robustness)
            * 100
        ),
    "seed_results": summaries,
    "checkpoint_hashes": checkpoint_hashes,
    "history_sha256": sha256(HISTORY),
    "summary_sha256": sha256(SUMMARY),
    "robustness_table_sha256": sha256(ROBUSTNESS_TABLE),
    "test_evaluated": False,
    "biological_independence_verified": False,
    "xai_planned": True,
    "limitations": [
        "Validation-only exploratory model development",
        "Lambda equals 1.0 and was not independently validated",
        "Synthetic adjacency permutations are not scanner simulations",
        "Repeated perturbations are not independent tissue samples",
        "Checkpoint selection uses original topology Recall@1",
        "No independent patient or slide validation",
        "Single biological source context",
        "Historical baseline may have different stochastic trajectories",
    ],
}

write_json_new(REPORT, report)

print("\n" + "=" * 70)
print("PHASE 4.2.16 - TOPOLOGY-CONSISTENT TRAINING COMPLETED")
print("=" * 70)

for row in summaries:
    print(
        f"Seed {row['seed']} | "
        f"Epoch {row['best_epoch']} | "
        f"Normal R@1 {row['validation_recall_at_1']:.4f} | "
        f"Discordant R@1 "
        f"{row['scanner_discordant_recall_at_1']:.4f} | "
        f"Gate {row['learned_gate']:.4f}"
    )

print(
    "\nMean normal validation R@1:",
    f"{original_scores.mean():.4f}"
)

print(
    "Mean scanner-discordant R@1:",
    f"{discordant_scores.mean():.4f}"
)

print(
    "Normal gain vs prior residual:",
    f"{report['normal_gain_vs_baseline_pp']:+.2f} pp"
)

print(
    "Discordant gain vs prior residual:",
    f"{report['discordant_gain_vs_baseline_pp']:+.2f} pp"
)

print("\nSaved:")
print(HISTORY)
print(SUMMARY)
print(ROBUSTNESS_TABLE)
print(REPORT)

print("\nCheckpoints:")
for path in checkpoint_paths.values():
    print(path)

print("\nHistorical experiments: UNCHANGED")
print("Held-out test: NOT EVALUATED")
print("XAI: PLANNED")
print("\nSTEP 4.2.16 COMPLETED")
