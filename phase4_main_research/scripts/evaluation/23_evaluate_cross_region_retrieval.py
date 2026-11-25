
"""
PHASE 4.2.8 - CROSS-REGION RETRIEVAL EVALUATION

Validation-only evaluation of:
    Frozen DINOv2
    MLP
    GCN
    Original GraphSAGE
    Permuted GraphSAGE
    Self-only GraphSAGE

Each validation query matches its corresponding
tile among all 360 validation tiles, across 10 regions.

No training, no test evaluation, no source modification.
"""

from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


BASE = (
    Path.home() / "MEDICAL_GNN_RESEARCH"
    / "phase4_main_research"
)

GRAPH = BASE / "outputs/features/15_stage_a_spatial_graphs.npz"
GRAPH_REPORT = BASE / "outputs/reports/15_graph_dataset_report.json"
PROTOCOL = BASE / "configs/16_initial_gnn_experimental_protocol.json"

REPORT_FILES = {
    "MLP": "17_mlp_baseline_report.json",
    "GCN": "18_gcn_baseline_report.json",
    "GraphSAGE": "19_graphsage_baseline_report.json",
    "Permuted GraphSAGE": "21_permuted_topology_report.json",
    "Self-only GraphSAGE": "22_self_only_graphsage_report.json",
}

CHECKPOINT_FOLDERS = {
    "MLP": ("17_mlp_baseline", "mlp_seed_{}.pt"),
    "GCN": ("18_gcn_baseline", "gcn_seed_{}.pt"),
    "GraphSAGE": ("19_graphsage_baseline", "graphsage_seed_{}.pt"),
    "Permuted GraphSAGE": (
        "21_permuted_topology", "permuted_seed_{}.pt"
    ),
    "Self-only GraphSAGE": (
        "22_self_only_graphsage", "self_only_seed_{}.pt"
    ),
}

TABLE = BASE / "outputs/tables/23_cross_region_retrieval.csv"
REGIONS_TABLE = BASE / "outputs/tables/23_retrieval_by_region.csv"
REPORT = BASE / "outputs/reports/23_cross_region_retrieval_report.json"

SEEDS = (42, 123, 2026)
DEVICE = torch.device("cpu")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, adjacency):
        return torch.matmul(adjacency, self.linear(x))


class SpatialGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "adjacency", torch.zeros(36, 36)
        )
        self.layer1 = GCNLayer(384, 256)
        self.layer2 = GCNLayer(256, 384)

    def forward(self, x):
        h = F.gelu(self.layer1(x, self.adjacency))
        z = self.layer2(h, self.adjacency)
        return F.normalize(z, dim=-1)


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
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "neighbor_weights", torch.zeros(36, 36)
        )
        self.conv1 = SAGEConv(384, 256)
        self.conv2 = SAGEConv(256, 384)

    def forward(self, x):
        h = F.gelu(
            self.conv1(x, self.neighbor_weights)
        )
        z = self.conv2(h, self.neighbor_weights)
        return F.normalize(z, dim=-1)


def retrieval(a, b):
    """
    a and b have matched row ordering.
    Full candidate set is all rows of b.
    """
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)

    sim = a @ b.T
    n = sim.shape[0]

    positives = sim.diagonal()
    ranks = 1 + (
        sim > positives.unsqueeze(1)
    ).sum(dim=1)

    targets = torch.arange(n)

    sorted_candidates = sim.argsort(
        dim=1, descending=True
    )

    return {
        "recall_at_1": float(
            (sim.argmax(dim=1) == targets)
            .float().mean().item()
        ),
        "recall_at_5": float(
            (sorted_candidates[:, :5] == targets[:, None])
            .any(dim=1).float().mean().item()
        ),
        "mrr": float(
            (1.0 / ranks.float()).mean().item()
        ),
        "median_positive_cosine": float(
            positives.median().item()
        ),
    }


def evaluate_model(model, data):
    with torch.inference_mode():
        if model is None:
            embedded = F.normalize(data, dim=-1)
        else:
            model.eval()
            embedded = model(data)

        a = embedded[:, 0].reshape(-1, 384)
        b = embedded[:, 1].reshape(-1, 384)

        full_forward = retrieval(a, b)
        full_reverse = retrieval(b, a)

        region_rows = []
        for index in range(len(data)):
            region_rows.append(
                retrieval(
                    embedded[index, 0],
                    embedded[index, 1],
                )
            )

    within_recall = float(np.mean([
        item["recall_at_1"]
        for item in region_rows
    ]))

    return full_forward, full_reverse, within_recall, region_rows


print("=" * 70)
print("PHASE 4.2.8 - CROSS-REGION RETRIEVAL")
print("=" * 70)

print("\n[1] Validating inputs...")

for path in (GRAPH, GRAPH_REPORT, PROTOCOL):
    if not path.exists():
        raise RuntimeError(f"Missing input: {path}")

for path in (TABLE, REGIONS_TABLE, REPORT):
    if path.exists():
        raise RuntimeError(f"Refusing overwrite: {path}")

graph_hash = sha256(GRAPH)
protocol_hash = sha256(PROTOCOL)

if read_json(GRAPH_REPORT)["graph_archive_sha256"] != graph_hash:
    raise RuntimeError("Graph provenance mismatch.")

if read_json(PROTOCOL)["source_graph_sha256"] != graph_hash:
    raise RuntimeError("Protocol mismatch.")

references = {}

for name, filename in REPORT_FILES.items():
    path = BASE / "outputs/reports" / filename
    reference = read_json(path)

    if (
        reference["graph_archive_sha256"] != graph_hash
        or reference["protocol_sha256"] != protocol_hash
    ):
        raise RuntimeError(f"Reference mismatch: {name}")

    references[name] = reference


print("\n[2] Loading validation graph pairs...")

with np.load(GRAPH, allow_pickle=False) as z:
    features = z["node_features"].copy()
    regions = z["region_id"].astype(str)
    splits = z["spatial_split"].astype(str)
    scanners = z["scanner"].astype(str)
    tiles = z["tile_id"].astype(str)

if features.shape != (120, 36, 384):
    raise RuntimeError("Graph dimensions invalid.")

if not np.isfinite(features).all():
    raise RuntimeError("Non-finite features.")

by_region = defaultdict(dict)

for index, region in enumerate(regions):
    if splits[index] == "validation":
        if scanners[index] in by_region[region]:
            raise RuntimeError("Duplicate scanner graph.")
        by_region[region][scanners[index]] = index

if len(by_region) != 10:
    raise RuntimeError("Expected 10 validation regions.")

validation_ids = sorted(by_region)
pairs = []

for region in validation_ids:
    indices = by_region[region]

    if set(indices) != {"AT2", "GT450"}:
        raise RuntimeError("Incomplete validation pair.")

    ai, bi = indices["AT2"], indices["GT450"]

    if not np.array_equal(tiles[ai], tiles[bi]):
        raise RuntimeError("Node correspondence mismatch.")

    pairs.append(
        np.stack([features[ai], features[bi]], axis=0)
    )

data = torch.tensor(
    np.stack(pairs),
    dtype=torch.float32,
    device=DEVICE,
)

assert data.shape == (10, 2, 36, 384)

print("Validation regions:", len(validation_ids))
print("Candidates per query: 360")
print("Test regions: NOT USED")

results = []
per_region = []


def record(name, seed, model):
    forward, reverse, within, regions_metrics = (
        evaluate_model(model, data)
    )

    results.append({
        "model": name,
        "seed": str(seed),
        "within_region_recall_at_1": within,
        "global_recall_at_1": forward["recall_at_1"],
        "global_recall_at_5": forward["recall_at_5"],
        "global_mrr": forward["mrr"],
        "reverse_global_recall_at_1":
            reverse["recall_at_1"],
        "reverse_global_mrr": reverse["mrr"],
    })

    for region, metrics in zip(
        validation_ids, regions_metrics
    ):
        per_region.append({
            "model": name,
            "seed": str(seed),
            "region_id": region,
            **metrics,
        })

    print(
        f"  {name}, seed {seed}: "
        f"Within R@1={within:.4f}, "
        f"Global R@1={forward['recall_at_1']:.4f}, "
        f"Global R@5={forward['recall_at_5']:.4f}, "
        f"Global MRR={forward['mrr']:.4f}",
        flush=True,
    )


print("\n[3] Evaluating frozen DINOv2...")

record("Frozen DINOv2", "none", None)


print("\n[4] Evaluating saved models...")

for name in REPORT_FILES:
    reference = references[name]
    folder, pattern = CHECKPOINT_FOLDERS[name]

    for seed in SEEDS:
        checkpoint = (
            BASE / "outputs/checkpoints"
            / folder / pattern.format(seed)
        )

        expected_hash = reference[
            "checkpoint_hashes"
        ][str(seed)]

        if sha256(checkpoint) != expected_hash:
            raise RuntimeError(
                f"Checkpoint SHA256 mismatch: {checkpoint}"
            )

        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        if (
            int(payload["seed"]) != seed
            or payload["graph_archive_sha256"] != graph_hash
            or payload["protocol_sha256"] != protocol_hash
        ):
            raise RuntimeError("Checkpoint provenance mismatch.")

        if name == "MLP":
            model = SharedMLP()
        elif name == "GCN":
            model = SpatialGCN()
        else:
            model = SpatialGraphSAGE()

        model.load_state_dict(
            payload["model_state_dict"],
            strict=True,
        )

        model.to(DEVICE)
        record(name, seed, model)


print("\n[5] Saving validation-only retrieval results...")

write_csv(
    TABLE,
    results,
    [
        "model", "seed", "within_region_recall_at_1",
        "global_recall_at_1", "global_recall_at_5",
        "global_mrr", "reverse_global_recall_at_1",
        "reverse_global_mrr",
    ],
)

write_csv(
    REGIONS_TABLE,
    per_region,
    [
        "model", "seed", "region_id", "recall_at_1",
        "recall_at_5", "mrr",
        "median_positive_cosine",
    ],
)

summary = {}

for row in results:
    summary.setdefault(row["model"], []).append(
        row["global_recall_at_1"]
    )

summary = {
    model: {
        "runs": len(values),
        "mean_global_recall_at_1": float(
            np.mean(values)
        ),
        "std_global_recall_at_1": (
            float(np.std(values, ddof=1))
            if len(values) > 1 else None
        ),
    }
    for model, values in summary.items()
}

report = {
    "phase": "4.2",
    "step": "4.2.8",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "graph_sha256": graph_hash,
    "protocol_sha256": protocol_hash,
    "validation_regions": 10,
    "queries": 360,
    "candidates_per_query": 360,
    "test_evaluated": False,
    "summary": summary,
    "table_sha256": sha256(TABLE),
    "region_table_sha256": sha256(REGIONS_TABLE),
    "limitations": [
        "Validation regions share biological material",
        "Same-source cross-scanner retrieval",
        "No independent patient or slide test",
        "Only 10 validation regions",
        "Models selected using within-region validation Recall@1",
        "Full-candidate evaluation is exploratory and must not be used as an unrecorded new checkpoint-selection criterion",
    ],
}

REPORT.parent.mkdir(parents=True, exist_ok=True)
with REPORT.open("x", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 70)
print("PHASE 4.2.8 - RETRIEVAL EVALUATION COMPLETED")
print("=" * 70)

for name, item in summary.items():
    print(
        f"{name}: mean global R@1 = "
        f"{item['mean_global_recall_at_1']:.4f}"
    )

print("\nSaved:")
print(TABLE)
print(REGIONS_TABLE)
print(REPORT)

print("\nTraining: NO")
print("Test partition: NOT EVALUATED")
print("Existing checkpoints: UNCHANGED")
print("XAI: PLANNED")
print("\nSTEP 4.2.8 COMPLETED")
