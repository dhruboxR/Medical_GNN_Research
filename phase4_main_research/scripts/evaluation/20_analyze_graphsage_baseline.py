
"""
PHASE 4.2.5 - GRAPHSAGE DIAGNOSTICS

Compare on VALIDATION ONLY:
  1. Frozen DINOv2 embeddings
  2. Trained GraphSAGE with original neighbors
  3. Same GraphSAGE checkpoint with neighbor
     messages disabled during inference

The third condition is an inference-time ablation,
not a separately trained no-graph baseline.

Audit:
  - Recall@1 and MRR
  - Node off-diagonal cosine
  - Per-region retrieval performance
  - Dependence on neighborhood aggregation
  - Saved checkpoint/source provenance

No training.
No test evaluation.
No existing artifacts modified.
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


ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "phase4_main_research"

GRAPH = BASE / "outputs/features/15_stage_a_spatial_graphs.npz"
GRAPH_REPORT = BASE / "outputs/reports/15_graph_dataset_report.json"
PROTOCOL = BASE / "configs/16_initial_gnn_experimental_protocol.json"

REPORTS = {
    "MLP": BASE / "outputs/reports/17_mlp_baseline_report.json",
    "GCN": BASE / "outputs/reports/18_gcn_baseline_report.json",
    "GraphSAGE": BASE / "outputs/reports/19_graphsage_baseline_report.json",
}

CHECKPOINTS = BASE / "outputs/checkpoints/19_graphsage_baseline"

PER_REGION = BASE / "outputs/tables/20_graphsage_region_diagnostics.csv"
COMPARISON = BASE / "outputs/tables/20_baseline_comparison.csv"
REPORT = BASE / "outputs/reports/20_graphsage_diagnostics.json"

SEEDS = (42, 123, 2026)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def save_csv(path, records, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


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
            + self.neighbor_linear(torch.matmul(weights, x))
        )


class SpatialGraphSAGE(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer("neighbor_weights", weights.clone())
        self.conv1 = SAGEConv(384, 256)
        self.conv2 = SAGEConv(256, 384)

    def forward(self, x, disable_neighbors=False):
        weights = (
            torch.zeros_like(self.neighbor_weights)
            if disable_neighbors
            else self.neighbor_weights
        )
        hidden = F.gelu(self.conv1(x, weights))
        output = self.conv2(hidden, weights)
        return F.normalize(output, dim=-1)


def neighbor_matrix(edges):
    if edges.shape != (2, 120):
        raise RuntimeError("Unexpected edge shape.")

    matrix = torch.zeros(36, 36)
    seen = set()

    for source, target in edges.T.tolist():
        if (
            source == target
            or not 0 <= source < 36
            or not 0 <= target < 36
        ):
            raise RuntimeError("Invalid graph edge.")

        if (source, target) in seen:
            raise RuntimeError("Duplicate edge.")
        seen.add((source, target))
        matrix[target, source] = 1.0

    if len(seen) != 120 or any(
        (v, u) not in seen for u, v in seen
    ):
        raise RuntimeError("Invalid graph topology.")

    degree = matrix.sum(-1, keepdim=True)
    if (degree == 0).any():
        raise RuntimeError("Isolated node.")

    return (matrix / degree).to(DEVICE)


def metrics(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)

    scores = a @ b.transpose(-1, -2)
    n = scores.shape[-1]
    target = torch.arange(n, device=scores.device)

    positive = scores.diagonal(dim1=-2, dim2=-1)
    rank = 1 + (scores > positive.unsqueeze(-1)).sum(-1)

    off_diagonal_mask = ~torch.eye(
        n, dtype=torch.bool, device=scores.device
    )

    node_a = a @ a.transpose(-1, -2)
    node_b = b @ b.transpose(-1, -2)

    return {
        "recall_at_1": float(
            (scores.argmax(-1) == target).float().mean().item()
        ),
        "mrr": float((1.0 / rank.float()).mean().item()),
        "median_positive_cosine": float(positive.median().item()),
        "at2_offdiagonal_cosine": float(
            node_a[off_diagonal_mask].mean().item()
        ),
        "gt450_offdiagonal_cosine": float(
            node_b[off_diagonal_mask].mean().item()
        ),
    }


print("=" * 70)
print("PHASE 4.2.5 - GRAPHSAGE DIAGNOSTICS")
print("=" * 70)

print("\n[1] Verifying experiment provenance...")

for path in (GRAPH, GRAPH_REPORT, PROTOCOL, *REPORTS.values()):
    if not path.is_file():
        raise RuntimeError(f"Missing source: {path}")

for path in (PER_REGION, COMPARISON, REPORT):
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite: {path}")

graph_hash = sha256(GRAPH)
protocol_hash = sha256(PROTOCOL)

if load_json(GRAPH_REPORT)["graph_archive_sha256"] != graph_hash:
    raise RuntimeError("Graph hash mismatch.")

if load_json(PROTOCOL)["source_graph_sha256"] != graph_hash:
    raise RuntimeError("Protocol graph mismatch.")

references = {name: load_json(path) for name, path in REPORTS.items()}

for name, result in references.items():
    if (
        result["graph_archive_sha256"] != graph_hash
        or result["protocol_sha256"] != protocol_hash
    ):
        raise RuntimeError(f"Provenance mismatch: {name}")

with np.load(GRAPH, allow_pickle=False) as archive:
    features = archive["node_features"].copy()
    regions = archive["region_id"].astype(str)
    splits = archive["spatial_split"].astype(str)
    scanners = archive["scanner"].astype(str)
    tiles = archive["tile_id"].astype(str)
    edges = archive["edge_index"].copy()

if features.shape != (120, 36, 384):
    raise RuntimeError("Unexpected graph tensor.")

if not np.isfinite(features).all():
    raise RuntimeError("Invalid input features.")

# Materialize only validation graph pairs.
by_region = defaultdict(dict)

for i, region in enumerate(regions):
    if splits[i] != "validation":
        continue

    if scanners[i] in by_region[region]:
        raise RuntimeError("Duplicate validation graph.")

    by_region[region][scanners[i]] = i

if len(by_region) != 10:
    raise RuntimeError("Expected 10 validation regions.")

pairs = {}

for region, pair in sorted(by_region.items()):
    if set(pair) != {"AT2", "GT450"}:
        raise RuntimeError("Incomplete scanner pair.")

    ai, bi = pair["AT2"], pair["GT450"]

    if not np.array_equal(tiles[ai], tiles[bi]):
        raise RuntimeError("Node order mismatch.")

    pairs[region] = (
        torch.from_numpy(features[ai]).to(DEVICE),
        torch.from_numpy(features[bi]).to(DEVICE),
    )

weights = neighbor_matrix(edges)

print("Validation region pairs:", len(pairs))
print("Seeds:", SEEDS)
print("Test region pairs: NOT LOADED INTO EVALUATION")


print("\n[2] Measuring frozen encoder retrieval...")

rows = []

with torch.inference_mode():
    for region, (a, b) in pairs.items():
        result = metrics(a, b)
        rows.append({
            "model": "Frozen DINOv2",
            "seed": "none",
            "region_id": region,
            **result,
        })


print("\n[3] Running GraphSAGE neighbor ablation...")

for seed in SEEDS:
    checkpoint = CHECKPOINTS / f"graphsage_seed_{seed}.pt"

    if not checkpoint.is_file():
        raise RuntimeError(f"Missing checkpoint: {checkpoint}")

    expected_hash = references["GraphSAGE"][
        "checkpoint_hashes"
    ][str(seed)]

    if sha256(checkpoint) != expected_hash:
        raise RuntimeError("Checkpoint SHA256 mismatch.")

    # Load only checkpoints produced by our own
    # trusted training script.
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    if (
        payload["seed"] != seed
        or payload["graph_archive_sha256"] != graph_hash
        or payload["protocol_sha256"] != protocol_hash
        or payload["test_evaluated"] is not False
    ):
        raise RuntimeError("Checkpoint provenance mismatch.")

    model = SpatialGraphSAGE(weights).to(DEVICE)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    with torch.inference_mode():
        for region, (a, b) in pairs.items():
            for name, disable in (
                ("GraphSAGE original", False),
                ("GraphSAGE neighbors disabled", True),
            ):
                za = model(a, disable_neighbors=disable)
                zb = model(b, disable_neighbors=disable)

                rows.append({
                    "model": name,
                    "seed": seed,
                    "region_id": region,
                    **metrics(za, zb),
                })

    print(f"  Seed {seed}: completed")


print("\n[4] Aggregating validation statistics...")

comparison = []

for name in (
    "Frozen DINOv2",
    "MLP",
    "GCN",
    "GraphSAGE original",
    "GraphSAGE neighbors disabled",
):
    if name == "MLP" or name == "GCN":
        source = references[name]
        scores = [
            float(r["validation_recall_at_1"])
            for r in source["seed_results"]
        ]
        mrr = [
            float(r["validation_mrr"])
            for r in source["seed_results"]
        ]
        off_a = [
            float(r["validation_at2_offdiagonal_cosine"])
            for r in source["seed_results"]
        ] if name == "GCN" else []
        off_b = [
            float(r["validation_gt450_offdiagonal_cosine"])
            for r in source["seed_results"]
        ] if name == "GCN" else []
    else:
        grouped = defaultdict(list)
        for row in rows:
            if row["model"] == name:
                grouped[str(row["seed"])].append(row)

        scores = [
            float(np.mean([r["recall_at_1"] for r in group]))
            for group in grouped.values()
        ]
        mrr = [
            float(np.mean([r["mrr"] for r in group]))
            for group in grouped.values()
        ]
        off_a = [
            float(np.mean([
                r["at2_offdiagonal_cosine"] for r in group
            ]))
            for group in grouped.values()
        ]
        off_b = [
            float(np.mean([
                r["gt450_offdiagonal_cosine"] for r in group
            ]))
            for group in grouped.values()
        ]

    comparison.append({
        "model": name,
        "runs": len(scores),
        "mean_validation_recall_at_1": float(np.mean(scores)),
        "std_validation_recall_at_1": (
            float(np.std(scores, ddof=1))
            if len(scores) > 1 else None
        ),
        "mean_validation_mrr": float(np.mean(mrr)),
        "mean_at2_offdiagonal_cosine": (
            float(np.mean(off_a)) if off_a else None
        ),
        "mean_gt450_offdiagonal_cosine": (
            float(np.mean(off_b)) if off_b else None
        ),
    })

for result in comparison:
    print(
        f"  {result['model']}: "
        f"R@1={result['mean_validation_recall_at_1']:.4f} "
        f"MRR={result['mean_validation_mrr']:.4f}"
    )


print("\n[5] Saving reproducible diagnostic outputs...")

save_csv(
    PER_REGION,
    rows,
    [
        "model", "seed", "region_id",
        "recall_at_1", "mrr", "median_positive_cosine",
        "at2_offdiagonal_cosine", "gt450_offdiagonal_cosine",
    ],
)

save_csv(
    COMPARISON,
    comparison,
    [
        "model", "runs",
        "mean_validation_recall_at_1",
        "std_validation_recall_at_1",
        "mean_validation_mrr",
        "mean_at2_offdiagonal_cosine",
        "mean_gt450_offdiagonal_cosine",
    ],
)

lookup = {
    item["model"]: item
    for item in comparison
}

neighbor_delta = (
    lookup["GraphSAGE original"]["mean_validation_recall_at_1"]
    - lookup["GraphSAGE neighbors disabled"][
        "mean_validation_recall_at_1"
    ]
)

report = {
    "phase": "4.2",
    "step": "4.2.5",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "graph_sha256": graph_hash,
    "protocol_sha256": protocol_hash,
    "validation_regions": len(pairs),
    "seeds": list(SEEDS),
    "models": comparison,
    "graphsage_neighbor_ablation_delta_pp": neighbor_delta * 100,
    "ablation": {
        "method": "zero_neighbor_messages_at_inference",
        "retrained": False,
        "interpretation": (
            "Measures sensitivity to message removal; "
            "does not isolate the causal training benefit of edges."
        ),
    },
    "per_region_metrics_sha256": sha256(PER_REGION),
    "comparison_sha256": sha256(COMPARISON),
    "test_evaluated": False,
    "biological_independence_verified": False,
    "xai_planned": True,
}

REPORT.parent.mkdir(parents=True, exist_ok=True)
with REPORT.open("x", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)

print("\n" + "=" * 70)
print("PHASE 4.2.5 - DIAGNOSTICS COMPLETED")
print("=" * 70)

print(
    "GraphSAGE original - neighbors disabled:",
    f"{neighbor_delta * 100:+.2f} pp"
)

print("\nSaved:")
print(PER_REGION)
print(COMPARISON)
print(REPORT)

print("\nTraining performed: NO")
print("Test partition evaluated: NO")
print("Original models and data: UNCHANGED")
print("XAI: PLANNED")
print("\nSTEP 4.2.5 COMPLETED")
