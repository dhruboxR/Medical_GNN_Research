
"""
PHASE 4.2.15 - CROSS-SCANNER TOPOLOGY ROBUSTNESS

Evaluate frozen trained checkpoints:
    Original GraphSAGE
    Gated Residual GraphSAGE, initial gate 0.5

Conditions:
    1. Original spatial topology on both scanners
    2. Same randomized topology on both scanners
    3. Independently randomized topology per scanner

Five deterministic randomization repetitions per
validation region and training seed.

Measures:
    Within-region Recall@1 and MRR
    Hardest-negative cosine margin
    Matched cross-scanner cosine
    Off-diagonal node similarity
    Performance changes relative to original topology

No retraining.
No checkpoint selection.
No held-out test evaluation.
No existing artifact overwritten.
"""

from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import csv
import hashlib
import json

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

MODEL_REPORTS = {
    "GraphSAGE":
        BASE / "outputs/reports/19_graphsage_baseline_report.json",
    "ResidualGate05":
        BASE / "outputs/reports/28_gate_init05_report.json",
}

CHECKPOINTS = {
    "GraphSAGE": (
        "17_unused",
        "unused",
    ),
    "ResidualGate05": (
        "28_residual_gate_init05",
        "gate_init05_seed_{seed}.pt",
    ),
}

CHECKPOINTS["GraphSAGE"] = (
    "19_graphsage_baseline",
    "graphsage_seed_{seed}.pt",
)

TABLE = BASE / "outputs/tables/30_topology_robustness_by_region.csv"
SUMMARY = BASE / "outputs/tables/30_topology_robustness_summary.csv"
REPORT = BASE / "outputs/reports/30_topology_robustness_report.json"

SEEDS = (42, 123, 2026)
REPETITIONS = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def sha256(path):
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
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_json_new(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(content, stream, indent=2)


def make_adjacency(edges):
    if edges.shape != (2, 120):
        raise RuntimeError("Unexpected edge-index shape.")

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
        raise RuntimeError("Expected 120 directed edges.")

    if any((v, u) not in seen for u, v in seen):
        raise RuntimeError("Graph edges are not reciprocal.")

    if not np.array_equal(a, a.T):
        raise RuntimeError("Adjacency is not symmetric.")

    return a


def normalize_adjacency(adjacency):
    degree = adjacency.sum(axis=1, keepdims=True)

    if np.any(degree <= 0):
        raise RuntimeError("Unexpected isolated node.")

    return adjacency / degree


def stable_rng(*parts):
    key = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    seed = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(seed)


def random_adjacency(base, *key):
    """
    Permute adjacency labels, keeping feature order fixed.

    This preserves degree multiset, edges, and graph
    isomorphism, but changes node-specific neighborhoods.
    """

    rng = stable_rng("phase4215", *key)

    for _ in range(200):
        permutation = rng.permutation(36)
        candidate = base[np.ix_(permutation, permutation)]

        if np.array_equal(candidate, base):
            continue

        if int(candidate.sum()) != int(base.sum()):
            raise RuntimeError("Edge count changed.")

        if not np.array_equal(
            np.sort(candidate.sum(axis=1)),
            np.sort(base.sum(axis=1)),
        ):
            raise RuntimeError("Degree distribution changed.")

        if not np.array_equal(candidate, candidate.T):
            raise RuntimeError("Random topology not symmetric.")

        return normalize_adjacency(candidate)

    raise RuntimeError("Could not generate random adjacency.")


class SAGEConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.self_linear = nn.Linear(
            input_dim, output_dim
        )
        self.neighbor_linear = nn.Linear(
            input_dim, output_dim, bias=False
        )

    def forward(self, x, weights):
        return (
            self.self_linear(x)
            + self.neighbor_linear(
                torch.matmul(weights, x)
            )
        )


class GraphSAGE(nn.Module):
    def __init__(self, original_weights):
        super().__init__()
        self.register_buffer(
            "neighbor_weights",
            original_weights.clone(),
        )
        self.conv1 = SAGEConv(384, 256)
        self.conv2 = SAGEConv(256, 384)

    def forward(self, x, override=None):
        weights = (
            self.neighbor_weights
            if override is None else override
        )

        h = F.gelu(self.conv1(x, weights))
        return F.normalize(
            self.conv2(h, weights), dim=-1
        )


class ResidualGraphSAGE(nn.Module):
    def __init__(self, original_weights):
        super().__init__()
        self.register_buffer(
            "neighbor_weights",
            original_weights.clone(),
        )
        self.conv1 = SAGEConv(384, 256)
        self.conv2 = SAGEConv(256, 384)
        self.gate_logit = nn.Parameter(
            torch.zeros((), dtype=torch.float32)
        )

    def forward(self, x, override=None):
        weights = (
            self.neighbor_weights
            if override is None else override
        )

        identity = F.normalize(x, dim=-1)

        hidden = F.gelu(
            self.conv1(x, weights)
        )
        graph = F.normalize(
            self.conv2(hidden, weights),
            dim=-1,
        )

        gate = torch.sigmoid(self.gate_logit)

        return F.normalize(
            identity + gate * graph,
            dim=-1,
        )


def retrieval_metrics(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)

    scores = a @ b.T
    n = scores.shape[0]
    targets = torch.arange(n, device=a.device)

    positive = scores.diagonal()

    negative_scores = scores.clone()
    negative_scores.fill_diagonal_(float("-inf"))

    hard_negative = negative_scores.max(dim=1).values
    margin = positive - hard_negative

    ranks = 1 + (
        scores > positive[:, None]
    ).sum(dim=1)

    at2_similarity = a @ a.T
    gt_similarity = b @ b.T

    mask = ~torch.eye(
        n, device=a.device, dtype=torch.bool
    )

    return {
        "recall_at_1": float(
            (scores.argmax(-1) == targets)
            .float().mean().item()
        ),
        "mrr": float(
            (1.0 / ranks.float()).mean().item()
        ),
        "hard_negative_margin": float(
            margin.mean().item()
        ),
        "matched_cosine": float(
            positive.mean().item()
        ),
        "at2_offdiagonal_cosine": float(
            at2_similarity[mask].mean().item()
        ),
        "gt450_offdiagonal_cosine": float(
            gt_similarity[mask].mean().item()
        ),
    }


print("=" * 70, flush=True)
print("PHASE 4.2.15 - TOPOLOGY ROBUSTNESS AUDIT", flush=True)
print("=" * 70, flush=True)


# --------------------------------------------------
# 1. Validate provenance
# --------------------------------------------------

print("\n[1] Verifying experiment sources...", flush=True)

for path in (
    GRAPH, GRAPH_REPORT, PROTOCOL,
    ARCH, *MODEL_REPORTS.values()
):
    if not path.is_file():
        raise RuntimeError(f"Missing source: {path}")

for path in (TABLE, SUMMARY, REPORT):
    if path.exists():
        raise RuntimeError(
            f"Refusing to overwrite: {path}"
        )

graph_hash = sha256(GRAPH)
protocol_hash = sha256(PROTOCOL)
architecture_hash = sha256(ARCH)

if (
    load_json(GRAPH_REPORT)["graph_archive_sha256"]
    != graph_hash
):
    raise RuntimeError("Graph archive hash mismatch.")

if (
    load_json(PROTOCOL)["source_graph_sha256"]
    != graph_hash
):
    raise RuntimeError("Experimental protocol mismatch.")

architecture = load_json(ARCH)

if (
    architecture["graph_archive_sha256"] != graph_hash
    or architecture["initial_protocol_sha256"] != protocol_hash
):
    raise RuntimeError("Architecture provenance mismatch.")

references = {
    name: load_json(path)
    for name, path in MODEL_REPORTS.items()
}

for name, reference in references.items():
    if (
        reference["graph_archive_sha256"] != graph_hash
        or reference["protocol_sha256"] != protocol_hash
        or reference.get("test_evaluated", False)
    ):
        raise RuntimeError(
            f"Reference integrity mismatch: {name}"
        )

if (
    references["ResidualGate05"]["parent_architecture_sha256"]
    != architecture_hash
):
    raise RuntimeError("Residual model architecture mismatch.")


# --------------------------------------------------
# 2. Load validation features only
# --------------------------------------------------

print("\n[2] Loading paired validation regions...", flush=True)

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
    raise RuntimeError("Invalid graph feature archive.")

if dict(Counter(splits)) != {
    "train": 80, "validation": 20, "test": 20
}:
    raise RuntimeError("Unexpected region partitions.")

by_region = defaultdict(dict)

for i, region in enumerate(regions):
    if splits[i] != "validation":
        continue

    scanner = scanners[i]

    if scanner in by_region[region]:
        raise RuntimeError("Duplicate scanner graph.")

    by_region[region][scanner] = i

if len(by_region) != 10:
    raise RuntimeError("Expected 10 validation regions.")

region_names = sorted(by_region)
pairs = {}

for region in region_names:
    indices = by_region[region]

    if set(indices) != {"AT2", "GT450"}:
        raise RuntimeError("Incomplete validation pair.")

    ai = indices["AT2"]
    bi = indices["GT450"]

    if not np.array_equal(
        tiles[ai], tiles[bi]
    ):
        raise RuntimeError(
            "Scanner node ordering mismatch."
        )

    pairs[region] = (
        torch.from_numpy(features[ai]).to(DEVICE),
        torch.from_numpy(features[bi]).to(DEVICE),
    )

base_adjacency = make_adjacency(edges)
original_weights = torch.from_numpy(
    normalize_adjacency(base_adjacency)
).to(DEVICE)

print("Validation regions:", len(pairs))
print("Randomization repetitions:", REPETITIONS)
print("Test partition: NOT EVALUATED")


# --------------------------------------------------
# 3. Evaluate frozen checkpoints
# --------------------------------------------------

print("\n[3] Evaluating topology interventions...", flush=True)

conditions = (
    "original",
    "shared_random",
    "scanner_discordant_random",
)

rows = []

for model_name in ("GraphSAGE", "ResidualGate05"):
    reference = references[model_name]
    folder, filename = CHECKPOINTS[model_name]

    for seed in SEEDS:
        checkpoint = (
            BASE / "outputs/checkpoints"
            / folder
            / filename.format(seed=seed)
        )

        if not checkpoint.is_file():
            raise RuntimeError(
                f"Missing checkpoint: {checkpoint}"
            )

        if (
            sha256(checkpoint)
            != reference["checkpoint_hashes"][str(seed)]
        ):
            raise RuntimeError(
                "Checkpoint integrity mismatch."
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
            or payload["test_evaluated"] is not False
        ):
            raise RuntimeError(
                "Checkpoint provenance mismatch."
            )

        if model_name == "GraphSAGE":
            model = GraphSAGE(original_weights)
        else:
            if (
                payload["parent_architecture_sha256"]
                != architecture_hash
                or payload["initial_gate"] != 0.5
            ):
                raise RuntimeError(
                    "Residual checkpoint metadata mismatch."
                )

            model = ResidualGraphSAGE(
                original_weights
            )

        model.load_state_dict(
            payload["model_state_dict"],
            strict=True,
        )

        model = model.to(DEVICE)
        model.eval()

        for repetition in range(REPETITIONS):
            for region in region_names:
                a, b = pairs[region]

                shared = random_adjacency(
                    base_adjacency,
                    seed, repetition, region, "shared"
                )

                at2_random = random_adjacency(
                    base_adjacency,
                    seed, repetition, region, "AT2"
                )

                gt_random = random_adjacency(
                    base_adjacency,
                    seed, repetition, region, "GT450"
                )

                adjacency_options = {
                    "original": (
                        original_weights,
                        original_weights,
                    ),
                    "shared_random": (
                        torch.from_numpy(shared).to(DEVICE),
                        torch.from_numpy(shared).to(DEVICE),
                    ),
                    "scanner_discordant_random": (
                        torch.from_numpy(at2_random).to(DEVICE),
                        torch.from_numpy(gt_random).to(DEVICE),
                    ),
                }

                with torch.inference_mode():
                    for condition in conditions:
                        wa, wb = adjacency_options[condition]

                        za = model(a, override=wa)
                        zb = model(b, override=wb)

                        result = retrieval_metrics(
                            za, zb
                        )

                        rows.append({
                            "model": model_name,
                            "seed": seed,
                            "repetition": repetition,
                            "region_id": region,
                            "condition": condition,
                            **result,
                        })

        print(
            f"  {model_name}, seed {seed}: completed",
            flush=True,
        )


# --------------------------------------------------
# 4. Verify unchanged original scores
# --------------------------------------------------

print("\n[4] Verifying baseline reproduction...", flush=True)

for model_name in ("GraphSAGE", "ResidualGate05"):
    reference = references[model_name]

    for seed in SEEDS:
        subset = [
            row for row in rows
            if (
                row["model"] == model_name
                and row["seed"] == seed
                and row["condition"] == "original"
            )
        ]

        observed = float(np.mean([
            row["recall_at_1"] for row in subset
        ]))

        saved = next(
            row for row in reference["seed_results"]
            if int(row["seed"]) == seed
        )["validation_recall_at_1"]

        if abs(observed - float(saved)) > 1e-5:
            raise RuntimeError(
                f"Baseline reproduction failed: "
                f"{model_name}, seed={seed}, "
                f"observed={observed:.6f}, "
                f"expected={float(saved):.6f}"
            )

print("Original checkpoint performance: VERIFIED")


# --------------------------------------------------
# 5. Aggregate model and condition results
# --------------------------------------------------

print("\n[5] Aggregating robustness statistics...", flush=True)

metrics = (
    "recall_at_1",
    "mrr",
    "hard_negative_margin",
    "matched_cosine",
    "at2_offdiagonal_cosine",
    "gt450_offdiagonal_cosine",
)

summary_rows = []

for model_name in ("GraphSAGE", "ResidualGate05"):
    for condition in conditions:
        subset = [
            row for row in rows
            if (
                row["model"] == model_name
                and row["condition"] == condition
            )
        ]

        result = {
            "model": model_name,
            "condition": condition,
            "seeds": len(SEEDS),
            "repetitions": REPETITIONS,
            "regions": len(region_names),
        }

        for metric in metrics:
            result[f"mean_{metric}"] = float(
                np.mean([
                    row[metric] for row in subset
                ])
            )

        original = [
            row for row in rows
            if (
                row["model"] == model_name
                and row["condition"] == "original"
            )
        ]

        original_mean = np.mean([
            row["recall_at_1"] for row in original
        ])

        result["change_from_original_pp"] = float(
            (
                result["mean_recall_at_1"]
                - original_mean
            ) * 100
        )

        summary_rows.append(result)

        print(
            f"  {model_name:16s} | "
            f"{condition:27s} | "
            f"R@1 {result['mean_recall_at_1']:.4f} | "
            f"Delta {result['change_from_original_pp']:+.2f} pp",
            flush=True,
        )


# --------------------------------------------------
# 6. Save reproducible artifacts
# --------------------------------------------------

print("\n[6] Saving robustness audit...", flush=True)

save_csv_new(
    TABLE,
    rows,
    [
        "model", "seed", "repetition",
        "region_id", "condition",
        *metrics,
    ],
)

save_csv_new(
    SUMMARY,
    summary_rows,
    [
        "model", "condition",
        "seeds", "repetitions", "regions",
        *[f"mean_{metric}" for metric in metrics],
        "change_from_original_pp",
    ],
)

report = {
    "phase": "4.2",
    "step": "4.2.15",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "experiment": "frozen_checkpoint_topology_robustness",
    "graph_archive_sha256": graph_hash,
    "protocol_sha256": protocol_hash,
    "architecture_sha256": architecture_hash,
    "source_report_hashes": {
        name: sha256(path)
        for name, path in MODEL_REPORTS.items()
    },
    "source_checkpoint_hashes": {
        name: references[name]["checkpoint_hashes"]
        for name in MODEL_REPORTS
    },
    "seeds": list(SEEDS),
    "repetitions": REPETITIONS,
    "validation_regions": len(region_names),
    "conditions": list(conditions),
    "degree_multiset_preserved": True,
    "graph_edge_count_preserved": True,
    "node_features_changed": False,
    "training_performed": False,
    "checkpoint_selection_performed": False,
    "test_evaluated": False,
    "models": summary_rows,
    "detail_table_sha256": sha256(TABLE),
    "summary_table_sha256": sha256(SUMMARY),
    "limitations": [
        "Inference-time topology interventions",
        "Randomized graphs are out of training distribution",
        "AT2 and GT450 use same graph in shared condition",
        "Independent scanner graphs may differ in node degree",
        "Repeated randomizations are not independent specimens",
        "Only 10 spatial validation regions",
        "No claim of biological independence",
        "Validation analysis remains exploratory",
    ],
    "xai_planned": True,
}

save_json_new(REPORT, report)

print("\n" + "=" * 70)
print("PHASE 4.2.15 - TOPOLOGY ROBUSTNESS COMPLETED")
print("=" * 70)

for result in summary_rows:
    print(
        f"{result['model']} | "
        f"{result['condition']} | "
        f"R@1 {result['mean_recall_at_1']:.4f} | "
        f"Difference {result['change_from_original_pp']:+.2f} pp"
    )

print("\nSaved:")
print(TABLE)
print(SUMMARY)
print(REPORT)

print("\nTraining performed: NO")
print("Existing checkpoints modified: NO")
print("Test partition evaluated: NO")
print("XAI: PLANNED")
print("\nSTEP 4.2.15 COMPLETED")
