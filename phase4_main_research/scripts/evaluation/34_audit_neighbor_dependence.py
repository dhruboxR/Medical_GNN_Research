
"""
PHASE 4.2.19 - NEIGHBORHOOD DEPENDENCE AUDIT

Models:
  ResidualGate05 (lambda=0)
  TopologyConsistentResidual (lambda=1)

Validation: 10 regions, 3 seeds, 5 perturbations.

Conditions:
  normal_spatial
  neighbors_disabled
  scanner_discordant_random

Also measures:
  - final embedding vs input identity cosine
  - graph-branch vs identity cosine
  - learned residual gate
  - retrieval Recall@1, MRR

No training, checkpoint selection, or test evaluation.
Previous artifacts are never overwritten.
"""

from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import hashlib
import json
import csv
import math

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

REPORTS = {
    "ResidualGate05":
        BASE / "outputs/reports/28_gate_init05_report.json",
    "TopologyConsistentResidual":
        BASE / "outputs/reports/31_topology_consistent_report.json",
}

ROBUSTNESS_REPORT = (
    BASE / "outputs/reports/30_topology_robustness_report.json"
)

DETAIL = BASE / "outputs/tables/34_neighbor_dependence_detail.csv"
SUMMARY = BASE / "outputs/tables/34_neighbor_dependence_summary.csv"
REPORT = BASE / "outputs/reports/34_neighbor_dependence_report.json"

SEEDS = (42, 123, 2026)
REPETITIONS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, records, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def deterministic_rng(*parts):
    key = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    seed = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(seed)


def adjacency_from_edges(edges):
    if edges.shape != (2, 120):
        raise RuntimeError("Invalid edge-index shape.")

    adjacency = np.zeros((36, 36), dtype=np.float32)
    observed = set()

    for source, target in edges.T.tolist():
        source, target = int(source), int(target)

        if (
            source == target
            or not 0 <= source < 36
            or not 0 <= target < 36
            or (source, target) in observed
        ):
            raise RuntimeError("Invalid spatial edge.")

        observed.add((source, target))
        adjacency[target, source] = 1.0

    if len(observed) != 120:
        raise RuntimeError("Unexpected edge count.")

    if any((b, a) not in observed for a, b in observed):
        raise RuntimeError("Nonreciprocal adjacency.")

    if not np.array_equal(adjacency, adjacency.T):
        raise RuntimeError("Adjacency symmetry error.")

    return adjacency


def normalize_adjacency(adjacency):
    degree = adjacency.sum(axis=1, keepdims=True)

    if np.any(degree <= 0):
        raise RuntimeError("Isolated node.")

    return (adjacency / degree).astype(np.float32)


def random_adjacency(original, *key):
    # Exactly matches the Phase 4.2.15 perturbation keys.
    rng = deterministic_rng("phase4215", *key)

    for _ in range(200):
        permutation = rng.permutation(36)
        candidate = original[
            np.ix_(permutation, permutation)
        ]

        if np.array_equal(candidate, original):
            continue

        if int(candidate.sum()) != 120:
            raise RuntimeError("Edge count changed.")

        if not np.array_equal(
            np.sort(candidate.sum(axis=1)),
            np.sort(original.sum(axis=1)),
        ):
            raise RuntimeError("Degree multiset changed.")

        if not np.array_equal(candidate, candidate.T):
            raise RuntimeError("Symmetry changed.")

        return normalize_adjacency(candidate)

    raise RuntimeError("Could not create random topology.")


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


class ResidualGraphSAGE(nn.Module):
    def __init__(self, weights):
        super().__init__()

        self.register_buffer(
            "neighbor_weights",
            weights.clone()
        )

        self.conv1 = SAGEConv(384, 256)
        self.conv2 = SAGEConv(256, 384)
        self.gate_logit = nn.Parameter(torch.zeros(()))

    def gate(self):
        return torch.sigmoid(self.gate_logit)

    def branches(self, x, override=None):
        weights = (
            self.neighbor_weights
            if override is None
            else override
        )

        identity = F.normalize(x, dim=-1)

        h = F.gelu(self.conv1(x, weights))
        graph = F.normalize(
            self.conv2(h, weights), dim=-1
        )

        final = F.normalize(
            identity + self.gate() * graph,
            dim=-1
        )

        return identity, graph, final

    def forward(self, x, override=None):
        return self.branches(x, override)[-1]


def retrieval(a, b):
    sim = a @ b.T
    targets = torch.arange(len(a), device=a.device)
    positives = sim.diagonal()

    ranks = 1 + (
        sim > positives[:, None]
    ).sum(-1)

    return {
        "recall_at_1": float(
            (sim.argmax(-1) == targets)
            .float().mean().item()
        ),
        "mrr": float(
            (1.0 / ranks.float()).mean().item()
        ),
    }


def representation_metrics(
    model, a, b, wa, wb
):
    ia, ga, za = model.branches(a, wa)
    ib, gb, zb = model.branches(b, wb)

    result = retrieval(za, zb)

    result.update({
        "final_identity_cosine": float(
            0.5 * (
                F.cosine_similarity(
                    za, ia, dim=-1
                ).mean().item()
                + F.cosine_similarity(
                    zb, ib, dim=-1
                ).mean().item()
            )
        ),
        "graph_identity_cosine": float(
            0.5 * (
                F.cosine_similarity(
                    ga, ia, dim=-1
                ).mean().item()
                + F.cosine_similarity(
                    gb, ib, dim=-1
                ).mean().item()
            )
        ),
        "learned_gate": float(model.gate().item()),
    })

    return result


def checkpoint_path(name, seed):
    if name == "ResidualGate05":
        return (
            BASE
            / "outputs/checkpoints/28_residual_gate_init05"
            / f"gate_init05_seed_{seed}.pt"
        )

    return (
        BASE
        / "outputs/checkpoints/31_topology_consistent_residual"
        / f"topology_consistent_seed_{seed}.pt"
    )


print("=" * 70, flush=True)
print("PHASE 4.2.19 - NEIGHBORHOOD DEPENDENCE AUDIT", flush=True)
print("=" * 70, flush=True)

print("\n[1] Verifying sources...", flush=True)

sources = [
    GRAPH, GRAPH_REPORT, PROTOCOL, ARCH,
    ROBUSTNESS_REPORT, *REPORTS.values(),
]

for path in sources:
    if not path.is_file():
        raise RuntimeError(f"Missing source: {path}")

for path in (DETAIL, SUMMARY, REPORT):
    if path.exists():
        raise RuntimeError(f"Refusing overwrite: {path}")

graph_hash = sha256(GRAPH)
protocol_hash = sha256(PROTOCOL)
arch_hash = sha256(ARCH)

if read_json(GRAPH_REPORT)["graph_archive_sha256"] != graph_hash:
    raise RuntimeError("Graph provenance mismatch.")

if read_json(PROTOCOL)["source_graph_sha256"] != graph_hash:
    raise RuntimeError("Protocol mismatch.")

architecture = read_json(ARCH)

if (
    architecture["graph_archive_sha256"] != graph_hash
    or architecture["initial_protocol_sha256"] != protocol_hash
):
    raise RuntimeError("Architecture provenance mismatch.")

references = {
    name: read_json(path)
    for name, path in REPORTS.items()
}

robustness_reference = read_json(ROBUSTNESS_REPORT)

for name, reference in references.items():
    if (
        reference["graph_archive_sha256"] != graph_hash
        or reference["protocol_sha256"] != protocol_hash
        or reference["parent_architecture_sha256"] != arch_hash
        or reference["test_evaluated"] is not False
    ):
        raise RuntimeError(f"Model provenance mismatch: {name}")

    if tuple(reference["seeds"]) != SEEDS:
        raise RuntimeError("Training seed mismatch.")

if (
    robustness_reference["graph_archive_sha256"] != graph_hash
    or robustness_reference["protocol_sha256"] != protocol_hash
    or robustness_reference["test_evaluated"] is not False
    or robustness_reference["repetitions"] != REPETITIONS
):
    raise RuntimeError("Robustness protocol mismatch.")


print("\n[2] Preparing validation pairs...", flush=True)

with np.load(GRAPH, allow_pickle=False) as archive:
    features = archive["node_features"].copy()
    edges = archive["edge_index"].copy()
    regions = archive["region_id"].astype(str)
    splits = archive["spatial_split"].astype(str)
    scanners = archive["scanner"].astype(str)
    tile_ids = archive["tile_id"].astype(str)

if (
    features.shape != (120, 36, 384)
    or features.dtype != np.float32
    or not np.isfinite(features).all()
):
    raise RuntimeError("Invalid feature archive.")

if Counter(splits) != Counter({
    "train": 80,
    "validation": 20,
    "test": 20,
}):
    raise RuntimeError("Split counts incorrect.")

by_region = defaultdict(dict)

for index, region in enumerate(regions):
    if splits[index] != "validation":
        continue

    scanner = scanners[index]

    if scanner in by_region[region]:
        raise RuntimeError("Duplicate scanner graph.")

    by_region[region][scanner] = index

if len(by_region) != 10:
    raise RuntimeError("Expected 10 validation regions.")

pairs = {}

for region, scans in sorted(by_region.items()):
    if set(scans) != {"AT2", "GT450"}:
        raise RuntimeError("Incomplete scanner pair.")

    ai, bi = scans["AT2"], scans["GT450"]

    if not np.array_equal(tile_ids[ai], tile_ids[bi]):
        raise RuntimeError("Unmatched node order.")

    pairs[region] = (
        torch.from_numpy(features[ai]).to(DEVICE),
        torch.from_numpy(features[bi]).to(DEVICE),
    )

base = adjacency_from_edges(edges)

weights = torch.from_numpy(
    normalize_adjacency(base)
).to(DEVICE)

zero_weights = torch.zeros_like(weights)

print("Validation regions:", len(pairs))
print("Seeds:", SEEDS)
print("Perturbation repetitions:", REPETITIONS)
print("Test: RESERVED")


print("\n[3] Evaluating saved models...", flush=True)

rows = []

for model_name, reference in references.items():
    for seed in SEEDS:
        path = checkpoint_path(model_name, seed)

        if not path.is_file():
            raise RuntimeError(f"Missing checkpoint: {path}")

        if sha256(path) != reference["checkpoint_hashes"][str(seed)]:
            raise RuntimeError("Checkpoint hash mismatch.")

        payload = torch.load(
            path, map_location="cpu", weights_only=True
        )

        if (
            int(payload["seed"]) != seed
            or payload["graph_archive_sha256"] != graph_hash
            or payload["protocol_sha256"] != protocol_hash
            or payload["parent_architecture_sha256"] != arch_hash
            or payload["initial_gate"] != 0.5
            or payload["test_evaluated"] is not False
        ):
            raise RuntimeError("Checkpoint metadata mismatch.")

        if model_name == "TopologyConsistentResidual":
            if payload["lambda_topology"] != 1.0:
                raise RuntimeError("Incorrect topology weight.")

        model = ResidualGraphSAGE(weights).to(DEVICE)
        model.load_state_dict(
            payload["model_state_dict"], strict=True
        )
        model.eval()

        with torch.inference_mode():
            for region, (a, b) in pairs.items():
                # Evaluate normal and neighbor-disabled once.
                # Store each on repetition=0, rather than
                # falsely treating it as five observations.
                for condition, wa, wb in (
                    ("normal_spatial", weights, weights),
                    ("neighbors_disabled",
                     zero_weights, zero_weights),
                ):
                    result = representation_metrics(
                        model, a, b, wa, wb
                    )

                    rows.append({
                        "model": model_name,
                        "seed": seed,
                        "region_id": region,
                        "repetition": 0,
                        "condition": condition,
                        **result,
                    })

                for repetition in range(REPETITIONS):
                    wa = torch.from_numpy(
                        random_adjacency(
                            base, seed, repetition,
                            region, "AT2"
                        )
                    ).to(DEVICE)

                    wb = torch.from_numpy(
                        random_adjacency(
                            base, seed, repetition,
                            region, "GT450"
                        )
                    ).to(DEVICE)

                    result = representation_metrics(
                        model, a, b, wa, wb
                    )

                    rows.append({
                        "model": model_name,
                        "seed": seed,
                        "region_id": region,
                        "repetition": repetition,
                        "condition": "scanner_discordant_random",
                        **result,
                    })

        print(
            f"  {model_name}, seed {seed}: completed",
            flush=True,
        )


print("\n[4] Checking prior results...", flush=True)

# Each original condition contains 10 regions, one
# observation per seed. Each discordant condition
# contains 50 region-perturbation observations per seed.

for model_name, reference in references.items():
    for seed in SEEDS:
        normal = [
            r for r in rows
            if (
                r["model"] == model_name
                and r["seed"] == seed
                and r["condition"] == "normal_spatial"
            )
        ]

        if len(normal) != 10:
            raise RuntimeError("Normal condition incomplete.")

        reproduced = float(np.mean([
            r["recall_at_1"] for r in normal
        ]))

        expected = next(
            r["validation_recall_at_1"]
            for r in reference["seed_results"]
            if int(r["seed"]) == seed
        )

        if abs(reproduced - float(expected)) > 1e-5:
            raise RuntimeError(
                f"Normal baseline reproduction failed: "
                f"{model_name}, seed {seed}"
            )

    discordant = [
        r for r in rows
        if (
            r["model"] == model_name
            and r["condition"] == "scanner_discordant_random"
        )
    ]

    if len(discordant) != 150:
        raise RuntimeError("Discordant condition incomplete.")

    observed = float(np.mean([
        r["recall_at_1"] for r in discordant
    ]))

    if model_name == "TopologyConsistentResidual":
        expected = reference[
            "mean_scanner_discordant_recall_at_1"
        ]
    else:
        candidates = [
            r for r in robustness_reference["models"]
            if (
                r["model"] == "ResidualGate05"
                and r["condition"]
                == "scanner_discordant_random"
            )
        ]

        if len(candidates) != 1:
            raise RuntimeError("Missing robustness reference.")

        expected = candidates[0]["mean_recall_at_1"]

    if abs(observed - float(expected)) > 1e-5:
        raise RuntimeError(
            f"Discordant reproduction failed: {model_name}"
        )

print("Normal and discordant retrieval: VERIFIED")


print("\n[5] Summarizing neighborhood dependence...", flush=True)

conditions = (
    "normal_spatial",
    "neighbors_disabled",
    "scanner_discordant_random",
)

metric_names = (
    "recall_at_1",
    "mrr",
    "final_identity_cosine",
    "graph_identity_cosine",
    "learned_gate",
)

summary_rows = []

for model_name in references:
    grouped = {}

    for condition in conditions:
        subset = [
            r for r in rows
            if (
                r["model"] == model_name
                and r["condition"] == condition
            )
        ]

        expected_count = (
            150 if condition == "scanner_discordant_random"
            else 30
        )

        if len(subset) != expected_count:
            raise RuntimeError("Unexpected sample count.")

        summary = {
            "model": model_name,
            "condition": condition,
            "measurements": len(subset),
        }

        for metric in metric_names:
            summary[f"mean_{metric}"] = float(
                np.mean([r[metric] for r in subset])
            )

        grouped[condition] = summary

    normal_recall = grouped[
        "normal_spatial"
    ]["mean_recall_at_1"]

    disabled_recall = grouped[
        "neighbors_disabled"
    ]["mean_recall_at_1"]

    discordant_recall = grouped[
        "scanner_discordant_random"
    ]["mean_recall_at_1"]

    for condition in conditions:
        row = grouped[condition]
        row["neighbor_removal_drop_pp"] = (
            normal_recall - disabled_recall
        ) * 100

        row["discordant_drop_pp"] = (
            normal_recall - discordant_recall
        ) * 100

        summary_rows.append(row)

    print(
        f"  {model_name}: "
        f"Normal {normal_recall:.4f} | "
        f"No neighbors {disabled_recall:.4f} | "
        f"Discordant {discordant_recall:.4f} | "
        f"Removal penalty "
        f"{(normal_recall-disabled_recall)*100:+.2f} pp",
        flush=True,
    )


print("\n[6] Saving diagnostic outputs...", flush=True)

write_csv(
    DETAIL,
    rows,
    [
        "model", "seed", "region_id",
        "repetition", "condition",
        *metric_names,
    ],
)

write_csv(
    SUMMARY,
    summary_rows,
    [
        "model", "condition", "measurements",
        *[f"mean_{m}" for m in metric_names],
        "neighbor_removal_drop_pp",
        "discordant_drop_pp",
    ],
)

report = {
    "phase": "4.2",
    "step": "4.2.19",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "experiment": "neighbor_dependence_audit",
    "graph_archive_sha256": graph_hash,
    "protocol_sha256": protocol_hash,
    "architecture_sha256": arch_hash,
    "source_model_report_hashes": {
        name: sha256(path)
        for name, path in REPORTS.items()
    },
    "source_robustness_report_sha256":
        sha256(ROBUSTNESS_REPORT),
    "training_seeds": list(SEEDS),
    "validation_regions": 10,
    "discordant_repetitions": REPETITIONS,
    "intervention": (
        "zero neighbor matrices in both SAGE layers "
        "during inference, without retraining"
    ),
    "summary": summary_rows,
    "detail_sha256": sha256(DETAIL),
    "summary_sha256": sha256(SUMMARY),
    "training_performed": False,
    "checkpoint_selection_performed": False,
    "test_evaluated": False,
    "limitations": [
        "Neighbor removal is an inference-time intervention",
        "No-neighbor checkpoints were not retrained",
        "Different active message strengths may affect sensitivity",
        "Same ten validation regions repeatedly analyzed",
        "Synthetic topology changes are not real scanner shifts",
        "Results do not establish clinical generalization",
    ],
    "xai_planned": True,
}

write_json(REPORT, report)

print("\n" + "=" * 70)
print("PHASE 4.2.19 - NEIGHBOR AUDIT COMPLETED")
print("=" * 70)

for row in summary_rows:
    print(
        f"{row['model']} | {row['condition']} | "
        f"R@1 {row['mean_recall_at_1']:.4f} | "
        f"Final-identity cosine "
        f"{row['mean_final_identity_cosine']:.4f}"
    )

print("\nSaved:")
print(DETAIL)
print(SUMMARY)
print(REPORT)

print("\nTraining: NO")
print("Existing checkpoints modified: NO")
print("Held-out test evaluated: NO")
print("XAI: PLANNED")
print("\nSTEP 4.2.19 COMPLETED")
