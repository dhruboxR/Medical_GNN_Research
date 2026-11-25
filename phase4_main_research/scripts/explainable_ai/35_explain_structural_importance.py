
"""
PHASE 4.2.20 - STRUCTURAL EXPLAINABILITY AUDIT

Models:
    ResidualGate05, lambda=0
    TopologyConsistentResidual, lambda=1

Technique:
    Leave-one-node's-incident-edges-out perturbation.

For each region and scanner:
    1. Compute original bidirectional InfoNCE score.
    2. Delete the edges incident to each node.
    3. Measure the retrieval-objective decrease.
    4. Compare AT2/GT450 importance rankings.
    5. Evaluate top-5 versus bottom-5 edge deletion.

This tests structural importance and explanation
consistency, not tissue-level biological relevance.

10 validation regions, 3 seeds, no training.
No test data, checkpoint modifications, or model selection.
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

SOURCES = {
    "ResidualGate05": {
        "report": BASE / "outputs/reports/28_gate_init05_report.json",
        "folder": BASE / "outputs/checkpoints/28_residual_gate_init05",
        "filename": "gate_init05_seed_{seed}.pt",
    },
    "TopologyConsistentResidual": {
        "report": BASE / "outputs/reports/31_topology_consistent_report.json",
        "folder": BASE / "outputs/checkpoints/31_topology_consistent_residual",
        "filename": "topology_consistent_seed_{seed}.pt",
    },
}

DETAIL = BASE / "outputs/xai/35_structural_node_importance.csv"
SUMMARY = BASE / "outputs/xai/35_structural_explanation_summary.csv"
REPORT = BASE / "outputs/reports/35_structural_xai_report.json"

SEEDS = (42, 123, 2026)
TEMPERATURE = 0.07
TOP_K = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def adjacency_from_edges(edges):
    if edges.shape != (2, 120):
        raise RuntimeError("Unexpected edge-index shape.")

    adjacency = np.zeros((36, 36), dtype=np.float32)
    seen = set()

    for source, target in edges.T.tolist():
        source, target = int(source), int(target)

        if (
            source == target
            or not 0 <= source < 36
            or not 0 <= target < 36
            or (source, target) in seen
        ):
            raise RuntimeError("Invalid spatial edge.")

        seen.add((source, target))
        adjacency[target, source] = 1.0

    if len(seen) != 120:
        raise RuntimeError("Incorrect edge count.")

    if not np.array_equal(adjacency, adjacency.T):
        raise RuntimeError("Graph symmetry mismatch.")

    return adjacency


def normalized(a):
    degree = a.sum(-1, keepdims=True)

    # Zero-degree rows are allowed after intervention.
    # They receive zero neighbor messages.
    return np.divide(
        a,
        degree,
        out=np.zeros_like(a),
        where=degree > 0,
    ).astype(np.float32)


def remove_incident_edges(a, node_indices):
    changed = a.copy()

    for index in node_indices:
        changed[index, :] = 0.0
        changed[:, index] = 0.0

    return normalized(changed)


def average_ranks(values):
    """
    Ascending average ranks, including tied values.
    Used for descriptive Spearman correlation.
    """
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)

    sorted_x = x[order]
    start = 0

    while start < len(x):
        end = start + 1

        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1

        # Ranks are one-based.
        rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = rank
        start = end

    return ranks


def rank_correlation(x, y):
    a = average_ranks(x)
    b = average_ranks(y)

    a -= a.mean()
    b -= b.mean()

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator <= 1e-12:
        return None

    return float(np.dot(a, b) / denominator)


class SAGEConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neighbor_linear = nn.Linear(
            in_dim, out_dim, bias=False
        )

    def forward(self, x, weights):
        neighbors = torch.matmul(weights, x)

        return (
            self.self_linear(x)
            + self.neighbor_linear(neighbors)
        )


class ResidualGraphSAGE(nn.Module):
    def __init__(self, weights):
        super().__init__()

        self.register_buffer(
            "neighbor_weights", weights.clone()
        )

        self.conv1 = SAGEConv(384, 256)
        self.conv2 = SAGEConv(256, 384)
        self.gate_logit = nn.Parameter(torch.zeros(()))

    def forward(self, x, override=None):
        weights = (
            self.neighbor_weights
            if override is None else override
        )

        identity = F.normalize(x, dim=-1)
        hidden = F.gelu(self.conv1(x, weights))

        graph = F.normalize(
            self.conv2(hidden, weights), dim=-1
        )

        gate = torch.sigmoid(self.gate_logit)

        return F.normalize(
            identity + gate * graph, dim=-1
        )


def retrieval_accuracy(a, b):
    scores = a @ b.T
    target = torch.arange(len(a), device=a.device)

    return float(
        (scores.argmax(-1) == target).float().mean().item()
    )


def explanation_score(a, b):
    """
    Higher is better.

    Negative bidirectional within-region InfoNCE.
    """
    logits = (a @ b.T) / TEMPERATURE
    targets = torch.arange(
        len(a), device=a.device
    )

    return -0.5 * (
        F.cross_entropy(logits, targets)
        + F.cross_entropy(logits.T, targets)
    )


print("=" * 70, flush=True)
print("PHASE 4.2.20 - STRUCTURAL EXPLAINABILITY AUDIT", flush=True)
print("=" * 70, flush=True)


# --------------------------------------------------
# 1. Verify source provenance
# --------------------------------------------------

print("\n[1] Verifying experiment sources...", flush=True)

required = [
    GRAPH, GRAPH_REPORT, PROTOCOL, ARCH,
]

required += [
    item["report"] for item in SOURCES.values()
]

for path in required:
    if not path.is_file():
        raise RuntimeError(f"Missing input: {path}")

for path in (DETAIL, SUMMARY, REPORT):
    if path.exists():
        raise RuntimeError(f"Refusing overwrite: {path}")

graph_hash = sha256(GRAPH)
protocol_hash = sha256(PROTOCOL)
arch_hash = sha256(ARCH)

graph_report = read_json(GRAPH_REPORT)
protocol = read_json(PROTOCOL)
arch = read_json(ARCH)

if graph_report["graph_archive_sha256"] != graph_hash:
    raise RuntimeError("Graph hash mismatch.")

if protocol["source_graph_sha256"] != graph_hash:
    raise RuntimeError("Protocol mismatch.")

if (
    arch["graph_archive_sha256"] != graph_hash
    or arch["initial_protocol_sha256"] != protocol_hash
):
    raise RuntimeError("Architecture provenance mismatch.")

references = {}

for name, spec in SOURCES.items():
    report = read_json(spec["report"])

    if (
        report["graph_archive_sha256"] != graph_hash
        or report["protocol_sha256"] != protocol_hash
        or report["parent_architecture_sha256"] != arch_hash
        or report["test_evaluated"] is not False
    ):
        raise RuntimeError(f"Model source mismatch: {name}")

    if tuple(report["seeds"]) != SEEDS:
        raise RuntimeError("Seed mismatch.")

    references[name] = report


# --------------------------------------------------
# 2. Load validation region pairs
# --------------------------------------------------

print("\n[2] Loading validation graph pairs...", flush=True)

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
    raise RuntimeError("Invalid graph tensor.")

if Counter(splits) != Counter({
    "train": 80,
    "validation": 20,
    "test": 20,
}):
    raise RuntimeError("Spatial partition mismatch.")

by_region = defaultdict(dict)

for index, region in enumerate(regions):
    if splits[index] != "validation":
        continue

    scanner = scanners[index]

    if scanner in by_region[region]:
        raise RuntimeError("Duplicate validation graph.")

    by_region[region][scanner] = index

if len(by_region) != 10:
    raise RuntimeError("Expected ten validation regions.")

pairs = {}

for region, indices in sorted(by_region.items()):
    if set(indices) != {"AT2", "GT450"}:
        raise RuntimeError("Incomplete scanner pair.")

    ai, bi = indices["AT2"], indices["GT450"]

    if not np.array_equal(
        tile_ids[ai], tile_ids[bi]
    ):
        raise RuntimeError("Node correspondence mismatch.")

    pairs[region] = (
        torch.from_numpy(features[ai]).to(DEVICE),
        torch.from_numpy(features[bi]).to(DEVICE),
    )

base = adjacency_from_edges(edges)
weights = torch.from_numpy(normalized(base)).to(DEVICE)

degrees = base.sum(axis=1).astype(int)

# Precompute all node-removal and five-node
# intervention matrices where applicable.
single_interventions = [
    torch.from_numpy(
        remove_incident_edges(base, [node])
    ).to(DEVICE)
    for node in range(36)
]

print("Validation pairs:", len(pairs))
print("Nodes per graph: 36")
print("Structural interventions per scanner: 36")
print("Device:", DEVICE)
print("Test partition: NOT EVALUATED")


# --------------------------------------------------
# 3. Generate structural importance explanations
# --------------------------------------------------

print("\n[3] Computing structural attributions...", flush=True)

detail_rows = []
summary_rows = []

for model_name, spec in SOURCES.items():
    source = references[model_name]

    for seed in SEEDS:
        checkpoint = (
            spec["folder"]
            / spec["filename"].format(seed=seed)
        )

        if not checkpoint.is_file():
            raise RuntimeError(
                f"Missing checkpoint: {checkpoint}"
            )

        if (
            sha256(checkpoint)
            != source["checkpoint_hashes"][str(seed)]
        ):
            raise RuntimeError("Checkpoint hash mismatch.")

        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
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
                raise RuntimeError(
                    "Incorrect topology-consistency weight."
                )

        model = ResidualGraphSAGE(weights).to(DEVICE)

        model.load_state_dict(
            payload["model_state_dict"],
            strict=True,
        )
        model.eval()

        reproduced_recalls = []

        with torch.inference_mode():
            for region, (a, b) in pairs.items():
                original_a = model(a)
                original_b = model(b)

                base_score = float(
                    explanation_score(
                        original_a, original_b
                    ).item()
                )

                reproduced_recalls.append(
                    retrieval_accuracy(
                        original_a, original_b
                    )
                )

                importance_a = np.zeros(36, np.float64)
                importance_b = np.zeros(36, np.float64)

                for node, adjacency in enumerate(
                    single_interventions
                ):
                    altered_a = model(
                        a, override=adjacency
                    )

                    altered_b = model(
                        b, override=adjacency
                    )

                    score_a = float(
                        explanation_score(
                            altered_a, original_b
                        ).item()
                    )

                    score_b = float(
                        explanation_score(
                            original_a, altered_b
                        ).item()
                    )

                    importance_a[node] = (
                        base_score - score_a
                    )

                    importance_b[node] = (
                        base_score - score_b
                    )

                    for scanner, value in (
                        ("AT2", importance_a[node]),
                        ("GT450", importance_b[node]),
                    ):
                        detail_rows.append({
                            "model": model_name,
                            "seed": seed,
                            "region_id": region,
                            "scanner": scanner,
                            "node_index": node,
                            "node_degree": degrees[node],
                            "importance": float(value),
                        })

                rank_consistency = rank_correlation(
                    importance_a,
                    importance_b,
                )

                top_a = np.argsort(
                    -importance_a, kind="stable"
                )[:TOP_K]

                bottom_a = np.argsort(
                    importance_a, kind="stable"
                )[:TOP_K]

                top_b = np.argsort(
                    -importance_b, kind="stable"
                )[:TOP_K]

                bottom_b = np.argsort(
                    importance_b, kind="stable"
                )[:TOP_K]

                top_overlap = len(
                    set(top_a.tolist())
                    & set(top_b.tolist())
                )

                def fidelity(scanner, selected):
                    changed = torch.from_numpy(
                        remove_incident_edges(
                            base, selected.tolist()
                        )
                    ).to(DEVICE)

                    if scanner == "AT2":
                        za = model(a, override=changed)
                        zb = original_b
                    else:
                        za = original_a
                        zb = model(b, override=changed)

                    return base_score - float(
                        explanation_score(za, zb).item()
                    )

                top_a_drop = fidelity("AT2", top_a)
                bottom_a_drop = fidelity("AT2", bottom_a)

                top_b_drop = fidelity("GT450", top_b)
                bottom_b_drop = fidelity("GT450", bottom_b)

                summary_rows.append({
                    "model": model_name,
                    "seed": seed,
                    "region_id": region,
                    "baseline_recall_at_1":
                        reproduced_recalls[-1],
                    "attribution_rank_correlation":
                        rank_consistency,
                    "top5_overlap": top_overlap,
                    "mean_at2_importance":
                        float(importance_a.mean()),
                    "mean_gt450_importance":
                        float(importance_b.mean()),
                    "at2_top5_score_drop": top_a_drop,
                    "at2_bottom5_score_drop": bottom_a_drop,
                    "gt450_top5_score_drop": top_b_drop,
                    "gt450_bottom5_score_drop": bottom_b_drop,
                    "at2_fidelity_gap":
                        top_a_drop - bottom_a_drop,
                    "gt450_fidelity_gap":
                        top_b_drop - bottom_b_drop,
                })

        mean_reproduced = float(
            np.mean(reproduced_recalls)
        )

        expected = next(
            item["validation_recall_at_1"]
            for item in source["seed_results"]
            if int(item["seed"]) == seed
        )

        if abs(mean_reproduced - float(expected)) > 1e-5:
            raise RuntimeError(
                f"Baseline reproduction failed: "
                f"{model_name}, seed {seed}"
            )

        print(
            f"  {model_name}, seed {seed} | "
            f"Normal R@1 {mean_reproduced:.4f} | "
            f"Attribution complete",
            flush=True,
        )


# --------------------------------------------------
# 4. Aggregate model-level explanation measurements
# --------------------------------------------------

print("\n[4] Aggregating XAI results...", flush=True)

model_summary = []

for name in SOURCES:
    subset = [
        row for row in summary_rows
        if row["model"] == name
    ]

    if len(subset) != 30:
        raise RuntimeError("Expected 30 region-seed results.")

    correlations = [
        row["attribution_rank_correlation"]
        for row in subset
        if row["attribution_rank_correlation"] is not None
    ]

    if not correlations:
        raise RuntimeError(
            "No defined attribution correlations."
        )

    result = {
        "model": name,
        "region_seed_units": len(subset),
        "defined_rank_correlations": len(correlations),
        "mean_attribution_rank_correlation":
            float(np.mean(correlations)),
        "mean_top5_overlap":
            float(np.mean([
                row["top5_overlap"] for row in subset
            ])),
        "mean_at2_fidelity_gap":
            float(np.mean([
                row["at2_fidelity_gap"]
                for row in subset
            ])),
        "mean_gt450_fidelity_gap":
            float(np.mean([
                row["gt450_fidelity_gap"]
                for row in subset
            ])),
    }

    model_summary.append(result)

    print(
        f"  {name} | "
        f"Rank correlation "
        f"{result['mean_attribution_rank_correlation']:.4f} | "
        f"Top-5 overlap "
        f"{result['mean_top5_overlap']:.2f}/5 | "
        f"AT2 fidelity gap "
        f"{result['mean_at2_fidelity_gap']:.4f}",
        flush=True,
    )


# --------------------------------------------------
# 5. Save all XAI outputs
# --------------------------------------------------

print("\n[5] Saving explanation artifacts...", flush=True)

save_csv_new(
    DETAIL,
    detail_rows,
    [
        "model", "seed", "region_id",
        "scanner", "node_index",
        "node_degree", "importance",
    ],
)

summary_fields = [
    "model", "seed", "region_id",
    "baseline_recall_at_1",
    "attribution_rank_correlation",
    "top5_overlap",
    "mean_at2_importance",
    "mean_gt450_importance",
    "at2_top5_score_drop",
    "at2_bottom5_score_drop",
    "gt450_top5_score_drop",
    "gt450_bottom5_score_drop",
    "at2_fidelity_gap",
    "gt450_fidelity_gap",
]

save_csv_new(
    SUMMARY,
    summary_rows,
    summary_fields,
)

report = {
    "phase": "4.2",
    "step": "4.2.20",
    "generated_utc":
        datetime.now(timezone.utc).isoformat(),
    "experiment":
        "structural_edge_perturbation_explanations",
    "method":
        "node_incident_edge_removal",
    "retrieval_objective":
        "negative_bidirectional_within_region_InfoNCE",
    "temperature": TEMPERATURE,
    "top_k": TOP_K,
    "graph_archive_sha256": graph_hash,
    "protocol_sha256": protocol_hash,
    "architecture_sha256": arch_hash,
    "source_reports_sha256": {
        name: sha256(spec["report"])
        for name, spec in SOURCES.items()
    },
    "validation_regions": len(pairs),
    "training_seeds": list(SEEDS),
    "model_summary": model_summary,
    "detail_table_sha256": sha256(DETAIL),
    "summary_table_sha256": sha256(SUMMARY),
    "training_performed": False,
    "checkpoint_selection_performed": False,
    "test_evaluated": False,
    "limitations": [
        "Structural attribution is not morphology attribution",
        "Both scanners share the original graph topology",
        "Node degree may influence attribution rankings",
        "Top-k fidelity is in-sample explanatory fidelity",
        "Incident-edge deletion can disconnect graph nodes",
        "Mean score change can include indirect message effects",
        "Validation regions share biological material",
        "No independent scanner or patient validation",
        "No claim of clinical explanation validity",
    ],
    "next_xai_methods": [
        "Integrated Gradients",
        "GNNExplainer",
        "Feature perturbation fidelity",
    ],
}

save_json_new(REPORT, report)

print("\n" + "=" * 70)
print("PHASE 4.2.20 - STRUCTURAL XAI COMPLETED")
print("=" * 70)

for item in model_summary:
    print(
        f"{item['model']} | "
        f"Mean rank correlation "
        f"{item['mean_attribution_rank_correlation']:.4f} | "
        f"Top-5 overlap "
        f"{item['mean_top5_overlap']:.2f}/5"
    )

print("\nSaved:")
print(DETAIL)
print(SUMMARY)
print(REPORT)

print("\nTraining performed: NO")
print("Existing checkpoints modified: NO")
print("Held-out test evaluated: NO")
print("XAI stage: STARTED")
print("\nSTEP 4.2.20 COMPLETED")
