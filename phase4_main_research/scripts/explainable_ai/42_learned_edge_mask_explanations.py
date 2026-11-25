
"""
PHASE 4.2.27 - LEARNED EDGE-MASK EXPLANATIONS

Custom GNNExplainer-style structural attribution.

For each model, seed, region and scanner:
    Optimize 60 undirected edge-mask logits.
    Freeze pretrained GraphSAGE weights.
    Keep the opposite scanner's embeddings fixed.
    Optimize negative bidirectional InfoNCE
    with a fixed sparsity penalty.

Evaluate:
    Learned mask sparsity and fidelity
    AT2/GT450 edge-mask agreement
    Top-12 edge deletion against random controls
    Preservation of original checkpoint retrieval

This is exploratory validation-only XAI.
No model training, checkpoint selection or test use.
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


BASE = Path.home() / "MEDICAL_GNN_RESEARCH" / "phase4_main_research"

GRAPH = BASE / "outputs/features/15_stage_a_spatial_graphs.npz"
GRAPH_REPORT = BASE / "outputs/reports/15_graph_dataset_report.json"
PROTOCOL = BASE / "configs/16_initial_gnn_experimental_protocol.json"
ARCH = BASE / "configs/25_stability_aware_gnn_architecture_v1.json"
IG_REPORT = BASE / "outputs/reports/41_corrected_ig_audit_report.json"

SOURCES = {
    "ResidualGate05": {
        "report": BASE / "outputs/reports/28_gate_init05_report.json",
        "directory": BASE / "outputs/checkpoints/28_residual_gate_init05",
        "filename": "gate_init05_seed_{seed}.pt",
    },
    "TopologyConsistentResidual": {
        "report": BASE / "outputs/reports/31_topology_consistent_report.json",
        "directory": BASE / "outputs/checkpoints/31_topology_consistent_residual",
        "filename": "topology_consistent_seed_{seed}.pt",
    },
}

EDGE_OUTPUT = BASE / "outputs/xai/42_learned_edge_masks.csv"
DETAIL_OUTPUT = BASE / "outputs/xai/42_edge_mask_explanation_detail.csv"
SUMMARY_OUTPUT = BASE / "outputs/xai/42_edge_mask_summary.csv"
REPORT_OUTPUT = BASE / "outputs/reports/42_learned_edge_mask_report.json"

SEEDS = (42, 123, 2026)
OPTIMIZATION_STEPS = 120
MASK_LR = 0.05
MASK_INITIAL_LOGIT = 2.0
SPARSITY_WEIGHT = 0.03
TOP_K = 12
RANDOM_CONTROLS = 20
TEMPERATURE = 0.07

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


def write_csv_new(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json_new(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as stream:
        json.dump(content, stream, indent=2)


def stable_rng(*items):
    content = "|".join(map(str, items)).encode("utf-8")
    digest = hashlib.sha256(content).digest()

    return np.random.default_rng(
        int.from_bytes(digest[:8], "little")
    )


def build_edges(edge_index):
    if edge_index.shape != (2, 120):
        raise RuntimeError("Unexpected edge index.")

    directed = set()

    for u, v in edge_index.T.tolist():
        u, v = int(u), int(v)

        if (
            not 0 <= u < 36
            or not 0 <= v < 36
            or u == v
            or (u, v) in directed
        ):
            raise RuntimeError("Invalid edge.")

        directed.add((u, v))

    if len(directed) != 120:
        raise RuntimeError("Expected 120 directed edges.")

    if any((v, u) not in directed for u, v in directed):
        raise RuntimeError("Missing reciprocal edges.")

    undirected = sorted(
        (u, v)
        for u, v in directed
        if u < v
    )

    if len(undirected) != 60:
        raise RuntimeError("Expected 60 undirected edges.")

    adjacency = torch.zeros(36, 36, dtype=torch.float32)

    for u, v in undirected:
        adjacency[u, v] = 1
        adjacency[v, u] = 1

    if not torch.equal(adjacency, adjacency.T):
        raise RuntimeError("Asymmetric graph.")

    degree = adjacency.sum(-1, keepdim=True)

    if torch.any(degree <= 0):
        raise RuntimeError("Unexpected isolated node.")

    weights = adjacency / degree

    return undirected, weights


def weights_from_edge_mask(edges, mask, device):
    """
    Differentiable symmetric edge weighting.

    Row-normalization allows edge importance
    to affect neighborhood aggregation.

    Zero-degree rows receive zero messages.
    """
    adjacency = torch.zeros(
        36, 36, dtype=mask.dtype, device=device
    )

    for index, (u, v) in enumerate(edges):
        adjacency[u, v] = mask[index]
        adjacency[v, u] = mask[index]

    degree = adjacency.sum(dim=-1, keepdim=True)

    return adjacency / degree.clamp_min(1e-12)


class SAGEConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.neighbor_linear = nn.Linear(
            input_dim, output_dim, bias=False
        )

    def forward(self, x, weights):
        return (
            self.self_linear(x)
            + self.neighbor_linear(weights @ x)
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

        return F.normalize(
            identity + torch.sigmoid(self.gate_logit) * graph,
            dim=-1,
        )


def objective(a, b):
    logits = (a @ b.T) / TEMPERATURE
    targets = torch.arange(36, device=logits.device)

    return -0.5 * (
        F.cross_entropy(logits, targets)
        + F.cross_entropy(logits.T, targets)
    )


def retrieval(a, b):
    similarity = a @ b.T

    targets = torch.arange(
        len(a), device=a.device
    )

    return float(
        (
            similarity.argmax(-1) == targets
        ).float().mean().item()
    )


def evaluate_scanner(
    model, scanner, features, opponent, weights
):
    embeddings = model(features, override=weights)

    if scanner == "AT2":
        score = objective(embeddings, opponent)
    else:
        score = objective(opponent, embeddings)

    return score


def ranks(x):
    values = np.asarray(x, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)

    sorted_values = values[order]
    start = 0

    while start < len(values):
        end = start + 1

        while (
            end < len(values)
            and sorted_values[end] == sorted_values[start]
        ):
            end += 1

        result[order[start:end]] = (
            start + 1 + end
        ) / 2.0

        start = end

    return result


def spearman(x, y):
    a, b = ranks(x), ranks(y)
    a -= a.mean()
    b -= b.mean()

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator <= 1e-12:
        return None

    return float(np.dot(a, b) / denominator)


def top_indices(values):
    return np.argsort(
        -np.asarray(values),
        kind="stable"
    )[:TOP_K]


@torch.no_grad()
def deletion_drop(
    model, scanner, features, opponent,
    edges, removed_indices, baseline_score
):
    mask = torch.ones(
        len(edges), device=DEVICE
    )

    mask[removed_indices] = 0.0

    weights = weights_from_edge_mask(
        edges, mask, DEVICE
    )

    new_score = evaluate_scanner(
        model,
        scanner,
        features,
        opponent,
        weights,
    )

    return baseline_score - float(new_score.item())


def optimize_mask(
    model, scanner, features, opponent, edges,
    original_weights
):
    """
    Explanations are optimized independently for
    AT2 and GT450.

    Opposite scanner embeddings are frozen.

    Mask size: 60 continuous undirected edges.
    """

    edge_logits = nn.Parameter(
        torch.full(
            (len(edges),),
            MASK_INITIAL_LOGIT,
            device=DEVICE,
            dtype=torch.float32,
        )
    )

    optimizer = torch.optim.Adam(
        [edge_logits], lr=MASK_LR
    )

    with torch.no_grad():
        baseline_score = float(
            evaluate_scanner(
                model, scanner, features,
                opponent, original_weights
            ).item()
        )

    for _ in range(OPTIMIZATION_STEPS):
        optimizer.zero_grad(set_to_none=True)

        mask = torch.sigmoid(edge_logits)

        weights = weights_from_edge_mask(
            edges, mask, DEVICE
        )

        score = evaluate_scanner(
            model, scanner, features,
            opponent, weights
        )

        loss = (
            -score
            + SPARSITY_WEIGHT * mask.mean()
        )

        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite explainer loss.")

        loss.backward()

        if edge_logits.grad is None:
            raise RuntimeError("Edge mask received no gradients.")

        optimizer.step()

    with torch.no_grad():
        learned = torch.sigmoid(
            edge_logits
        ).cpu().numpy().astype(np.float64)

        weights = weights_from_edge_mask(
            edges,
            torch.from_numpy(
                learned.astype(np.float32)
            ).to(DEVICE),
            DEVICE,
        )

        masked_score = float(
            evaluate_scanner(
                model, scanner, features,
                opponent, weights
            ).item()
        )

    if not np.isfinite(learned).all():
        raise RuntimeError("Invalid learned edge mask.")

    return {
        "mask": learned,
        "original_score": baseline_score,
        "masked_score": masked_score,
        "mask_mean": float(learned.mean()),
        "mask_fraction_above_half": float(
            (learned > 0.5).mean()
        ),
    }


print("=" * 70, flush=True)
print("PHASE 4.2.27 - LEARNED EDGE-MASK XAI", flush=True)
print("=" * 70, flush=True)

# --------------------------------------------------
# 1. Provenance
# --------------------------------------------------

print("\n[1] Verifying source experiments...", flush=True)

required = [
    GRAPH, GRAPH_REPORT, PROTOCOL, ARCH, IG_REPORT,
    *[value["report"] for value in SOURCES.values()],
]

for path in required:
    if not path.is_file():
        raise RuntimeError(f"Missing source: {path}")

for path in (
    EDGE_OUTPUT, DETAIL_OUTPUT,
    SUMMARY_OUTPUT, REPORT_OUTPUT
):
    if path.exists():
        raise RuntimeError(f"Refusing overwrite: {path}")

graph_hash = sha256(GRAPH)
protocol_hash = sha256(PROTOCOL)
architecture_hash = sha256(ARCH)

if (
    read_json(GRAPH_REPORT)["graph_archive_sha256"]
    != graph_hash
):
    raise RuntimeError("Graph integrity mismatch.")

if (
    read_json(PROTOCOL)["source_graph_sha256"]
    != graph_hash
):
    raise RuntimeError("Protocol integrity mismatch.")

architecture = read_json(ARCH)

if (
    architecture["graph_archive_sha256"] != graph_hash
    or architecture["initial_protocol_sha256"]
    != protocol_hash
):
    raise RuntimeError("Architecture provenance mismatch.")

previous_xai = read_json(IG_REPORT)

if (
    previous_xai["graph_archive_sha256"] != graph_hash
    or previous_xai["protocol_sha256"] != protocol_hash
    or previous_xai["test_evaluated"] is not False
):
    raise RuntimeError("Prior XAI provenance mismatch.")

references = {}

for name, spec in SOURCES.items():
    reference = read_json(spec["report"])

    if (
        reference["graph_archive_sha256"] != graph_hash
        or reference["protocol_sha256"] != protocol_hash
        or reference["parent_architecture_sha256"]
        != architecture_hash
        or reference["test_evaluated"] is not False
    ):
        raise RuntimeError(f"Model provenance mismatch: {name}")

    references[name] = reference


# --------------------------------------------------
# 2. Load paired validation regions
# --------------------------------------------------

print("\n[2] Preparing validation-only graphs...", flush=True)

with np.load(GRAPH, allow_pickle=False) as data:
    features = data["node_features"].copy()
    edges_raw = data["edge_index"].copy()
    regions = data["region_id"].astype(str)
    splits = data["spatial_split"].astype(str)
    scanners = data["scanner"].astype(str)
    tiles = data["tile_id"].astype(str)

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
    raise RuntimeError("Spatial partitions invalid.")

by_region = defaultdict(dict)

for index, region in enumerate(regions):
    if splits[index] != "validation":
        continue

    scanner = scanners[index]

    if scanner in by_region[region]:
        raise RuntimeError("Duplicate validation scanner.")

    by_region[region][scanner] = index

if len(by_region) != 10:
    raise RuntimeError("Expected 10 validation regions.")

pairs = {}

for region, indices in sorted(by_region.items()):
    if set(indices) != {"AT2", "GT450"}:
        raise RuntimeError("Incomplete scanner region.")

    a, b = indices["AT2"], indices["GT450"]

    if not np.array_equal(tiles[a], tiles[b]):
        raise RuntimeError("Node correspondence mismatch.")

    pairs[region] = (
        torch.from_numpy(features[a]).to(DEVICE),
        torch.from_numpy(features[b]).to(DEVICE),
    )

edges, original_weights = build_edges(edges_raw)
original_weights = original_weights.to(DEVICE)

print("Validation region pairs:", len(pairs))
print("Undirected edges:", len(edges))
print("Mask optimization steps:", OPTIMIZATION_STEPS)
print("Device:", DEVICE)
print("Test: RESERVED")


# --------------------------------------------------
# 3. Optimize explanations
# --------------------------------------------------

print("\n[3] Optimizing edge explanations...", flush=True)

edge_rows = []
detail_rows = []

for model_name, spec in SOURCES.items():
    reference = references[model_name]

    for seed in SEEDS:
        checkpoint = (
            spec["directory"]
            / spec["filename"].format(seed=seed)
        )

        if (
            not checkpoint.is_file()
            or sha256(checkpoint)
            != reference["checkpoint_hashes"][str(seed)]
        ):
            raise RuntimeError(
                f"Checkpoint integrity error: {checkpoint}"
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
            or payload["parent_architecture_sha256"]
            != architecture_hash
            or payload["initial_gate"] != 0.5
            or payload["test_evaluated"] is not False
        ):
            raise RuntimeError("Checkpoint metadata mismatch.")

        if model_name == "TopologyConsistentResidual":
            if payload["lambda_topology"] != 1.0:
                raise RuntimeError("Incorrect consistency weight.")

        model = ResidualGraphSAGE(
            original_weights
        ).to(DEVICE)

        model.load_state_dict(
            payload["model_state_dict"],
            strict=True,
        )

        model.eval()

        for parameter in model.parameters():
            parameter.requires_grad_(False)

        recalls = []

        for region, (a, b) in pairs.items():
            with torch.no_grad():
                emb_a = model(a)
                emb_b = model(b)

                recalls.append(retrieval(emb_a, emb_b))

            profiles = {}

            for scanner in ("AT2", "GT450"):
                x = a if scanner == "AT2" else b

                opponent = (
                    emb_b.detach()
                    if scanner == "AT2"
                    else emb_a.detach()
                )

                result = optimize_mask(
                    model,
                    scanner,
                    x,
                    opponent,
                    edges,
                    original_weights,
                )

                mask = result["mask"]

                top = top_indices(mask)

                top_drop = deletion_drop(
                    model,
                    scanner,
                    x,
                    opponent,
                    edges,
                    top,
                    result["original_score"],
                )

                rng = stable_rng(
                    "phase4227", seed,
                    region, scanner
                )

                random_drops = []

                for _ in range(RANDOM_CONTROLS):
                    random_nodes = rng.choice(
                        len(edges),
                        size=TOP_K,
                        replace=False,
                    )

                    random_drops.append(
                        deletion_drop(
                            model,
                            scanner,
                            x,
                            opponent,
                            edges,
                            random_nodes,
                            result["original_score"],
                        )
                    )

                random_drop = float(
                    np.mean(random_drops)
                )

                profiles[scanner] = {
                    "mask": mask,
                    "top": top,
                    "original_score": result["original_score"],
                    "masked_score": result["masked_score"],
                    "mask_mean": result["mask_mean"],
                    "fraction_above_half":
                        result["mask_fraction_above_half"],
                    "top12_score_drop": top_drop,
                    "random_score_drop": random_drop,
                    "fidelity_excess":
                        top_drop - random_drop,
                }

                for edge_index, (u, v) in enumerate(edges):
                    edge_rows.append({
                        "model": model_name,
                        "seed": seed,
                        "region_id": region,
                        "scanner": scanner,
                        "edge_index": edge_index,
                        "source_node": u,
                        "target_node": v,
                        "mask_weight": float(
                            mask[edge_index]
                        ),
                    })

            pa = profiles["AT2"]
            pb = profiles["GT450"]

            correlation = spearman(
                pa["mask"],
                pb["mask"],
            )

            top_overlap = len(
                set(pa["top"].tolist())
                & set(pb["top"].tolist())
            )

            detail_rows.append({
                "model": model_name,
                "seed": seed,
                "region_id": region,
                "edge_mask_spearman": correlation,
                "top12_overlap": top_overlap,
                "at2_mask_mean": pa["mask_mean"],
                "gt450_mask_mean": pb["mask_mean"],
                "at2_fraction_above_half":
                    pa["fraction_above_half"],
                "gt450_fraction_above_half":
                    pb["fraction_above_half"],
                "at2_objective_change":
                    pa["masked_score"] - pa["original_score"],
                "gt450_objective_change":
                    pb["masked_score"] - pb["original_score"],
                "at2_top12_score_drop":
                    pa["top12_score_drop"],
                "at2_random_score_drop":
                    pa["random_score_drop"],
                "at2_fidelity_excess":
                    pa["fidelity_excess"],
                "gt450_top12_score_drop":
                    pb["top12_score_drop"],
                "gt450_random_score_drop":
                    pb["random_score_drop"],
                "gt450_fidelity_excess":
                    pb["fidelity_excess"],
            })

        historical_recall = next(
            r["validation_recall_at_1"]
            for r in reference["seed_results"]
            if int(r["seed"]) == seed
        )

        current_recall = float(np.mean(recalls))

        if abs(
            current_recall - float(historical_recall)
        ) > 1e-5:
            raise RuntimeError(
                "Frozen-model retrieval reproduction failed."
            )

        print(
            f"  {model_name}, seed {seed} | "
            f"Normal R@1 {current_recall:.4f} | "
            "Edge masks complete",
            flush=True,
        )


# --------------------------------------------------
# 4. Summarize explanations
# --------------------------------------------------

print("\n[4] Aggregating learned-mask metrics...", flush=True)

metrics = [
    "edge_mask_spearman",
    "top12_overlap",
    "at2_mask_mean",
    "gt450_mask_mean",
    "at2_fraction_above_half",
    "gt450_fraction_above_half",
    "at2_objective_change",
    "gt450_objective_change",
    "at2_fidelity_excess",
    "gt450_fidelity_excess",
]

summary_rows = []

for model_name in SOURCES:
    subset = [
        row for row in detail_rows
        if row["model"] == model_name
    ]

    if len(subset) != 30:
        raise RuntimeError("Expected 30 region-seed explanations.")

    summary = {
        "model": model_name,
        "region_seed_pairs": len(subset),
    }

    for metric in metrics:
        values = [
            float(r[metric])
            for r in subset
            if r[metric] is not None
        ]

        summary[f"mean_{metric}"] = (
            float(np.mean(values))
            if values else None
        )

    summary_rows.append(summary)

    print(
        f"  {model_name} | "
        f"Edge rho {summary['mean_edge_mask_spearman']:.4f} | "
        f"Top-12 overlap "
        f"{summary['mean_top12_overlap']:.2f}/12 | "
        f"AT2 fidelity "
        f"{summary['mean_at2_fidelity_excess']:+.4f}",
        flush=True,
    )


# --------------------------------------------------
# 5. Save outputs
# --------------------------------------------------

print("\n[5] Saving edge-mask explanations...", flush=True)

edge_columns = [
    "model", "seed", "region_id",
    "scanner", "edge_index",
    "source_node", "target_node",
    "mask_weight",
]

detail_columns = [
    "model", "seed", "region_id",
    "edge_mask_spearman",
    "top12_overlap",
    "at2_mask_mean",
    "gt450_mask_mean",
    "at2_fraction_above_half",
    "gt450_fraction_above_half",
    "at2_objective_change",
    "gt450_objective_change",
    "at2_top12_score_drop",
    "at2_random_score_drop",
    "at2_fidelity_excess",
    "gt450_top12_score_drop",
    "gt450_random_score_drop",
    "gt450_fidelity_excess",
]

summary_columns = (
    ["model", "region_seed_pairs"]
    + [f"mean_{name}" for name in metrics]
)

write_csv_new(
    EDGE_OUTPUT, edge_rows, edge_columns
)

write_csv_new(
    DETAIL_OUTPUT, detail_rows, detail_columns
)

write_csv_new(
    SUMMARY_OUTPUT, summary_rows, summary_columns
)

report = {
    "phase": "4.2",
    "step": "4.2.27",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "experiment": "learned_undirected_edge_mask_explanation",
    "method": "custom_GNNExplainer_style_edge_mask",
    "graph_archive_sha256": graph_hash,
    "protocol_sha256": protocol_hash,
    "architecture_sha256": architecture_hash,
    "source_model_report_hashes": {
        name: sha256(spec["report"])
        for name, spec in SOURCES.items()
    },
    "source_ig_audit_sha256": sha256(IG_REPORT),
    "validation_regions": len(pairs),
    "seeds": list(SEEDS),
    "undirected_edges": len(edges),
    "optimization_steps": OPTIMIZATION_STEPS,
    "mask_learning_rate": MASK_LR,
    "initial_mask_logit": MASK_INITIAL_LOGIT,
    "sparsity_weight": SPARSITY_WEIGHT,
    "retrieval_objective":
        "negative_bidirectional_within_region_InfoNCE",
    "top_k": TOP_K,
    "random_deletion_controls": RANDOM_CONTROLS,
    "model_summary": summary_rows,
    "edge_table_sha256": sha256(EDGE_OUTPUT),
    "detail_table_sha256": sha256(DETAIL_OUTPUT),
    "summary_table_sha256": sha256(SUMMARY_OUTPUT),
    "model_training_performed": False,
    "explainer_optimization_performed": True,
    "checkpoint_selection_performed": False,
    "test_evaluated": False,
    "limitations": [
        "Edge masks are optimized on validation instances",
        "Explainer fidelity uses the optimization objective",
        "Random edge-deletion controls do not match edge degree types",
        "Low mask weights do not imply unimportant biological tissue",
        "Masks use row-normalized neighbor aggregation",
        "Weak sparsity may retain most graph edges",
        "Edge explanations are not pathologist annotations",
        "Both scanners share the same original topology",
        "Ten validation regions are not independent patients",
        "No real scanner-shift generalization tested",
    ],
    "next_steps": [
        "Audit mask sparsity and optimization behavior",
        "Compare model explanations by spatial region",
        "Introduce degree-aware random edge controls",
        "Assess explanation fidelity and consistency jointly",
    ],
}

write_json_new(REPORT_OUTPUT, report)

print("\n" + "=" * 70)
print("PHASE 4.2.27 - LEARNED EDGE-MASK XAI COMPLETED")
print("=" * 70)

for row in summary_rows:
    print(
        f"{row['model']} | "
        f"Mask correlation "
        f"{row['mean_edge_mask_spearman']:.4f} | "
        f"Top-12 overlap "
        f"{row['mean_top12_overlap']:.2f} | "
        f"AT2 fidelity excess "
        f"{row['mean_at2_fidelity_excess']:+.4f}"
    )

print("\nSaved:")
print(EDGE_OUTPUT)
print(DETAIL_OUTPUT)
print(SUMMARY_OUTPUT)
print(REPORT_OUTPUT)

print("\nModel training: NO")
print("Existing checkpoints: UNCHANGED")
print("Held-out test: NOT EVALUATED")
print("XAI: IN PROGRESS")
print("\nSTEP 4.2.27 COMPLETED")
