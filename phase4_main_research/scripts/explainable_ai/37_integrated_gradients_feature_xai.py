
"""
PHASE 4.2.22 - INTEGRATED GRADIENTS FEATURE XAI

Explain DINOv2 feature inputs to two saved GNNs.

For each model / seed / validation region / scanner:
  - Integrated Gradients with 32 midpoint steps
  - Negative bidirectional InfoNCE attribution target
  - Opposite scanner embeddings fixed
  - Zero-feature baseline
  - Node importance = sum(abs(IG features))
  - Feature-channel importance = sum(abs(IG nodes))
  - Completeness residual
  - Top-5 deletion versus 20 degree-matched
    random node sets

Across AT2 and GT450:
  - Node-importance Spearman correlation
  - Node top-5 overlap
  - Feature-channel attribution cosine

No training.
No test evaluation.
No checkpoint selection.
No historical artifacts overwritten.
"""

from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
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
STRUCTURAL_REPORT = BASE / "outputs/reports/35_structural_xai_report.json"
DEGREE_REPORT = BASE / "outputs/reports/36_degree_controlled_xai_report.json"

SOURCES = {
    "ResidualGate05": {
        "report": BASE / "outputs/reports/28_gate_init05_report.json",
        "folder": BASE / "outputs/checkpoints/28_residual_gate_init05",
        "filename": "gate_init05_seed_{seed}.pt",
        "lambda_topology": 0.0,
    },
    "TopologyConsistentResidual": {
        "report": BASE / "outputs/reports/31_topology_consistent_report.json",
        "folder": BASE / "outputs/checkpoints/31_topology_consistent_residual",
        "filename": "topology_consistent_seed_{seed}.pt",
        "lambda_topology": 1.0,
    },
}

NODE_TABLE = BASE / "outputs/xai/37_ig_node_attributions.csv"
CHANNEL_TABLE = BASE / "outputs/xai/37_ig_feature_channels.csv"
SUMMARY_TABLE = BASE / "outputs/xai/37_ig_explanation_summary.csv"
REPORT = BASE / "outputs/reports/37_integrated_gradients_report.json"

SEEDS = (42, 123, 2026)
STEPS = 32
TOP_K = 5
RANDOM_CONTROLS = 20
TEMPERATURE = 0.07

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


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


def deterministic_rng(*parts):
    raw = "|".join(map(str, parts)).encode("utf-8")
    h = hashlib.sha256(raw).digest()
    seed = int.from_bytes(h[:8], "little")
    return np.random.default_rng(seed)


def adjacency_from_edges(edges):
    if edges.shape != (2, 120):
        raise RuntimeError("Unexpected edge index.")

    adjacency = np.zeros((36, 36), dtype=np.float32)
    seen = set()

    for source, target in edges.T.tolist():
        source, target = int(source), int(target)

        if (
            not 0 <= source < 36
            or not 0 <= target < 36
            or source == target
            or (source, target) in seen
        ):
            raise RuntimeError("Invalid graph edge.")

        seen.add((source, target))
        adjacency[target, source] = 1.0

    if len(seen) != 120:
        raise RuntimeError("Expected 120 directed edges.")

    if any((v, u) not in seen for u, v in seen):
        raise RuntimeError("Missing reciprocal edge.")

    if not np.array_equal(adjacency, adjacency.T):
        raise RuntimeError("Asymmetric graph.")

    return adjacency


def normalize_adjacency(a):
    degrees = a.sum(axis=1, keepdims=True)

    if np.any(degrees <= 0):
        raise RuntimeError("Isolated node.")

    return (a / degrees).astype(np.float32)


class SAGEConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neighbor_linear = nn.Linear(
            in_dim, out_dim, bias=False
        )

    def forward(self, x, adjacency):
        return (
            self.self_linear(x)
            + self.neighbor_linear(
                torch.matmul(adjacency, x)
            )
        )


class ResidualGraphSAGE(nn.Module):
    def __init__(self, adjacency):
        super().__init__()

        self.register_buffer(
            "neighbor_weights", adjacency.clone()
        )

        self.conv1 = SAGEConv(384, 256)
        self.conv2 = SAGEConv(256, 384)
        self.gate_logit = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        identity = F.normalize(x, dim=-1)

        h = F.gelu(
            self.conv1(x, self.neighbor_weights)
        )

        graph = F.normalize(
            self.conv2(h, self.neighbor_weights),
            dim=-1,
        )

        gate = torch.sigmoid(self.gate_logit)

        return F.normalize(
            identity + gate * graph,
            dim=-1,
        )


def retrieval_score(a, b):
    """
    Negative bidirectional within-region InfoNCE.

    Higher means a better objective score.
    """
    logits = (a @ b.T) / TEMPERATURE

    targets = torch.arange(
        len(a), device=a.device
    )

    return -0.5 * (
        F.cross_entropy(logits, targets)
        + F.cross_entropy(logits.T, targets)
    )


def retrieval_recall(a, b):
    target = torch.arange(
        len(a), device=a.device
    )

    similarity = a @ b.T

    return float(
        (similarity.argmax(-1) == target)
        .float().mean().item()
    )


def midpoint_integrated_gradients(
    model, original_features, opposite_embeddings,
    scanner
):
    """
    Numerically approximate IG with the midpoint rule.

    The opposite scanner's embeddings are fixed,
    while all 36x384 features of the explained
    scanner vary from zero to their true values.
    """
    baseline = torch.zeros_like(original_features)
    displacement = original_features - baseline

    gradient_sum = torch.zeros_like(
        original_features
    )

    for step in range(STEPS):
        alpha = (step + 0.5) / STEPS

        interpolated = (
            baseline + alpha * displacement
        ).detach().requires_grad_(True)

        explained = model(interpolated)

        if scanner == "AT2":
            score = retrieval_score(
                explained, opposite_embeddings
            )
        else:
            score = retrieval_score(
                opposite_embeddings, explained
            )

        gradient = torch.autograd.grad(
            score,
            interpolated,
            retain_graph=False,
            create_graph=False,
        )[0]

        gradient_sum += gradient.detach()

    ig = displacement * (gradient_sum / STEPS)

    with torch.no_grad():
        final_output = model(original_features)
        baseline_output = model(baseline)

        if scanner == "AT2":
            actual_score = retrieval_score(
                final_output, opposite_embeddings
            )
            baseline_score = retrieval_score(
                baseline_output, opposite_embeddings
            )
        else:
            actual_score = retrieval_score(
                opposite_embeddings, final_output
            )
            baseline_score = retrieval_score(
                opposite_embeddings, baseline_output
            )

        actual_delta = float(
            (actual_score - baseline_score).item()
        )

    attributed_delta = float(ig.sum().item())

    completeness_error = abs(
        actual_delta - attributed_delta
    )

    if not torch.isfinite(ig).all():
        raise RuntimeError(
            "Non-finite integrated gradients."
        )

    return {
        "ig": ig.detach(),
        "score_delta": actual_delta,
        "ig_sum": attributed_delta,
        "completeness_abs_error": completeness_error,
        "completeness_relative_error":
            completeness_error / max(abs(actual_delta), 1e-6),
    }


def ranks(values):
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    output = np.empty(len(x), dtype=np.float64)

    sorted_x = x[order]
    start = 0

    while start < len(x):
        end = start + 1

        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1

        output[order[start:end]] = (
            start + 1 + end
        ) / 2.0

        start = end

    return output


def spearman(a, b):
    ra = ranks(a)
    rb = ranks(b)

    ra = ra - ra.mean()
    rb = rb - rb.mean()

    denominator = np.linalg.norm(ra) * np.linalg.norm(rb)

    if denominator <= 1e-12:
        return None

    return float(np.dot(ra, rb) / denominator)


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator <= 1e-12:
        return None

    return float(np.dot(a, b) / denominator)


def matched_degree_random_set(
    ranked_nodes, degrees, rng
):
    """
    Select a random 5-node set that has the same
    counts of degree-2/3/4 nodes as the top-5 set.
    """
    selected = []

    for degree in sorted(set(degrees)):
        count = int(
            np.sum(degrees[ranked_nodes] == degree)
        )

        if count == 0:
            continue

        candidates = np.flatnonzero(
            degrees == degree
        )

        chosen = rng.choice(
            candidates,
            size=count,
            replace=False,
        )

        selected.extend(chosen.tolist())

    if len(selected) != TOP_K:
        raise RuntimeError(
            "Degree-matched random set incomplete."
        )

    return np.asarray(selected, dtype=np.int64)


@torch.no_grad()
def deletion_score_drop(
    model, features, opposite_embeddings,
    scanner, selected, original_score
):
    changed = features.clone()
    changed[selected] = 0.0

    explained = model(changed)

    if scanner == "AT2":
        changed_score = retrieval_score(
            explained, opposite_embeddings
        )
    else:
        changed_score = retrieval_score(
            opposite_embeddings, explained
        )

    return original_score - float(changed_score.item())


print("=" * 70, flush=True)
print("PHASE 4.2.22 - INTEGRATED GRADIENTS FEATURE XAI", flush=True)
print("=" * 70, flush=True)


# --------------------------------------------------
# 1. Verify provenance
# --------------------------------------------------

print("\n[1] Verifying model and XAI sources...", flush=True)

required = [
    GRAPH,
    GRAPH_REPORT,
    PROTOCOL,
    ARCH,
    STRUCTURAL_REPORT,
    DEGREE_REPORT,
    *[item["report"] for item in SOURCES.values()],
]

for path in required:
    if not path.is_file():
        raise RuntimeError(f"Missing source: {path}")

for path in (
    NODE_TABLE, CHANNEL_TABLE, SUMMARY_TABLE, REPORT
):
    if path.exists():
        raise RuntimeError(
            f"Refusing to overwrite: {path}"
        )

graph_hash = sha256(GRAPH)
protocol_hash = sha256(PROTOCOL)
arch_hash = sha256(ARCH)

if load_json(GRAPH_REPORT)["graph_archive_sha256"] != graph_hash:
    raise RuntimeError("Graph integrity mismatch.")

if load_json(PROTOCOL)["source_graph_sha256"] != graph_hash:
    raise RuntimeError("Protocol mismatch.")

architecture = load_json(ARCH)

if (
    architecture["graph_archive_sha256"] != graph_hash
    or architecture["initial_protocol_sha256"] != protocol_hash
):
    raise RuntimeError("Architecture mismatch.")

for xai_path in (STRUCTURAL_REPORT, DEGREE_REPORT):
    prior_xai = load_json(xai_path)

    if (
        prior_xai["graph_archive_sha256"] != graph_hash
        or prior_xai["protocol_sha256"] != protocol_hash
        or prior_xai["test_evaluated"] is not False
    ):
        raise RuntimeError(
            f"XAI provenance mismatch: {xai_path}"
        )

references = {}

for name, spec in SOURCES.items():
    report = load_json(spec["report"])

    if (
        report["graph_archive_sha256"] != graph_hash
        or report["protocol_sha256"] != protocol_hash
        or report["parent_architecture_sha256"] != arch_hash
        or report["test_evaluated"] is not False
        or tuple(report["seeds"]) != SEEDS
    ):
        raise RuntimeError(
            f"Model provenance mismatch: {name}"
        )

    references[name] = report


# --------------------------------------------------
# 2. Load validation-only graph pairs
# --------------------------------------------------

print("\n[2] Preparing paired validation features...", flush=True)

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
    raise RuntimeError("Invalid graph feature tensor.")

if Counter(splits) != Counter({
    "train": 80,
    "validation": 20,
    "test": 20,
}):
    raise RuntimeError("Unexpected partitions.")

by_region = defaultdict(dict)

for i, region in enumerate(regions):
    if splits[i] != "validation":
        continue

    scanner = scanners[i]

    if scanner in by_region[region]:
        raise RuntimeError("Duplicate scanner graph.")

    by_region[region][scanner] = i

if len(by_region) != 10:
    raise RuntimeError(
        "Expected ten validation regions."
    )

pairs = {}

for region, indices in sorted(by_region.items()):
    if set(indices) != {"AT2", "GT450"}:
        raise RuntimeError("Incomplete scanner pair.")

    ai, bi = indices["AT2"], indices["GT450"]

    if not np.array_equal(
        tile_ids[ai], tile_ids[bi]
    ):
        raise RuntimeError("Node ordering mismatch.")

    pairs[region] = (
        torch.from_numpy(features[ai]).to(DEVICE),
        torch.from_numpy(features[bi]).to(DEVICE),
    )

adjacency = adjacency_from_edges(edges)

degree = adjacency.sum(axis=1).astype(np.int64)

if Counter(degree) != Counter({
    2: 4, 3: 16, 4: 16
}):
    raise RuntimeError("Unexpected grid degree pattern.")

weights = torch.from_numpy(
    normalize_adjacency(adjacency)
).to(DEVICE)

print("Validation region pairs:", len(pairs))
print("IG integration steps:", STEPS)
print("Random fidelity controls:", RANDOM_CONTROLS)
print("Test: RESERVED")


# --------------------------------------------------
# 3. Generate input-feature explanations
# --------------------------------------------------

print("\n[3] Computing Integrated Gradients...", flush=True)

node_rows = []
channel_rows = []
summary_rows = []

for model_name, spec in SOURCES.items():
    reference = references[model_name]

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
            != reference["checkpoint_hashes"][str(seed)]
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
            raise RuntimeError(
                "Checkpoint metadata mismatch."
            )

        if model_name == "TopologyConsistentResidual":
            if payload["lambda_topology"] != 1.0:
                raise RuntimeError(
                    "Unexpected topology consistency weight."
                )

        model = ResidualGraphSAGE(weights).to(DEVICE)

        model.load_state_dict(
            payload["model_state_dict"],
            strict=True,
        )

        model.eval()

        for parameter in model.parameters():
            parameter.requires_grad_(False)

        reproduced = []

        for region, (at2, gt450) in pairs.items():
            with torch.no_grad():
                original_a = model(at2)
                original_b = model(gt450)

                score_original = float(
                    retrieval_score(
                        original_a, original_b
                    ).item()
                )

                reproduced.append(
                    retrieval_recall(original_a, original_b)
                )

            scanner_profiles = {}

            for scanner in ("AT2", "GT450"):
                if scanner == "AT2":
                    inputs = at2
                    opponent = original_b.detach()
                else:
                    inputs = gt450
                    opponent = original_a.detach()

                result = midpoint_integrated_gradients(
                    model,
                    inputs,
                    opponent,
                    scanner,
                )

                importance = (
                    result["ig"].abs().sum(dim=-1)
                    .cpu().numpy().astype(np.float64)
                )

                channels = (
                    result["ig"].abs().sum(dim=0)
                    .cpu().numpy().astype(np.float64)
                )

                top_nodes = np.argsort(
                    -importance, kind="stable"
                )[:TOP_K]

                top_drop = deletion_score_drop(
                    model,
                    inputs,
                    opponent,
                    scanner,
                    top_nodes,
                    score_original,
                )

                rng = deterministic_rng(
                    "phase4222",
                    model_name,
                    seed,
                    region,
                    scanner,
                )

                random_drops = []

                for _ in range(RANDOM_CONTROLS):
                    chosen = matched_degree_random_set(
                        top_nodes, degree, rng
                    )

                    random_drops.append(
                        deletion_score_drop(
                            model,
                            inputs,
                            opponent,
                            scanner,
                            chosen,
                            score_original,
                        )
                    )

                random_mean = float(
                    np.mean(random_drops)
                )

                scanner_profiles[scanner] = {
                    "node_importance": importance,
                    "channel_importance": channels,
                    "completeness_abs_error":
                        result["completeness_abs_error"],
                    "completeness_relative_error":
                        result["completeness_relative_error"],
                    "top5_score_drop": top_drop,
                    "random_score_drop": random_mean,
                }

                for node in range(36):
                    node_rows.append({
                        "model": model_name,
                        "seed": seed,
                        "region_id": region,
                        "scanner": scanner,
                        "node_index": node,
                        "node_degree": int(degree[node]),
                        "ig_absolute_importance":
                            float(importance[node]),
                    })

                for channel in range(384):
                    channel_rows.append({
                        "model": model_name,
                        "seed": seed,
                        "region_id": region,
                        "scanner": scanner,
                        "feature_channel": channel,
                        "ig_absolute_importance":
                            float(channels[channel]),
                    })

            a_profile = scanner_profiles["AT2"]
            b_profile = scanner_profiles["GT450"]

            a_top = set(
                np.argsort(
                    -a_profile["node_importance"],
                    kind="stable",
                )[:TOP_K].tolist()
            )

            b_top = set(
                np.argsort(
                    -b_profile["node_importance"],
                    kind="stable",
                )[:TOP_K].tolist()
            )

            rank_agreement = spearman(
                a_profile["node_importance"],
                b_profile["node_importance"],
            )

            channel_agreement = cosine(
                a_profile["channel_importance"],
                b_profile["channel_importance"],
            )

            summary_rows.append({
                "model": model_name,
                "seed": seed,
                "region_id": region,
                "baseline_recall_at_1":
                    reproduced[-1],
                "node_spearman": rank_agreement,
                "node_top5_overlap":
                    len(a_top & b_top),
                "channel_importance_cosine":
                    channel_agreement,
                "at2_completeness_abs_error":
                    a_profile["completeness_abs_error"],
                "gt450_completeness_abs_error":
                    b_profile["completeness_abs_error"],
                "at2_completeness_relative_error":
                    a_profile["completeness_relative_error"],
                "gt450_completeness_relative_error":
                    b_profile["completeness_relative_error"],
                "at2_top5_score_drop":
                    a_profile["top5_score_drop"],
                "at2_random_score_drop":
                    a_profile["random_score_drop"],
                "gt450_top5_score_drop":
                    b_profile["top5_score_drop"],
                "gt450_random_score_drop":
                    b_profile["random_score_drop"],
                "at2_fidelity_excess":
                    a_profile["top5_score_drop"]
                    - a_profile["random_score_drop"],
                "gt450_fidelity_excess":
                    b_profile["top5_score_drop"]
                    - b_profile["random_score_drop"],
            })

        mean_reproduced = float(np.mean(reproduced))

        expected = next(
            item["validation_recall_at_1"]
            for item in reference["seed_results"]
            if int(item["seed"]) == seed
        )

        if abs(
            mean_reproduced - float(expected)
        ) > 1e-5:
            raise RuntimeError(
                f"Baseline retrieval mismatch: "
                f"{model_name}, seed {seed}"
            )

        print(
            f"  {model_name}, seed {seed} | "
            f"R@1 {mean_reproduced:.4f} | "
            f"IG completed",
            flush=True,
        )


# --------------------------------------------------
# 4. Aggregate XAI results
# --------------------------------------------------

print("\n[4] Aggregating feature attribution metrics...", flush=True)

numeric_metrics = [
    "node_spearman",
    "node_top5_overlap",
    "channel_importance_cosine",
    "at2_completeness_abs_error",
    "gt450_completeness_abs_error",
    "at2_completeness_relative_error",
    "gt450_completeness_relative_error",
    "at2_top5_score_drop",
    "at2_random_score_drop",
    "gt450_top5_score_drop",
    "gt450_random_score_drop",
    "at2_fidelity_excess",
    "gt450_fidelity_excess",
]

model_summary = []

for model_name in SOURCES:
    subset = [
        item for item in summary_rows
        if item["model"] == model_name
    ]

    if len(subset) != 30:
        raise RuntimeError(
            "Expected 30 model-seed-region summaries."
        )

    result = {
        "model": model_name,
        "region_seed_units": len(subset),
    }

    for metric in numeric_metrics:
        valid = [
            float(row[metric])
            for row in subset
            if row[metric] is not None
        ]

        result[f"mean_{metric}"] = (
            float(np.mean(valid))
            if valid else None
        )

    model_summary.append(result)

    print(
        f"  {model_name} | "
        f"Node rho "
        f"{result['mean_node_spearman']} | "
        f"Top-5 overlap "
        f"{result['mean_node_top5_overlap']:.2f} | "
        f"Feature cosine "
        f"{result['mean_channel_importance_cosine']}",
        flush=True,
    )


# --------------------------------------------------
# 5. Save research artifacts
# --------------------------------------------------

print("\n[5] Saving Integrated Gradients results...", flush=True)

save_csv_new(
    NODE_TABLE,
    node_rows,
    [
        "model", "seed", "region_id",
        "scanner", "node_index", "node_degree",
        "ig_absolute_importance",
    ],
)

save_csv_new(
    CHANNEL_TABLE,
    channel_rows,
    [
        "model", "seed", "region_id",
        "scanner", "feature_channel",
        "ig_absolute_importance",
    ],
)

summary_fields = [
    "model", "seed", "region_id",
    "baseline_recall_at_1",
    *numeric_metrics,
]

save_csv_new(
    SUMMARY_TABLE,
    summary_rows,
    summary_fields,
)

report = {
    "phase": "4.2",
    "step": "4.2.22",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "experiment":
        "integrated_gradients_input_feature_attribution",
    "method": "midpoint_integrated_gradients",
    "integration_steps": STEPS,
    "baseline": "all_zero_dinov2_features",
    "attribution_objective":
        "negative_bidirectional_within_region_InfoNCE",
    "opposite_scanner_embeddings_fixed": True,
    "top_k": TOP_K,
    "degree_matched_random_controls":
        RANDOM_CONTROLS,
    "graph_archive_sha256": graph_hash,
    "protocol_sha256": protocol_hash,
    "architecture_sha256": arch_hash,
    "source_model_reports_sha256": {
        name: sha256(spec["report"])
        for name, spec in SOURCES.items()
    },
    "source_structural_xai_report_sha256":
        sha256(STRUCTURAL_REPORT),
    "source_degree_xai_report_sha256":
        sha256(DEGREE_REPORT),
    "validation_regions": len(pairs),
    "seeds": list(SEEDS),
    "model_summary": model_summary,
    "node_table_sha256": sha256(NODE_TABLE),
    "channel_table_sha256": sha256(CHANNEL_TABLE),
    "summary_table_sha256": sha256(SUMMARY_TABLE),
    "training_performed": False,
    "checkpoint_selection_performed": False,
    "test_evaluated": False,
    "limitations": [
        "Input attribution targets DINOv2 features, not pixels",
        "Zero-feature baseline is out of distribution",
        "Feature deletion can perturb the frozen input distribution",
        "Attribution magnitude loses sign information",
        "Midpoint integration is approximate",
        "Completeness errors must be examined",
        "Random fidelity controls match degree only",
        "Top-k fidelity uses the same data as attribution",
        "No pathologist-provided explanation labels",
        "Validation regions are not independent patients",
    ],
    "next_steps": [
        "Inspect IG completeness and fidelity",
        "Compare feature attribution agreement",
        "Implement GNNExplainer with a suitable score target",
        "Perform paired region-level XAI comparisons",
    ],
}

save_json_new(REPORT, report)

print("\n" + "=" * 70)
print("PHASE 4.2.22 - INTEGRATED GRADIENTS COMPLETED")
print("=" * 70)

for item in model_summary:
    print(
        f"{item['model']} | "
        f"Node correlation "
        f"{item['mean_node_spearman']} | "
        f"Top-5 overlap "
        f"{item['mean_node_top5_overlap']:.2f} | "
        f"Channel cosine "
        f"{item['mean_channel_importance_cosine']}"
    )

print("\nSaved:")
print(NODE_TABLE)
print(CHANNEL_TABLE)
print(SUMMARY_TABLE)
print(REPORT)

print("\nTraining: NO")
print("Previous checkpoints: UNCHANGED")
print("Test partition: NOT EVALUATED")
print("XAI: IN PROGRESS")
print("\nSTEP 4.2.22 COMPLETED")
