
"""
PHASE 4.2.25 - CORRECTED INTEGRATED GRADIENTS

Uses the scanner-specific training-mean baseline
with 128 midpoint integration steps.

Two attribution rankings:
  1. Absolute node importance: sum(abs(IG))
  2. Signed node contribution: sum(IG)

Validation-only comparisons:
  Cross-scanner node attribution agreement
  Top-5 overlap
  Feature-channel attribution agreement
  Completeness against Phase 4.2.24
  Top-5 deletion vs degree-matched random deletion

No training.
No model selection.
No held-out test evaluation.
No existing research files overwritten.
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


ROOT = (
    Path.home()
    / "MEDICAL_GNN_RESEARCH"
    / "phase4_main_research"
)

GRAPH = ROOT / "outputs/features/15_stage_a_spatial_graphs.npz"
GRAPH_REPORT = ROOT / "outputs/reports/15_graph_dataset_report.json"
PROTOCOL = ROOT / "configs/16_initial_gnn_experimental_protocol.json"
ARCH = ROOT / "configs/25_stability_aware_gnn_architecture_v1.json"

CONVERGENCE_REPORT = ROOT / "outputs/reports/39_ig_convergence_report.json"
CONVERGENCE_DETAIL = ROOT / "outputs/xai/39_ig_convergence_detail.csv"

SOURCES = {
    "ResidualGate05": {
        "report": ROOT / "outputs/reports/28_gate_init05_report.json",
        "checkpoint_dir":
            ROOT / "outputs/checkpoints/28_residual_gate_init05",
        "filename": "gate_init05_seed_{seed}.pt",
    },
    "TopologyConsistentResidual": {
        "report": ROOT / "outputs/reports/31_topology_consistent_report.json",
        "checkpoint_dir":
            ROOT / "outputs/checkpoints/31_topology_consistent_residual",
        "filename": "topology_consistent_seed_{seed}.pt",
    },
}

NODE_OUT = ROOT / "outputs/xai/40_ig_node_importance.csv"
CHANNEL_OUT = ROOT / "outputs/xai/40_ig_channel_importance.csv"
DETAIL_OUT = ROOT / "outputs/xai/40_ig_corrected_detail.csv"
SUMMARY_OUT = ROOT / "outputs/xai/40_ig_corrected_summary.csv"
REPORT_OUT = ROOT / "outputs/reports/40_corrected_ig_report.json"

SEEDS = (42, 123, 2026)
SCANNERS = ("AT2", "GT450")
STEPS = 128
STEP_BATCH = 16
TOP_K = 5
RANDOM_CONTROLS = 20
TEMPERATURE = 0.07

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def hash_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_new(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json_new(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def stable_rng(*items):
    key = "|".join(map(str, items)).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return np.random.default_rng(
        int.from_bytes(digest[:8], "little")
    )


def make_adjacency(edges):
    if edges.shape != (2, 120):
        raise RuntimeError("Invalid edge-index dimensions.")

    adj = np.zeros((36, 36), dtype=np.float32)
    seen = set()

    for u, v in edges.T.tolist():
        u, v = int(u), int(v)

        if not 0 <= u < 36 or not 0 <= v < 36 or u == v:
            raise RuntimeError("Invalid graph edge.")

        if (u, v) in seen:
            raise RuntimeError("Duplicate edge.")

        seen.add((u, v))
        adj[v, u] = 1.0

    if len(seen) != 120 or not np.array_equal(adj, adj.T):
        raise RuntimeError("Invalid 6x6 graph.")

    degree = adj.sum(axis=1)
    if np.any(degree <= 0):
        raise RuntimeError("Isolated node.")

    weights = adj / degree[:, None]
    return weights.astype(np.float32), degree.astype(int)


class SAGEConv(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.self_linear = nn.Linear(n_in, n_out)
        self.neighbor_linear = nn.Linear(
            n_in, n_out, bias=False
        )

    def forward(self, x, weights):
        return (
            self.self_linear(x)
            + self.neighbor_linear(weights @ x)
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

    def forward(self, x):
        identity = F.normalize(x, dim=-1)
        h = F.gelu(self.conv1(x, self.neighbor_weights))
        graph = F.normalize(
            self.conv2(h, self.neighbor_weights),
            dim=-1,
        )
        return F.normalize(
            identity + torch.sigmoid(self.gate_logit) * graph,
            dim=-1,
        )


def score(a, b):
    logits = (a @ b.T) / TEMPERATURE
    target = torch.arange(36, device=logits.device)
    return -0.5 * (
        F.cross_entropy(logits, target)
        + F.cross_entropy(logits.T, target)
    )


def batched_score(a, opponent, scanner):
    count = a.shape[0]
    b = opponent.unsqueeze(0).expand(count, -1, -1)

    if scanner == "AT2":
        logits = torch.bmm(a, b.transpose(1, 2)) / TEMPERATURE
    else:
        logits = torch.bmm(b, a.transpose(1, 2)) / TEMPERATURE

    targets = torch.arange(
        36, device=DEVICE
    ).expand(count, 36).reshape(-1)

    forward = F.cross_entropy(
        logits.reshape(-1, 36), targets,
        reduction="none",
    ).reshape(count, 36).mean(1)

    reverse = F.cross_entropy(
        logits.transpose(1, 2).reshape(-1, 36),
        targets,
        reduction="none",
    ).reshape(count, 36).mean(1)

    return (-0.5 * (forward + reverse)).sum()


def evaluate_explained(model, x, opponent, scanner):
    z = model(x)
    return (
        score(z, opponent) if scanner == "AT2"
        else score(opponent, z)
    )


def integrated_gradients(model, x, baseline, opponent, scanner):
    diff = x - baseline
    grad_sum = torch.zeros_like(x)

    for start in range(0, STEPS, STEP_BATCH):
        stop = min(start + STEP_BATCH, STEPS)
        alphas = (
            torch.arange(
                start, stop, device=DEVICE, dtype=x.dtype
            ) + 0.5
        ) / STEPS

        points = (
            baseline.unsqueeze(0)
            + alphas[:, None, None] * diff.unsqueeze(0)
        ).detach().requires_grad_(True)

        output = model(points)
        objective = batched_score(output, opponent, scanner)
        grads = torch.autograd.grad(objective, points)[0]
        grad_sum += grads.detach().sum(dim=0)

    ig = diff * grad_sum / STEPS

    with torch.no_grad():
        actual = (
            evaluate_explained(model, x, opponent, scanner)
            - evaluate_explained(model, baseline, opponent, scanner)
        ).item()

    estimated = ig.sum().item()
    error = abs(actual - estimated)

    if not torch.isfinite(ig).all():
        raise RuntimeError("Nonfinite IG.")

    return ig.detach(), float(actual), float(estimated), float(error)


def avg_ranks(v):
    v = np.asarray(v, dtype=np.float64)
    order = np.argsort(v, kind="mergesort")
    out = np.empty(len(v), dtype=np.float64)
    sorted_v = v[order]

    start = 0
    while start < len(v):
        end = start + 1
        while end < len(v) and sorted_v[end] == sorted_v[start]:
            end += 1
        out[order[start:end]] = (start + 1 + end) / 2
        start = end

    return out


def correlation(x, y):
    a, b = avg_ranks(x), avg_ranks(y)
    a -= a.mean()
    b -= b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 1e-12 else None


def cosine(x, y):
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denom) if denom > 1e-12 else None


def top_indices(values):
    return np.argsort(-values, kind="stable")[:TOP_K]


def degree_matched_set(selected, degrees, rng):
    output = []
    for d in sorted(set(degrees.tolist())):
        n = int(np.sum(degrees[selected] == d))
        if n:
            candidates = np.where(degrees == d)[0]
            output.extend(
                rng.choice(candidates, size=n, replace=False).tolist()
            )
    if len(output) != TOP_K:
        raise RuntimeError("Invalid degree-matched control.")
    return np.asarray(output, dtype=np.int64)


@torch.no_grad()
def deletion_drop(model, x, reference, opponent, scanner,
                  selected, original_score):
    changed = x.clone()
    changed[selected] = reference[selected]
    return float(
        original_score
        - evaluate_explained(
            model, changed, opponent, scanner
        ).item()
    )


print("=" * 70, flush=True)
print("PHASE 4.2.25 - CORRECTED INTEGRATED GRADIENTS", flush=True)
print("=" * 70, flush=True)

print("\n[1] Verifying source provenance...", flush=True)

required = [
    GRAPH, GRAPH_REPORT, PROTOCOL, ARCH,
    CONVERGENCE_REPORT, CONVERGENCE_DETAIL,
    *[s["report"] for s in SOURCES.values()],
]

for path in required:
    if not path.is_file():
        raise RuntimeError(f"Missing source: {path}")

for path in (NODE_OUT, CHANNEL_OUT, DETAIL_OUT,
             SUMMARY_OUT, REPORT_OUT):
    if path.exists():
        raise RuntimeError(f"Refusing overwrite: {path}")

graph_hash = hash_file(GRAPH)
protocol_hash = hash_file(PROTOCOL)
arch_hash = hash_file(ARCH)

if read_json(GRAPH_REPORT)["graph_archive_sha256"] != graph_hash:
    raise RuntimeError("Graph hash mismatch.")

if read_json(PROTOCOL)["source_graph_sha256"] != graph_hash:
    raise RuntimeError("Protocol hash mismatch.")

architecture = read_json(ARCH)
if (
    architecture["graph_archive_sha256"] != graph_hash
    or architecture["initial_protocol_sha256"] != protocol_hash
):
    raise RuntimeError("Architecture hash mismatch.")

convergence = read_json(CONVERGENCE_REPORT)
if (
    convergence["graph_archive_sha256"] != graph_hash
    or convergence["detail_table_sha256"]
       != hash_file(CONVERGENCE_DETAIL)
    or convergence["test_evaluated"] is not False
):
    raise RuntimeError("Convergence source mismatch.")

references = {}
for name, spec in SOURCES.items():
    report = read_json(spec["report"])
    if (
        report["graph_archive_sha256"] != graph_hash
        or report["protocol_sha256"] != protocol_hash
        or report["parent_architecture_sha256"] != arch_hash
        or report["test_evaluated"] is not False
    ):
        raise RuntimeError(f"Checkpoint provenance mismatch: {name}")
    references[name] = report

convergence_index = {}
for r in read_csv(CONVERGENCE_DETAIL):
    if r["baseline"] != "training_mean" or int(r["steps"]) != STEPS:
        continue
    key = (
        r["model"], int(r["seed"]),
        r["region_id"], r["scanner"]
    )
    if key in convergence_index:
        raise RuntimeError("Duplicate convergence record.")
    convergence_index[key] = r


print("\n[2] Loading train and validation partitions...", flush=True)

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
    or Counter(splits) != Counter({
        "train": 80, "validation": 20, "test": 20
    })
):
    raise RuntimeError("Invalid graph archive or partitions.")

baseline = {}
for scanner in SCANNERS:
    mask = (splits == "train") & (scanners == scanner)
    if int(mask.sum()) != 40:
        raise RuntimeError("Incorrect training scanner count.")

    mean = features[mask].mean(
        axis=(0, 1), dtype=np.float64
    ).astype(np.float32)

    baseline[scanner] = torch.from_numpy(
        np.broadcast_to(mean, (36, 384)).copy()
    ).to(DEVICE)

by_region = defaultdict(dict)
for i, region in enumerate(regions):
    if splits[i] != "validation":
        continue
    if scanners[i] in by_region[region]:
        raise RuntimeError("Duplicate validation graph.")
    by_region[region][scanners[i]] = i

if len(by_region) != 10:
    raise RuntimeError("Expected ten validation regions.")

pairs = {}
for region, indices in sorted(by_region.items()):
    if set(indices) != set(SCANNERS):
        raise RuntimeError("Incomplete scanner pair.")

    ia, ib = indices["AT2"], indices["GT450"]
    if not np.array_equal(tile_ids[ia], tile_ids[ib]):
        raise RuntimeError("Node correspondence mismatch.")

    pairs[region] = (
        torch.from_numpy(features[ia]).to(DEVICE),
        torch.from_numpy(features[ib]).to(DEVICE),
    )

adj, degrees = make_adjacency(edges)
weights = torch.from_numpy(adj).to(DEVICE)

expected = {
    (model, seed, region, scanner)
    for model in SOURCES
    for seed in SEEDS
    for region in pairs
    for scanner in SCANNERS
}
if set(convergence_index) != expected:
    raise RuntimeError("Incomplete convergence reference.")

print("Validation regions:", len(pairs), flush=True)
print("Integration steps:", STEPS, flush=True)
print("Test partition: RESERVED", flush=True)


print("\n[3] Computing validated feature attributions...", flush=True)

node_rows = []
channel_rows = []
detail_rows = []

for model_name, spec in SOURCES.items():
    source = references[model_name]

    for seed in SEEDS:
        ckpt = (
            spec["checkpoint_dir"]
            / spec["filename"].format(seed=seed)
        )

        if not ckpt.is_file():
            raise RuntimeError(f"Missing checkpoint: {ckpt}")

        if hash_file(ckpt) != source["checkpoint_hashes"][str(seed)]:
            raise RuntimeError("Checkpoint integrity mismatch.")

        payload = torch.load(
            ckpt, map_location="cpu", weights_only=True
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
                raise RuntimeError("Consistency weight mismatch.")

        model = ResidualGraphSAGE(weights).to(DEVICE)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()

        for parameter in model.parameters():
            parameter.requires_grad_(False)

        recalls = []

        for region, (at2, gt450) in pairs.items():
            with torch.no_grad():
                emb_a = model(at2)
                emb_b = model(gt450)
                original_score = float(score(emb_a, emb_b).item())

                recalls.append(float(
                    ((emb_a @ emb_b.T).argmax(-1)
                     == torch.arange(36, device=DEVICE))
                    .float().mean().item()
                ))

            profiles = {}

            for scanner in SCANNERS:
                x = at2 if scanner == "AT2" else gt450
                opponent = (
                    emb_b.detach() if scanner == "AT2"
                    else emb_a.detach()
                )
                ref = baseline[scanner]

                ig, delta, ig_sum, error = integrated_gradients(
                    model, x, ref, opponent, scanner
                )

                prior = convergence_index[
                    (model_name, seed, region, scanner)
                ]
                if (
                    abs(error - float(prior["absolute_error"])) > 0.002
                    or abs(delta - float(prior["actual_score_delta"])) > 0.002
                ):
                    raise RuntimeError(
                        "Phase 4.2.24 completeness not reproduced: "
                        f"{model_name}, {seed}, {region}, {scanner}"
                    )

                signed_node = ig.sum(-1).cpu().numpy().astype(np.float64)
                absolute_node = ig.abs().sum(-1).cpu().numpy().astype(np.float64)
                channel_abs = ig.abs().sum(0).cpu().numpy().astype(np.float64)

                rng = stable_rng(
                    "phase4225", seed, region, scanner
                )

                fidelity = {}
                for ranking, importance in (
                    ("absolute", absolute_node),
                    ("signed_positive", signed_node),
                ):
                    selected = top_indices(importance)
                    top_drop = deletion_drop(
                        model, x, ref, opponent, scanner,
                        selected, original_score
                    )

                    controls = []
                    for _ in range(RANDOM_CONTROLS):
                        control = degree_matched_set(
                            selected, degrees, rng
                        )
                        controls.append(deletion_drop(
                            model, x, ref, opponent, scanner,
                            control, original_score
                        ))

                    fidelity[ranking] = {
                        "top_drop": float(top_drop),
                        "random_drop": float(np.mean(controls)),
                        "excess": float(top_drop - np.mean(controls)),
                    }

                profiles[scanner] = {
                    "signed": signed_node,
                    "absolute": absolute_node,
                    "channels": channel_abs,
                    "completeness_error": error,
                    "score_delta": delta,
                    "ig_sum": ig_sum,
                    "fidelity": fidelity,
                }

                for node in range(36):
                    node_rows.append({
                        "model": model_name,
                        "seed": seed,
                        "region_id": region,
                        "scanner": scanner,
                        "node_index": node,
                        "degree": int(degrees[node]),
                        "signed_ig": float(signed_node[node]),
                        "absolute_ig": float(absolute_node[node]),
                    })

                for channel in range(384):
                    channel_rows.append({
                        "model": model_name,
                        "seed": seed,
                        "region_id": region,
                        "scanner": scanner,
                        "channel": channel,
                        "absolute_ig": float(channel_abs[channel]),
                    })

            pa, pb = profiles["AT2"], profiles["GT450"]

            record = {
                "model": model_name,
                "seed": seed,
                "region_id": region,
                "absolute_node_spearman":
                    correlation(pa["absolute"], pb["absolute"]),
                "signed_node_spearman":
                    correlation(pa["signed"], pb["signed"]),
                "absolute_top5_overlap":
                    len(set(top_indices(pa["absolute"]))
                        & set(top_indices(pb["absolute"]))),
                "signed_top5_overlap":
                    len(set(top_indices(pa["signed"]))
                        & set(top_indices(pb["signed"]))),
                "channel_absolute_cosine":
                    cosine(pa["channels"], pb["channels"]),
                "at2_completeness_error": pa["completeness_error"],
                "gt450_completeness_error": pb["completeness_error"],
            }

            for scanner in SCANNERS:
                prefix = scanner.lower()
                for ranking in ("absolute", "signed_positive"):
                    result = profiles[scanner]["fidelity"][ranking]
                    record[f"{prefix}_{ranking}_top_drop"] = result["top_drop"]
                    record[f"{prefix}_{ranking}_random_drop"] = result["random_drop"]
                    record[f"{prefix}_{ranking}_excess"] = result["excess"]

            detail_rows.append(record)

        historical = next(
            row for row in source["seed_results"]
            if int(row["seed"]) == seed
        )["validation_recall_at_1"]

        mean_recall = float(np.mean(recalls))
        if abs(mean_recall - float(historical)) > 1e-5:
            raise RuntimeError("Checkpoint retrieval mismatch.")

        print(
            f"  {model_name}, seed {seed} | "
            f"R@1 {mean_recall:.4f} | "
            "Corrected IG completed",
            flush=True,
        )


print("\n[4] Aggregating model-level XAI results...", flush=True)

metrics = [
    "absolute_node_spearman",
    "signed_node_spearman",
    "absolute_top5_overlap",
    "signed_top5_overlap",
    "channel_absolute_cosine",
    "at2_completeness_error",
    "gt450_completeness_error",
]

for scanner in SCANNERS:
    for ranking in ("absolute", "signed_positive"):
        metrics += [
            f"{scanner.lower()}_{ranking}_top_drop",
            f"{scanner.lower()}_{ranking}_random_drop",
            f"{scanner.lower()}_{ranking}_excess",
        ]

summary_rows = []

for model_name in SOURCES:
    subset = [r for r in detail_rows if r["model"] == model_name]
    if len(subset) != 30:
        raise RuntimeError("Expected 30 region-seed units.")

    summary = {"model": model_name, "units": len(subset)}

    for metric in metrics:
        values = [
            float(r[metric])
            for r in subset if r[metric] is not None
        ]
        summary[f"mean_{metric}"] = (
            float(np.mean(values)) if values else None
        )

    summary_rows.append(summary)

    print(
        f"  {model_name} | "
        f"Absolute node rho "
        f"{summary['mean_absolute_node_spearman']:.4f} | "
        f"Signed node rho "
        f"{summary['mean_signed_node_spearman']:.4f} | "
        f"Signed AT2 fidelity excess "
        f"{summary['mean_at2_signed_positive_excess']:+.4f}",
        flush=True,
    )


print("\n[5] Saving corrected IG artifacts...", flush=True)

node_fields = [
    "model", "seed", "region_id", "scanner",
    "node_index", "degree", "signed_ig", "absolute_ig",
]
channel_fields = [
    "model", "seed", "region_id",
    "scanner", "channel", "absolute_ig",
]
detail_fields = ["model", "seed", "region_id"] + metrics
summary_fields = ["model", "units"] + [
    f"mean_{metric}" for metric in metrics
]

write_csv_new(NODE_OUT, node_rows, node_fields)
write_csv_new(CHANNEL_OUT, channel_rows, channel_fields)
write_csv_new(DETAIL_OUT, detail_rows, detail_fields)
write_csv_new(SUMMARY_OUT, summary_rows, summary_fields)

report = {
    "phase": "4.2",
    "step": "4.2.25",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "experiment": "corrected_training_mean_integrated_gradients",
    "baseline": "scanner_specific_training_mean",
    "integration_method": "midpoint",
    "integration_steps": STEPS,
    "integration_batch_size": STEP_BATCH,
    "target": "negative_bidirectional_within_region_InfoNCE",
    "opposite_scanner_embeddings_fixed": True,
    "ranking_methods": ["absolute", "signed_positive"],
    "fidelity_intervention":
        "replace_selected_nodes_with_scanner_training_mean",
    "top_k": TOP_K,
    "degree_matched_random_controls": RANDOM_CONTROLS,
    "graph_archive_sha256": graph_hash,
    "protocol_sha256": protocol_hash,
    "architecture_sha256": arch_hash,
    "convergence_report_sha256": hash_file(CONVERGENCE_REPORT),
    "source_model_report_hashes": {
        name: hash_file(spec["report"])
        for name, spec in SOURCES.items()
    },
    "model_summary": summary_rows,
    "node_table_sha256": hash_file(NODE_OUT),
    "channel_table_sha256": hash_file(CHANNEL_OUT),
    "detail_table_sha256": hash_file(DETAIL_OUT),
    "summary_table_sha256": hash_file(SUMMARY_OUT),
    "training_performed": False,
    "checkpoint_selection_performed": False,
    "test_evaluated": False,
    "limitations": [
        "Explanations concern DINOv2 feature inputs, not pixels",
        "Training-mean replacement changes the node representation",
        "Signed sums can cancel strong positive and negative contributions",
        "Joint deletion has nonlinear interactions",
        "Degree matching does not control all spatial confounds",
        "Fidelity ranking and evaluation use the same validation regions",
        "No independent patients, slides, or biological explanation labels",
        "Comparing two rankings is exploratory",
    ],
}

write_json_new(REPORT_OUT, report)

print("\n" + "=" * 70)
print("PHASE 4.2.25 - CORRECTED IG COMPLETED")
print("=" * 70)

for s in summary_rows:
    print(
        f"{s['model']} | "
        f"Absolute rho {s['mean_absolute_node_spearman']:.4f} | "
        f"Signed rho {s['mean_signed_node_spearman']:.4f} | "
        f"Absolute AT2 excess "
        f"{s['mean_at2_absolute_excess']:+.4f} | "
        f"Signed AT2 excess "
        f"{s['mean_at2_signed_positive_excess']:+.4f}"
    )

print("\nSaved:")
for path in (
    NODE_OUT, CHANNEL_OUT, DETAIL_OUT,
    SUMMARY_OUT, REPORT_OUT
):
    print(path)

print("\nPhase 4.2.24 completeness: REPRODUCED")
print("Training performed: NO")
print("Checkpoints modified: NO")
print("Test evaluated: NO")
print("XAI: IN PROGRESS")
print("\nSTEP 4.2.25 COMPLETED")
