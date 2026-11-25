
"""
PHASE 4.2.18 - REPRESENTATION PRESERVATION AUDIT

Compare:
  ResidualGate05, lambda=0
  TopologyConsistentResidual, lambda=1

Validation only:
  10 regions, 3 training seeds, 5 perturbation repetitions.

Evaluate:
  Normal retrieval and discriminability
  Matched scanner cosine
  Hard-negative margin
  Within-scanner node similarity
  Effective representation rank
  Same-node topology consistency
  Scanner-discordant retrieval

Verify reproduction of saved performance.

No training, no checkpoint selection, no test evaluation.
Previous research artifacts remain unchanged.
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

OLD_REPORT = BASE / "outputs/reports/28_gate_init05_report.json"
NEW_REPORT = BASE / "outputs/reports/31_topology_consistent_report.json"
AUDIT_REPORT = BASE / "outputs/reports/32_paired_robustness_audit.json"

DETAIL = BASE / "outputs/tables/33_representation_preservation_detail.csv"
SUMMARY = BASE / "outputs/tables/33_representation_preservation_summary.csv"
REPORT = BASE / "outputs/reports/33_representation_preservation_report.json"

SEEDS = (42, 123, 2026)
REPETITIONS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def stable_rng(*values):
    key = "|".join(map(str, values)).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    seed = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(seed)


def raw_adjacency(edges):
    if edges.shape != (2, 120):
        raise RuntimeError("Unexpected edge_index shape.")

    a = np.zeros((36, 36), dtype=np.float32)
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
        a[target, source] = 1.0

    if len(seen) != 120:
        raise RuntimeError("Unexpected number of edges.")

    if any((b, a) not in seen for a, b in seen):
        raise RuntimeError("Nonreciprocal edges.")

    if not np.array_equal(a, a.T):
        raise RuntimeError("Asymmetric graph.")

    return a


def normalize_adjacency(a):
    degrees = a.sum(axis=1, keepdims=True)

    if (degrees <= 0).any():
        raise RuntimeError("Unexpected isolated node.")

    return (a / degrees).astype(np.float32)


def random_adjacency(base, *key):
    """
    Exactly the randomization key and graph
    permutation used in the earlier robustness audit.
    """
    rng = stable_rng("phase4215", *key)

    for _ in range(200):
        permutation = rng.permutation(36)
        candidate = base[np.ix_(permutation, permutation)]

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
            raise RuntimeError("Graph symmetry changed.")

        return normalize_adjacency(candidate)

    raise RuntimeError("Could not generate random adjacency.")


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
            "neighbor_weights",
            adjacency.clone(),
        )
        self.conv1 = SAGEConv(384, 256)
        self.conv2 = SAGEConv(256, 384)
        self.gate_logit = nn.Parameter(torch.zeros(()))

    def gate(self):
        return torch.sigmoid(self.gate_logit)

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
            identity + self.gate() * graph,
            dim=-1,
        )


def effective_rank(z):
    """
    Effective rank of the centered node Gram matrix,
    calculated on 36 nodes within one region.
    """
    x = z.detach().cpu().numpy().astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)

    spectrum = np.linalg.eigvalsh(x @ x.T)
    spectrum = np.maximum(spectrum, 0.0)

    if spectrum.sum() <= 1e-12:
        return 0.0

    p = spectrum / spectrum.sum()
    p = p[p > 1e-12]

    return float(np.exp(-(p * np.log(p)).sum()))


def metrics(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)

    similarity = a @ b.T
    n = len(a)

    target = torch.arange(n, device=a.device)
    positive = similarity.diagonal()

    negative = similarity.clone()
    negative.fill_diagonal_(float("-inf"))
    hardest = negative.max(dim=1).values

    ranks = 1 + (
        similarity > positive[:, None]
    ).sum(dim=1)

    mask = ~torch.eye(
        n, dtype=torch.bool, device=a.device
    )

    return {
        "recall_at_1": float(
            (similarity.argmax(-1) == target)
            .float().mean().item()
        ),
        "mrr": float(
            (1.0 / ranks.float()).mean().item()
        ),
        "matched_cosine": float(positive.mean().item()),
        "hard_negative_margin": float(
            (positive - hardest).mean().item()
        ),
        "at2_offdiag_cosine": float(
            (a @ a.T)[mask].mean().item()
        ),
        "gt450_offdiag_cosine": float(
            (b @ b.T)[mask].mean().item()
        ),
        "at2_effective_rank": effective_rank(a),
        "gt450_effective_rank": effective_rank(b),
    }


def get_checkpoint(model_name, seed):
    if model_name == "ResidualGate05":
        return (
            BASE / "outputs/checkpoints/28_residual_gate_init05"
            / f"gate_init05_seed_{seed}.pt"
        )

    return (
        BASE / "outputs/checkpoints/31_topology_consistent_residual"
        / f"topology_consistent_seed_{seed}.pt"
    )


print("=" * 70, flush=True)
print("PHASE 4.2.18 - REPRESENTATION PRESERVATION AUDIT", flush=True)
print("=" * 70, flush=True)


# ----------------------------------------------------
# 1. Validate provenance and output protection
# ----------------------------------------------------

print("\n[1] Validating provenance...", flush=True)

for path in (
    GRAPH, GRAPH_REPORT, PROTOCOL, ARCH,
    OLD_REPORT, NEW_REPORT, AUDIT_REPORT,
):
    if not path.is_file():
        raise RuntimeError(f"Missing source: {path}")

for path in (DETAIL, SUMMARY, REPORT):
    if path.exists():
        raise RuntimeError(f"Refusing overwrite: {path}")

graph_hash = sha256(GRAPH)
protocol_hash = sha256(PROTOCOL)
arch_hash = sha256(ARCH)

if read_json(GRAPH_REPORT)["graph_archive_sha256"] != graph_hash:
    raise RuntimeError("Graph integrity mismatch.")

if read_json(PROTOCOL)["source_graph_sha256"] != graph_hash:
    raise RuntimeError("Protocol integrity mismatch.")

arch_data = read_json(ARCH)

if (
    arch_data["graph_archive_sha256"] != graph_hash
    or arch_data["initial_protocol_sha256"] != protocol_hash
):
    raise RuntimeError("Architecture provenance mismatch.")

references = {
    "ResidualGate05": read_json(OLD_REPORT),
    "TopologyConsistentResidual": read_json(NEW_REPORT),
}

audit = read_json(AUDIT_REPORT)

for name, source in references.items():
    if (
        source["graph_archive_sha256"] != graph_hash
        or source["protocol_sha256"] != protocol_hash
        or source["parent_architecture_sha256"] != arch_hash
        or source["test_evaluated"] is not False
    ):
        raise RuntimeError(f"Source mismatch: {name}")

if (
    audit["source_old_report_sha256"] != sha256(
        BASE / "outputs/reports/30_topology_robustness_report.json"
    )
    or audit["source_new_report_sha256"] != sha256(NEW_REPORT)
    or audit["test_evaluated"] is not False
):
    raise RuntimeError("Paired audit provenance mismatch.")


# ----------------------------------------------------
# 2. Prepare validation region pairs only
# ----------------------------------------------------

print("\n[2] Loading validation regions...", flush=True)

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
    raise RuntimeError("Invalid graph features.")

if dict(Counter(splits)) != {
    "train": 80, "validation": 20, "test": 20
}:
    raise RuntimeError("Unexpected spatial splits.")

by_region = defaultdict(dict)

for i, region in enumerate(regions):
    if splits[i] != "validation":
        continue

    if scanners[i] in by_region[region]:
        raise RuntimeError("Duplicate scanner graph.")

    by_region[region][scanners[i]] = i

region_names = sorted(by_region)

if len(region_names) != 10:
    raise RuntimeError("Expected 10 validation regions.")

pairs = {}

for region in region_names:
    pair = by_region[region]

    if set(pair) != {"AT2", "GT450"}:
        raise RuntimeError("Missing scanner pair.")

    ai, bi = pair["AT2"], pair["GT450"]

    if not np.array_equal(tiles[ai], tiles[bi]):
        raise RuntimeError("Node correspondence mismatch.")

    pairs[region] = (
        torch.from_numpy(features[ai]).to(DEVICE),
        torch.from_numpy(features[bi]).to(DEVICE),
    )

base = raw_adjacency(edges)

original_weights = torch.from_numpy(
    normalize_adjacency(base)
).to(DEVICE)

print("Validation regions:", len(pairs))
print("Seeds:", SEEDS)
print("Perturbation repetitions:", REPETITIONS)
print("Test regions: RESERVED")


# ----------------------------------------------------
# 3. Evaluate both models under matched interventions
# ----------------------------------------------------

print("\n[3] Evaluating saved checkpoints...", flush=True)

rows = []

for model_name, source in references.items():
    for seed in SEEDS:
        checkpoint = get_checkpoint(model_name, seed)

        if not checkpoint.is_file():
            raise RuntimeError(f"Missing checkpoint: {checkpoint}")

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
                raise RuntimeError("Unexpected loss weight.")

        model = ResidualGraphSAGE(
            original_weights
        ).to(DEVICE)

        model.load_state_dict(
            payload["model_state_dict"], strict=True
        )
        model.eval()

        seed_rows = []

        with torch.inference_mode():
            for region in region_names:
                a, b = pairs[region]

                original_a = model(a)
                original_b = model(b)

                normal = metrics(original_a, original_b)

                for repetition in range(REPETITIONS):
                    wa = random_adjacency(
                        base, seed, repetition, region, "AT2"
                    )
                    wb = random_adjacency(
                        base, seed, repetition, region, "GT450"
                    )

                    wa = torch.from_numpy(wa).to(DEVICE)
                    wb = torch.from_numpy(wb).to(DEVICE)

                    changed_a = model(a, override=wa)
                    changed_b = model(b, override=wb)

                    discordant = metrics(
                        changed_a, changed_b
                    )

                    same_node_a = F.cosine_similarity(
                        original_a, changed_a, dim=-1
                    ).mean()

                    same_node_b = F.cosine_similarity(
                        original_b, changed_b, dim=-1
                    ).mean()

                    row = {
                        "model": model_name,
                        "seed": seed,
                        "region_id": region,
                        "repetition": repetition,
                        "normal_recall_at_1":
                            normal["recall_at_1"],
                        "normal_mrr": normal["mrr"],
                        "normal_matched_cosine":
                            normal["matched_cosine"],
                        "normal_hard_negative_margin":
                            normal["hard_negative_margin"],
                        "normal_at2_offdiag_cosine":
                            normal["at2_offdiag_cosine"],
                        "normal_gt450_offdiag_cosine":
                            normal["gt450_offdiag_cosine"],
                        "normal_at2_effective_rank":
                            normal["at2_effective_rank"],
                        "normal_gt450_effective_rank":
                            normal["gt450_effective_rank"],
                        "discordant_recall_at_1":
                            discordant["recall_at_1"],
                        "discordant_mrr":
                            discordant["mrr"],
                        "discordant_hard_negative_margin":
                            discordant["hard_negative_margin"],
                        "at2_topology_consistency":
                            float(same_node_a.item()),
                        "gt450_topology_consistency":
                            float(same_node_b.item()),
                        "learned_gate":
                            float(model.gate().item()),
                    }

                    rows.append(row)
                    seed_rows.append(row)

        normal_recall = float(np.mean([
            r["normal_recall_at_1"] for r in seed_rows
        ]))

        discordant_recall = float(np.mean([
            r["discordant_recall_at_1"]
            for r in seed_rows
        ]))

        saved_seed = next(
            result for result in source["seed_results"]
            if int(result["seed"]) == seed
        )

        if abs(
            normal_recall -
            float(saved_seed["validation_recall_at_1"])
        ) > 1e-5:
            raise RuntimeError(
                f"Normal retrieval reproduction failed: "
                f"{model_name}, seed {seed}"
            )

        print(
            f"  {model_name}, seed {seed} | "
            f"Normal {normal_recall:.4f} | "
            f"Discordant {discordant_recall:.4f}",
            flush=True,
        )


# ----------------------------------------------------
# 4. Validate saved robustness means
# ----------------------------------------------------

print("\n[4] Verifying robustness reproduction...", flush=True)

for name, source in references.items():
    subset = [r for r in rows if r["model"] == name]

    observed = float(np.mean([
        r["discordant_recall_at_1"]
        for r in subset
    ]))

    if name == "TopologyConsistentResidual":
        expected = float(
            source["mean_scanner_discordant_recall_at_1"]
        )
    else:
        expected = float(
            audit["baseline_discordant_recall_at_1"]
        )

    if abs(observed - expected) > 1e-5:
        raise RuntimeError(
            f"Robustness reproduction failed: {name}. "
            f"Observed {observed:.6f}, expected {expected:.6f}"
        )

print("Normal and discordant retrieval: VERIFIED")


# ----------------------------------------------------
# 5. Aggregate representation metrics
# ----------------------------------------------------

print("\n[5] Aggregating representation statistics...", flush=True)

metric_columns = [
    "normal_recall_at_1",
    "normal_mrr",
    "normal_matched_cosine",
    "normal_hard_negative_margin",
    "normal_at2_offdiag_cosine",
    "normal_gt450_offdiag_cosine",
    "normal_at2_effective_rank",
    "normal_gt450_effective_rank",
    "discordant_recall_at_1",
    "discordant_mrr",
    "discordant_hard_negative_margin",
    "at2_topology_consistency",
    "gt450_topology_consistency",
    "learned_gate",
]

summary_rows = []

for name in references:
    subset = [
        r for r in rows if r["model"] == name
    ]

    result = {
        "model": name,
        "seeds": len(SEEDS),
        "regions": len(region_names),
        "repetitions": REPETITIONS,
    }

    for metric in metric_columns:
        result[f"mean_{metric}"] = float(
            np.mean([r[metric] for r in subset])
        )

    result["robustness_drop_pp"] = (
        result["mean_normal_recall_at_1"]
        - result["mean_discordant_recall_at_1"]
    ) * 100

    summary_rows.append(result)

    print(
        f"  {name}: "
        f"Normal {result['mean_normal_recall_at_1']:.4f} | "
        f"Discordant {result['mean_discordant_recall_at_1']:.4f} | "
        f"Margin {result['mean_normal_hard_negative_margin']:.4f} | "
        f"Rank {result['mean_normal_at2_effective_rank']:.2f} | "
        f"AT2 stability "
        f"{result['mean_at2_topology_consistency']:.4f}",
        flush=True,
    )


# ----------------------------------------------------
# 6. Save validated audit
# ----------------------------------------------------

print("\n[6] Saving representation audit...", flush=True)

detail_fields = [
    "model", "seed", "region_id", "repetition",
    *metric_columns,
]

save_csv(DETAIL, rows, detail_fields)

summary_fields = [
    "model", "seeds", "regions", "repetitions",
    *[f"mean_{name}" for name in metric_columns],
    "robustness_drop_pp",
]

save_csv(SUMMARY, summary_rows, summary_fields)

report = {
    "phase": "4.2",
    "step": "4.2.18",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "experiment": "representation_preservation_and_graph_dependence",
    "graph_archive_sha256": graph_hash,
    "protocol_sha256": protocol_hash,
    "architecture_sha256": arch_hash,
    "source_reports_sha256": {
        name: sha256(
            OLD_REPORT if name == "ResidualGate05"
            else NEW_REPORT
        )
        for name in references
    },
    "source_paired_audit_sha256": sha256(AUDIT_REPORT),
    "training_seeds": list(SEEDS),
    "validation_regions": len(region_names),
    "repetitions": REPETITIONS,
    "models": summary_rows,
    "normal_retrieval_verified": True,
    "discordant_retrieval_verified": True,
    "training_performed": False,
    "checkpoint_selection_performed": False,
    "test_evaluated": False,
    "detail_table_sha256": sha256(DETAIL),
    "summary_table_sha256": sha256(SUMMARY),
    "limitations": [
        "Repeated perturbations are correlated",
        "Validation is repeatedly used for model development",
        "Feature diversity is descriptive",
        "High topology consistency can result from ignoring edges",
        "Synthetic topology intervention is not a real scanner shift",
        "Ten regions are not independent patients or slides",
        "No held-out test evaluation",
    ],
    "xai_planned": True,
}

save_json(REPORT, report)

print("\n" + "=" * 70)
print("PHASE 4.2.18 - REPRESENTATION AUDIT COMPLETED")
print("=" * 70)

for result in summary_rows:
    print(
        f"{result['model']} | "
        f"Normal R@1 {result['mean_normal_recall_at_1']:.4f} | "
        f"Discordant R@1 "
        f"{result['mean_discordant_recall_at_1']:.4f} | "
        f"AT2 rank {result['mean_normal_at2_effective_rank']:.2f} | "
        f"AT2 topology cosine "
        f"{result['mean_at2_topology_consistency']:.4f}"
    )

print("\nSaved:")
print(DETAIL)
print(SUMMARY)
print(REPORT)

print("\nTraining: NO")
print("Checkpoints modified: NO")
print("Test evaluated: NO")
print("XAI: PLANNED")
print("\nSTEP 4.2.18 COMPLETED")

