
"""
PHASE 4.4.1 - FROZEN HELD-OUT EVALUATION

Evaluates the two frozen residual GNN models once
on ten reserved matched AT2/GT450 test regions.

Prespecified conditions:
  1. Normal spatial topology, one evaluation.
  2. Scanner-discordant topology, five repetitions.

Primary outcome:
  Consistency minus residual discordant Recall@1,
  averaged across 3 seeds, 10 regions, 5 repeats.

Uncertainty:
  5000 paired region-cluster bootstrap samples,
  fixed seed 43001.

Safety:
  - Verify protocol and checkpoint hashes
  - Reject changed graph archive
  - Keep model parameters frozen
  - No model selection or retraining
  - No overwriting outputs
  - No validation-based adjustments
"""

from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
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

CONFIG = (
    ROOT / "configs/45_phase43_frozen_evaluation_protocol.json"
)

FREEZE_REPORT = (
    ROOT / "outputs/reports/45_phase43_protocol_freeze_report.json"
)

AUDIT = (
    ROOT / "outputs/reports/46_pre_evaluation_integrity_audit.json"
)

EXPECTED_PROTOCOL_HASH = (
    "e6c8516e2dd279ca58b7873050c227ebe3a958a39140869065b36d38b0af78a9"
)

DETAIL_OUT = (
    ROOT / "outputs/tables/47_heldout_retrieval_detail.csv"
)

SEED_OUT = (
    ROOT / "outputs/tables/47_heldout_retrieval_by_seed.csv"
)

REGION_OUT = (
    ROOT / "outputs/tables/47_heldout_paired_regions.csv"
)

SUMMARY_OUT = (
    ROOT / "outputs/tables/47_heldout_retrieval_summary.csv"
)

REPORT_OUT = (
    ROOT / "outputs/reports/47_heldout_evaluation_report.json"
)

MODELS = (
    "ResidualGate05",
    "TopologyConsistentResidual",
)

SEEDS = (42, 123, 2026)
REPETITIONS = 5
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 43001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_csv_new(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "x", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            data,
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def stable_rng(*items):
    key = "|".join(map(str, items)).encode("utf-8")
    digest = hashlib.sha256(key).digest()

    seed = int.from_bytes(
        digest[:8], "little"
    )

    return np.random.default_rng(seed)


def build_adjacency(edge_index):
    require(
        edge_index.shape == (2, 120),
        "Incorrect graph edge shape"
    )

    a = np.zeros((36, 36), dtype=np.float32)
    seen = set()

    for source, target in edge_index.T.tolist():
        u = int(source)
        v = int(target)

        require(
            0 <= u < 36 and 0 <= v < 36
            and u != v and (u, v) not in seen,
            "Invalid directed graph edge"
        )

        seen.add((u, v))
        a[v, u] = 1.0

    require(
        len(seen) == 120,
        "Incorrect directed edge count"
    )

    require(
        np.array_equal(a, a.T),
        "Graph is not symmetric"
    )

    require(
        all((v, u) in seen for u, v in seen),
        "Missing reciprocal edges"
    )

    degrees = a.sum(axis=1)

    require(
        Counter(degrees.astype(int).tolist())
        == Counter({2: 4, 3: 16, 4: 16}),
        "Unexpected 6x6 graph degree pattern"
    )

    return a


def normalize_adjacency(a):
    degree = a.sum(axis=1, keepdims=True)

    require(
        np.all(degree > 0),
        "Graph contains an isolated node"
    )

    return (a / degree).astype(np.float32)


def randomized_weights(
    base, seed, repetition, region, scanner
):
    rng = stable_rng(
        "phase4215",
        seed,
        repetition,
        region,
        scanner,
    )

    original_degrees = np.sort(
        base.sum(axis=1)
    )

    for _ in range(200):
        permutation = rng.permutation(36)
        changed = base[
            np.ix_(permutation, permutation)
        ]

        if np.array_equal(changed, base):
            continue

        require(
            np.array_equal(changed, changed.T),
            "Random topology is asymmetric"
        )

        require(
            int(changed.sum()) == 120,
            "Random topology edge count changed"
        )

        require(
            np.array_equal(
                np.sort(changed.sum(axis=1)),
                original_degrees,
            ),
            "Random topology degree multiset changed"
        )

        return normalize_adjacency(changed)

    raise RuntimeError(
        "Failed to generate randomized topology"
    )


class SAGEConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.self_linear = nn.Linear(
            in_features, out_features
        )
        self.neighbor_linear = nn.Linear(
            in_features,
            out_features,
            bias=False,
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
            "neighbor_weights", weights.clone()
        )

        self.conv1 = SAGEConv(384, 256)
        self.conv2 = SAGEConv(256, 384)
        self.gate_logit = nn.Parameter(
            torch.zeros(())
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

        gate = torch.sigmoid(
            self.gate_logit
        )

        return F.normalize(
            identity + gate * graph,
            dim=-1,
        )


def retrieval_metrics(at2, gt450):
    """
    AT2 queries against the matching 36-node GT450
    gallery.

    Deterministic tie handling:
    earlier gallery index wins tied similarities.
    """
    similarity = at2 @ gt450.T

    require(
        similarity.shape == (36, 36),
        "Unexpected similarity matrix"
    )

    target = torch.arange(
        36, device=similarity.device
    )

    positive = similarity.diagonal()

    predictions = similarity.argmax(dim=1)

    recall = (
        (predictions == target)
        .float().mean().item()
    )

    gallery_index = target.unsqueeze(0)

    strictly_higher = (
        similarity > positive.unsqueeze(1)
    )

    equal_earlier = (
        (similarity == positive.unsqueeze(1))
        & (
            gallery_index
            < target.unsqueeze(1)
        )
    )

    ranks = (
        1
        + strictly_higher.sum(dim=1)
        + equal_earlier.sum(dim=1)
    )

    mrr = (
        (1.0 / ranks.float())
        .mean().item()
    )

    return {
        "recall_at_1": float(recall),
        "mrr": float(mrr),
    }


def average(rows, field):
    values = np.asarray(
        [r[field] for r in rows],
        dtype=np.float64,
    )

    require(
        len(values) > 0 and np.isfinite(values).all(),
        f"Invalid metric {field}"
    )

    return float(values.mean())


print("=" * 70, flush=True)
print("PHASE 4.4.1 - FROZEN HELD-OUT EVALUATION", flush=True)
print("=" * 70, flush=True)


# --------------------------------------------------
# 1. Verify frozen protocol before opening graph data
# --------------------------------------------------

print("\n[1] Verifying frozen evaluation protocol...", flush=True)

for path in (CONFIG, FREEZE_REPORT, AUDIT):
    require(
        path.is_file(),
        f"Required source missing: {path}"
    )

outputs = (
    DETAIL_OUT,
    SEED_OUT,
    REGION_OUT,
    SUMMARY_OUT,
    REPORT_OUT,
)

for path in outputs:
    require(
        not path.exists(),
        f"Refusing to overwrite: {path}"
    )

protocol_hash = sha256(CONFIG)

require(
    protocol_hash == EXPECTED_PROTOCOL_HASH,
    "Frozen evaluation protocol changed"
)

protocol = read_json(CONFIG)
freeze_report = read_json(FREEZE_REPORT)
audit = read_json(AUDIT)

require(
    freeze_report["frozen_protocol_sha256"]
    == protocol_hash,
    "Freeze report mismatch"
)

require(
    audit["frozen_protocol_sha256"]
    == protocol_hash,
    "Integrity audit mismatch"
)

require(
    audit["status"]
    == "PRE_EVALUATION_INTEGRITY_VERIFIED",
    "Pre-evaluation audit did not pass"
)

require(
    audit["test_evaluated"] is False,
    "Pre-evaluation audit records test evaluation"
)

require(
    protocol["protocol_status"] == "FROZEN"
    and protocol["test_evaluated"] is False,
    "Frozen protocol status mismatch"
)

require(
    set(protocol["models"]) == set(MODELS),
    "Unexpected frozen models"
)

test_spec = protocol["test_partition"]
condition_spec = protocol["inference_conditions"]
metric_spec = protocol["metrics"]
uncertainty_spec = protocol["uncertainty"]

require(
    test_spec["expected_matched_region_pairs"] == 10
    and test_spec["expected_scanner_graphs"] == 20,
    "Test design differs from frozen protocol"
)

require(
    condition_spec["normal_spatial"]["repetitions"] == 1
    and condition_spec[
        "scanner_discordant_random"
    ]["repetitions"] == REPETITIONS,
    "Unexpected evaluation repetition counts"
)

require(
    metric_spec["retrieval_direction"] == "AT2_to_GT450",
    "Retrieval direction mismatch"
)

require(
    metric_spec["primary"]["name"]
    == "scanner_discordant_recall_at_1_difference",
    "Primary metric mismatch"
)

require(
    metric_spec["candidate_gallery"]
    == "36_GT450_nodes_from_corresponding_region",
    "Candidate gallery mismatch"
)

require(
    uncertainty_spec["bootstrap_repetitions"]
    == BOOTSTRAP_REPS
    and uncertainty_spec["random_seed"]
    == BOOTSTRAP_SEED
    and uncertainty_spec["resampling_unit"]
    == "spatial_region",
    "Bootstrap configuration mismatch"
)

key_spec = condition_spec[
    "scanner_discordant_random"
]["seed_derivation"]

require(
    key_spec["key_components"] == [
        "phase4215", "training_seed",
        "repetition_index", "region_id", "scanner",
    ]
    and key_spec["hash"] == "sha256"
    and key_spec["seed_bytes"] == "first_8"
    and key_spec["seed_byteorder"] == "little",
    "Perturbation seed derivation changed"
)

graph_spec = protocol["data_provenance"]
graph_path = ROOT / graph_spec["graph_archive_path"]

require(
    graph_path.is_file(),
    "Graph archive missing"
)

require(
    sha256(graph_path)
    == graph_spec["graph_archive_sha256"],
    "Graph archive hash mismatch"
)

print("Frozen protocol and graph hash: VERIFIED", flush=True)


# --------------------------------------------------
# 2. Verify all model checkpoints
# --------------------------------------------------

print("\n[2] Verifying frozen model checkpoints...", flush=True)

checkpoint_paths = {}

for model_name in MODELS:
    spec = protocol["models"][model_name]

    require(
        tuple(spec["training_seeds"]) == SEEDS,
        "Frozen training seeds mismatch"
    )

    require(
        spec["allow_retraining"] is False
        and spec["allow_checkpoint_reselection"] is False,
        "Forbidden model changes allowed"
    )

    model_report_path = (
        ROOT / spec["source_report"]
    )

    require(
        sha256(model_report_path)
        == spec["source_report_sha256"],
        "Model report hash mismatch"
    )

    model_report = read_json(
        model_report_path
    )

    for seed in SEEDS:
        item = spec["checkpoints"][str(seed)]
        checkpoint = ROOT / item["path"]

        require(
            checkpoint.is_file(),
            "Missing frozen checkpoint"
        )

        actual = sha256(checkpoint)

        require(
            actual == item["sha256"]
            and actual == model_report[
                "checkpoint_hashes"
            ][str(seed)],
            "Frozen checkpoint changed"
        )

        checkpoint_paths[
            (model_name, seed)
        ] = checkpoint

    print(
        f"  {model_name}: 3 checkpoints verified",
        flush=True,
    )


# --------------------------------------------------
# 3. Open only the test partition for evaluation
# --------------------------------------------------

print("\n[3] Loading reserved test-region pairs...", flush=True)

with np.load(
    graph_path, allow_pickle=False
) as data:
    shape = data["node_features"].shape

    require(
        shape == (120, 36, 384),
        "Unexpected feature tensor shape"
    )

    splits = data["spatial_split"].astype(str)
    regions = data["region_id"].astype(str)
    scanners = data["scanner"].astype(str)
    tile_ids = data["tile_id"].astype(str)
    edge_index = data["edge_index"].copy()

    require(
        Counter(splits) == Counter({
            "train": 80,
            "validation": 20,
            "test": 20,
        }),
        "Spatial partition counts changed"
    )

    selected = np.flatnonzero(
        splits == "test"
    )

    require(
        len(selected) == 20,
        "Expected 20 test scanner graphs"
    )

    test_features = data[
        "node_features"
    ][selected].copy()

require(
    test_features.shape == (20, 36, 384)
    and test_features.dtype == np.float32
    and np.isfinite(test_features).all(),
    "Test feature tensor invalid"
)

by_region = defaultdict(dict)

for local_index, archive_index in enumerate(selected):
    region = regions[archive_index]
    scanner = scanners[archive_index]

    require(
        scanner in ("AT2", "GT450"),
        "Unexpected test scanner"
    )

    require(
        scanner not in by_region[region],
        "Duplicate scanner test graph"
    )

    by_region[region][scanner] = local_index

region_names = sorted(by_region)

require(
    len(region_names) == 10,
    "Expected ten reserved test regions"
)

test_pairs = {}

for region in region_names:
    indices = by_region[region]

    require(
        set(indices) == {"AT2", "GT450"},
        "Incomplete test scanner pair"
    )

    ia = indices["AT2"]
    ib = indices["GT450"]

    global_a = selected[ia]
    global_b = selected[ib]

    require(
        np.array_equal(
            tile_ids[global_a],
            tile_ids[global_b],
        ),
        "Test-region node correspondence mismatch"
    )

    test_pairs[region] = (
        torch.from_numpy(
            test_features[ia]
        ).to(DEVICE),
        torch.from_numpy(
            test_features[ib]
        ).to(DEVICE),
    )

base_adjacency = build_adjacency(edge_index)

normal_weights = torch.from_numpy(
    normalize_adjacency(base_adjacency)
).to(DEVICE)

print("Reserved test regions:", len(test_pairs), flush=True)
print("Test scanner graphs:", len(selected), flush=True)
print("Device:", DEVICE, flush=True)


# --------------------------------------------------
# 4. Apply fixed perturbation instances
# --------------------------------------------------

print("\n[4] Preparing prespecified scanner perturbations...", flush=True)

perturbations = {}

for seed in SEEDS:
    for repetition in range(REPETITIONS):
        for region in region_names:
            perturbations[
                (seed, repetition, region)
            ] = (
                torch.from_numpy(
                    randomized_weights(
                        base_adjacency,
                        seed,
                        repetition,
                        region,
                        "AT2",
                    )
                ).to(DEVICE),
                torch.from_numpy(
                    randomized_weights(
                        base_adjacency,
                        seed,
                        repetition,
                        region,
                        "GT450",
                    )
                ).to(DEVICE),
            )

require(
    len(perturbations) == 150,
    "Unexpected perturbation count"
)

print("Paired perturbation instances: 150", flush=True)


# --------------------------------------------------
# 5. Perform held-out inference
# --------------------------------------------------

print("\n[5] Evaluating six frozen checkpoints...", flush=True)

detail_rows = []

for model_name in MODELS:
    for seed in SEEDS:
        checkpoint = checkpoint_paths[
            (model_name, seed)
        ]

        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        spec = protocol["models"][model_name]

        require(
            int(payload["seed"]) == seed
            and payload["test_evaluated"] is False,
            "Checkpoint metadata mismatch"
        )

        require(
            payload["graph_archive_sha256"]
            == graph_spec["graph_archive_sha256"],
            "Checkpoint graph provenance mismatch"
        )

        require(
            payload["protocol_sha256"]
            == graph_spec["initial_protocol_sha256"],
            "Checkpoint training protocol mismatch"
        )

        require(
            payload["parent_architecture_sha256"]
            == graph_spec["architecture_sha256"],
            "Checkpoint architecture mismatch"
        )

        require(
            float(payload["initial_gate"]) == 0.5,
            "Unexpected checkpoint gate initialization"
        )

        if model_name == "TopologyConsistentResidual":
            require(
                float(payload["lambda_topology"]) == 1.0,
                "Consistency objective weight mismatch"
            )

        model = ResidualGraphSAGE(
            normal_weights
        ).to(DEVICE)

        model.load_state_dict(
            payload["model_state_dict"],
            strict=True,
        )

        model.eval()

        with torch.inference_mode():
            for region in region_names:
                at2, gt450 = test_pairs[region]

                normal_at2 = model(at2)
                normal_gt450 = model(gt450)

                normal_result = retrieval_metrics(
                    normal_at2, normal_gt450
                )

                detail_rows.append({
                    "model": model_name,
                    "seed": seed,
                    "region_id": region,
                    "condition": "normal_spatial",
                    "repetition": 0,
                    **normal_result,
                })

                for repetition in range(REPETITIONS):
                    wa, wb = perturbations[
                        (seed, repetition, region)
                    ]

                    changed_at2 = model(
                        at2, override=wa
                    )

                    changed_gt450 = model(
                        gt450, override=wb
                    )

                    result = retrieval_metrics(
                        changed_at2, changed_gt450
                    )

                    detail_rows.append({
                        "model": model_name,
                        "seed": seed,
                        "region_id": region,
                        "condition":
                            "scanner_discordant_random",
                        "repetition": repetition,
                        **result,
                    })

        print(
            f"  {model_name}, seed {seed}: completed",
            flush=True,
        )

require(
    len(detail_rows) == 360,
    "Incorrect held-out measurement count"
)


# --------------------------------------------------
# 6. Prespecified aggregation and primary comparison
# --------------------------------------------------

print("\n[6] Computing frozen test metrics...", flush=True)

summary_rows = []
seed_rows = []

for model_name in MODELS:
    for condition, expected in (
        ("normal_spatial", 30),
        ("scanner_discordant_random", 150),
    ):
        subset = [
            r for r in detail_rows
            if r["model"] == model_name
            and r["condition"] == condition
        ]

        require(
            len(subset) == expected,
            "Incomplete inference condition"
        )

        summary_rows.append({
            "model": model_name,
            "condition": condition,
            "measurements": len(subset),
            "mean_recall_at_1":
                average(subset, "recall_at_1"),
            "mean_mrr":
                average(subset, "mrr"),
        })

        for seed in SEEDS:
            seed_subset = [
                r for r in subset
                if r["seed"] == seed
            ]

            require(
                len(seed_subset)
                == (10 if expected == 30 else 50),
                "Incomplete seed evaluation"
            )

            seed_rows.append({
                "model": model_name,
                "seed": seed,
                "condition": condition,
                "mean_recall_at_1":
                    average(seed_subset, "recall_at_1"),
                "mean_mrr":
                    average(seed_subset, "mrr"),
            })

summary_index = {
    (r["model"], r["condition"]): r
    for r in summary_rows
}

old_normal = summary_index[
    ("ResidualGate05", "normal_spatial")
]["mean_recall_at_1"]

new_normal = summary_index[
    ("TopologyConsistentResidual", "normal_spatial")
]["mean_recall_at_1"]

old_discordant = summary_index[
    ("ResidualGate05", "scanner_discordant_random")
]["mean_recall_at_1"]

new_discordant = summary_index[
    ("TopologyConsistentResidual", "scanner_discordant_random")
]["mean_recall_at_1"]

primary_diff_pp = (
    new_discordant - old_discordant
) * 100

normal_diff_pp = (
    new_normal - old_normal
) * 100

old_drop_pp = (
    old_normal - old_discordant
) * 100

new_drop_pp = (
    new_normal - new_discordant
) * 100


# --------------------------------------------------
# 7. Paired region bootstrap
# --------------------------------------------------

print("\n[7] Computing descriptive paired uncertainty...", flush=True)

discordant_index = {
    (
        row["model"],
        row["seed"],
        row["region_id"],
        row["repetition"],
    ): row
    for row in detail_rows
    if row["condition"] == "scanner_discordant_random"
}

normal_index = {
    (
        row["model"],
        row["seed"],
        row["region_id"],
    ): row
    for row in detail_rows
    if row["condition"] == "normal_spatial"
}

region_rows = []

for region in region_names:
    discordant_differences = []
    discordant_mrr_differences = []
    normal_differences = []
    normal_mrr_differences = []

    for seed in SEEDS:
        old = normal_index[
            ("ResidualGate05", seed, region)
        ]

        new = normal_index[
            ("TopologyConsistentResidual", seed, region)
        ]

        normal_differences.append(
            new["recall_at_1"] - old["recall_at_1"]
        )

        normal_mrr_differences.append(
            new["mrr"] - old["mrr"]
        )

        for repetition in range(REPETITIONS):
            old = discordant_index[
                ("ResidualGate05", seed, region, repetition)
            ]

            new = discordant_index[
                ("TopologyConsistentResidual", seed, region, repetition)
            ]

            discordant_differences.append(
                new["recall_at_1"] - old["recall_at_1"]
            )

            discordant_mrr_differences.append(
                new["mrr"] - old["mrr"]
            )

    require(
        len(discordant_differences) == 15
        and len(normal_differences) == 3,
        "Paired region design mismatch"
    )

    region_rows.append({
        "region_id": region,
        "discordant_recall_difference_pp":
            float(np.mean(discordant_differences) * 100),
        "discordant_mrr_difference":
            float(np.mean(discordant_mrr_differences)),
        "normal_recall_difference_pp":
            float(np.mean(normal_differences) * 100),
        "normal_mrr_difference":
            float(np.mean(normal_mrr_differences)),
    })

require(
    abs(
        np.mean([
            r["discordant_recall_difference_pp"]
            for r in region_rows
        ]) - primary_diff_pp
    ) < 1e-5,
    "Primary paired aggregation mismatch"
)

rng = np.random.default_rng(
    BOOTSTRAP_SEED
)

selection = rng.integers(
    0,
    len(region_rows),
    size=(BOOTSTRAP_REPS, len(region_rows)),
)

bootstrap = {}

for metric in (
    "discordant_recall_difference_pp",
    "discordant_mrr_difference",
    "normal_recall_difference_pp",
    "normal_mrr_difference",
):
    values = np.asarray(
        [row[metric] for row in region_rows],
        dtype=np.float64,
    )

    draws = values[selection].mean(axis=1)

    lower, upper = np.quantile(
        draws, (0.025, 0.975)
    )

    bootstrap[metric] = {
        "mean": float(values.mean()),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "positive_regions":
            int(np.sum(values > 1e-10)),
        "negative_regions":
            int(np.sum(values < -1e-10)),
        "unchanged_regions":
            int(np.sum(np.abs(values) <= 1e-10)),
    }


# --------------------------------------------------
# 8. Save all prespecified results
# --------------------------------------------------

print("\n[8] Saving held-out evaluation artifacts...", flush=True)

detail_fields = [
    "model",
    "seed",
    "region_id",
    "condition",
    "repetition",
    "recall_at_1",
    "mrr",
]

seed_fields = [
    "model",
    "seed",
    "condition",
    "mean_recall_at_1",
    "mean_mrr",
]

summary_fields = [
    "model",
    "condition",
    "measurements",
    "mean_recall_at_1",
    "mean_mrr",
]

region_fields = [
    "region_id",
    "discordant_recall_difference_pp",
    "discordant_mrr_difference",
    "normal_recall_difference_pp",
    "normal_mrr_difference",
]

write_csv_new(
    DETAIL_OUT, detail_rows, detail_fields
)

write_csv_new(
    SEED_OUT, seed_rows, seed_fields
)

write_csv_new(
    REGION_OUT, region_rows, region_fields
)

write_csv_new(
    SUMMARY_OUT, summary_rows, summary_fields
)

result = {
    "phase": "4.4",
    "step": "4.4.1",
    "generated_utc":
        datetime.now(timezone.utc).isoformat(),
    "status": "HELD_OUT_EVALUATION_COMPLETED",
    "frozen_protocol_sha256": protocol_hash,
    "pre_evaluation_audit_sha256": sha256(AUDIT),
    "graph_archive_sha256":
        graph_spec["graph_archive_sha256"],
    "models": list(MODELS),
    "seeds": list(SEEDS),
    "test_regions": len(region_names),
    "test_scanner_graphs": len(selected),
    "perturbation_repetitions": REPETITIONS,
    "normal_measurements_per_model": 30,
    "discordant_measurements_per_model": 150,
    "primary_metric": {
        "name":
            "scanner_discordant_recall_at_1_difference",
        "direction":
            "TopologyConsistentResidual_minus_ResidualGate05",
        "difference_percentage_points":
            float(primary_diff_pp),
    },
    "secondary_results": {
        "normal_difference_pp":
            float(normal_diff_pp),
        "residual_normal_recall":
            float(old_normal),
        "consistency_normal_recall":
            float(new_normal),
        "residual_discordant_recall":
            float(old_discordant),
        "consistency_discordant_recall":
            float(new_discordant),
        "residual_robustness_drop_pp":
            float(old_drop_pp),
        "consistency_robustness_drop_pp":
            float(new_drop_pp),
    },
    "paired_region_bootstrap": {
        "method": "descriptive_95_percentile",
        "resampling_unit": "spatial_region",
        "samples": BOOTSTRAP_REPS,
        "seed": BOOTSTRAP_SEED,
        "metrics": bootstrap,
    },
    "summary": summary_rows,
    "detail_csv_sha256": sha256(DETAIL_OUT),
    "seed_csv_sha256": sha256(SEED_OUT),
    "region_csv_sha256": sha256(REGION_OUT),
    "summary_csv_sha256": sha256(SUMMARY_OUT),
    "test_evaluated": True,
    "model_training_performed": False,
    "checkpoint_selection_performed": False,
    "hyperparameter_tuning_performed": False,
    "limitations": [
        "Single shared biological source context",
        "Spatially held-out regions are not independent patients",
        "Synthetic graph topology shift is not real scanner shift",
        "Bootstrap intervals are descriptive",
        "Three training seeds are not independent specimens",
        "Five topology repetitions share the same region features",
        "No clinical diagnosis or classification evaluated",
        "Test results must not trigger adaptive model tuning",
    ],
}

write_json_new(
    REPORT_OUT, result
)

print("\n" + "=" * 70)
print("PHASE 4.4.1 - HELD-OUT EVALUATION COMPLETED")
print("=" * 70)

print(
    f"Residual normal R@1: {old_normal:.4f}",
    flush=True,
)
print(
    f"Consistency normal R@1: {new_normal:.4f}",
    flush=True,
)
print(
    f"Residual discordant R@1: {old_discordant:.4f}",
    flush=True,
)
print(
    f"Consistency discordant R@1: {new_discordant:.4f}",
    flush=True,
)

print(
    f"Primary paired difference: {primary_diff_pp:+.2f} pp",
    flush=True,
)

interval = bootstrap[
    "discordant_recall_difference_pp"
]

print(
    "Descriptive 95% region interval: "
    f"[{interval['lower_95']:+.2f}, "
    f"{interval['upper_95']:+.2f}] pp",
    flush=True,
)

print(
    "Regions improved / worsened / unchanged:",
    interval["positive_regions"],
    interval["negative_regions"],
    interval["unchanged_regions"],
    flush=True,
)

print("\nSaved:")
for path in outputs:
    print(path)

print("\nTraining: NO")
print("Checkpoints modified: NO")
print("Hyperparameter tuning: NO")
print("Held-out test evaluated: YES")
print("Further adaptive test tuning: PROHIBITED")
print("\nSTEP 4.4.1 COMPLETED")
