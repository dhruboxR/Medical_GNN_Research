
"""
PHASE 4.3.2 - PRE-EVALUATION INTEGRITY AUDIT

Checks:
  1. Frozen protocol checksum
  2. Freeze verification report
  3. All six checkpoint file hashes
  4. Referenced model report hashes
  5. Frozen test evaluation specification
  6. Synthetic graph perturbation reproducibility
  7. Graph symmetry, edge count and degree preservation

Does NOT:
  - Open the graph archive
  - Read test features or identifiers
  - Load model weights into a network
  - Perform inference
  - Retrain or select checkpoints
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np


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
OUTPUT = (
    ROOT / "outputs/reports/46_pre_evaluation_integrity_audit.json"
)

EXPECTED_FREEZE_SHA256 = (
    "e6c8516e2dd279ca58b7873050c227ebe3a958a39140869065b36d38b0af78a9"
)

SEEDS = (42, 123, 2026)
MODELS = (
    "ResidualGate05",
    "TopologyConsistentResidual",
)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)

    return h.hexdigest()


def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_new_json(path, value):
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    with path.open("x", encoding="utf-8") as f:
        json.dump(
            value, f, indent=2,
            sort_keys=True, allow_nan=False
        )
        f.write("\n")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def stable_rng(*parts):
    key = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.sha256(key).digest()

    seed = int.from_bytes(
        digest[:8], "little"
    )

    return np.random.default_rng(seed)


def synthetic_grid():
    """
    Construct a 6x6 four-neighbor grid using
    generated coordinates only.
    """
    a = np.zeros((36, 36), dtype=np.float32)

    for r in range(6):
        for c in range(6):
            u = r * 6 + c

            if c + 1 < 6:
                v = u + 1
                a[u, v] = 1.0
                a[v, u] = 1.0

            if r + 1 < 6:
                v = u + 6
                a[u, v] = 1.0
                a[v, u] = 1.0

    return a


def normalize_adjacency(a):
    degree = a.sum(axis=1, keepdims=True)

    require(
        np.all(degree > 0),
        "Unexpected isolated node"
    )

    return (
        a / degree
    ).astype(np.float32)


def random_adjacency(base, seed, repetition,
                     region, scanner):
    """
    Reproduce the frozen perturbation scheme.

    Node features stay in their original order.
    The graph is permuted independently for
    each scanner.
    """
    rng = stable_rng(
        "phase4215",
        seed,
        repetition,
        region,
        scanner,
    )

    for _ in range(200):
        permutation = rng.permutation(36)

        candidate = base[
            np.ix_(permutation, permutation)
        ]

        if np.array_equal(candidate, base):
            continue

        return candidate

    raise RuntimeError(
        "Failed to generate changed adjacency"
    )


print("=" * 70, flush=True)
print("PHASE 4.3.2 - PRE-EVALUATION INTEGRITY AUDIT", flush=True)
print("=" * 70, flush=True)


# --------------------------------------------------
# 1. Verify protocol checksum and report
# --------------------------------------------------

print("\n[1] Checking frozen protocol...", flush=True)

for path in (CONFIG, FREEZE_REPORT):
    require(
        path.is_file(),
        f"Required file missing: {path}"
    )

require(
    not OUTPUT.exists(),
    "Audit output already exists; refusing overwrite"
)

actual_hash = sha256(CONFIG)

require(
    actual_hash == EXPECTED_FREEZE_SHA256,
    "Frozen protocol SHA-256 changed"
)

protocol = load_json(CONFIG)
freeze_report = load_json(FREEZE_REPORT)

require(
    protocol["protocol_status"] == "FROZEN",
    "Protocol is not frozen"
)
require(
    protocol["step"] == "4.3.1",
    "Unexpected protocol version"
)
require(
    protocol["test_evaluated"] is False,
    "Unexpected test status"
)
require(
    freeze_report["frozen_protocol_sha256"] == actual_hash,
    "Freeze report checksum mismatch"
)
require(
    freeze_report["test_data_loaded"] is False,
    "Freeze recorded test-data access"
)
require(
    freeze_report["test_evaluated"] is False,
    "Freeze recorded test evaluation"
)

print("Frozen protocol checksum: VERIFIED", flush=True)


# --------------------------------------------------
# 2. Check model and checkpoint provenance
# --------------------------------------------------

print("\n[2] Verifying six frozen checkpoints...", flush=True)

require(
    set(protocol["models"]) == set(MODELS),
    "Unexpected model list"
)

checkpoint_count = 0

for name in MODELS:
    model_spec = protocol["models"][name]

    require(
        tuple(model_spec["training_seeds"]) == SEEDS,
        "Seed list mismatch"
    )

    require(
        model_spec["allow_retraining"] is False,
        "Retraining unexpectedly permitted"
    )

    require(
        model_spec["allow_checkpoint_reselection"] is False,
        "Checkpoint reselection unexpectedly permitted"
    )

    source_report = (
        ROOT / model_spec["source_report"]
    )

    require(
        source_report.is_file(),
        "Source model report missing"
    )

    require(
        sha256(source_report)
        == model_spec["source_report_sha256"],
        "Source model report changed"
    )

    reference = load_json(source_report)

    require(
        reference["test_evaluated"] is False,
        "Model source records test evaluation"
    )

    for seed in SEEDS:
        checkpoint_info = model_spec[
            "checkpoints"
        ][str(seed)]

        checkpoint = (
            ROOT / checkpoint_info["path"]
        )

        require(
            checkpoint.is_file(),
            f"Checkpoint missing: {checkpoint}"
        )

        observed = sha256(checkpoint)

        require(
            observed == checkpoint_info["sha256"],
            f"Checkpoint changed: {checkpoint}"
        )

        require(
            observed
            == reference["checkpoint_hashes"][str(seed)],
            "Checkpoint source hash mismatch"
        )

        checkpoint_count += 1

    print(
        f"  {name}: 3 checkpoints VERIFIED",
        flush=True
    )

require(
    checkpoint_count == 6,
    "Incorrect checkpoint count"
)


# --------------------------------------------------
# 3. Verify evaluation specifications
# --------------------------------------------------

print("\n[3] Verifying locked test design...", flush=True)

test = protocol["test_partition"]
conditions = protocol["inference_conditions"]
metrics = protocol["metrics"]
rules = protocol["evaluation_rules"]

require(
    test["split_name"] == "test",
    "Test split mismatch"
)

require(
    test["expected_matched_region_pairs"] == 10,
    "Expected test regions changed"
)

require(
    test["expected_scanner_graphs"] == 20,
    "Expected test graphs changed"
)

require(
    all(
        test[field] is False
        for field in (
            "allow_checkpoint_selection",
            "allow_model_training",
            "allow_hyperparameter_tuning",
            "allow_repeated_adaptive_test_evaluation",
        )
    ),
    "Test tuning restrictions changed"
)

require(
    conditions["normal_spatial"]["repetitions"] == 1,
    "Normal evaluation repetitions changed"
)

discordant = conditions[
    "scanner_discordant_random"
]

require(
    discordant["repetitions"] == 5,
    "Discordant repetitions changed"
)

require(
    discordant["preserve_edge_count"] is True
    and discordant["preserve_degree_multiset"] is True
    and discordant["preserve_undirected_symmetry"] is True
    and discordant["node_feature_order_fixed"] is True,
    "Graph intervention requirements changed"
)

key_spec = discordant["seed_derivation"]

require(
    key_spec["key_components"] == [
        "phase4215",
        "training_seed",
        "repetition_index",
        "region_id",
        "scanner",
    ],
    "Randomization key specification changed"
)

require(
    key_spec["hash"] == "sha256"
    and key_spec["seed_bytes"] == "first_8"
    and key_spec["seed_byteorder"] == "little",
    "Randomization hash specification changed"
)

require(
    metrics["primary"]["name"]
    == "scanner_discordant_recall_at_1_difference",
    "Primary metric changed"
)

require(
    metrics["retrieval_direction"] == "AT2_to_GT450",
    "Retrieval direction changed"
)

require(
    metrics["candidate_gallery"]
    == "36_GT450_nodes_from_corresponding_region",
    "Candidate gallery changed"
)

require(
    rules["no_retraining_after_test_inspection"] is True
    and rules["no_model_changes_after_test_inspection"] is True,
    "Post-test restrictions changed"
)

print("Frozen evaluation requirements: VERIFIED", flush=True)


# --------------------------------------------------
# 4. Synthetic perturbation reproducibility
# --------------------------------------------------

print("\n[4] Testing synthetic graph perturbations...", flush=True)

base = synthetic_grid()

require(
    int(base.sum()) == 120,
    "Synthetic graph edge count incorrect"
)

require(
    np.array_equal(base, base.T),
    "Synthetic grid is asymmetric"
)

degree_multiset = np.sort(base.sum(axis=1))
perturbation_cases = 0

for seed in SEEDS:
    for repetition in range(5):
        for scanner in ("AT2", "GT450"):
            a = random_adjacency(
                base,
                seed,
                repetition,
                "SYNTHETIC_VALIDATION_ONLY",
                scanner,
            )

            repeated = random_adjacency(
                base,
                seed,
                repetition,
                "SYNTHETIC_VALIDATION_ONLY",
                scanner,
            )

            require(
                np.array_equal(a, repeated),
                "Graph randomization is not reproducible"
            )

            require(
                not np.array_equal(a, base),
                "Random graph equals original"
            )

            require(
                np.array_equal(a, a.T),
                "Random graph asymmetric"
            )

            require(
                int(a.sum()) == 120,
                "Edge count changed"
            )

            require(
                np.array_equal(
                    np.sort(a.sum(axis=1)),
                    degree_multiset,
                ),
                "Degree multiset changed"
            )

            weights = normalize_adjacency(a)

            require(
                np.allclose(
                    weights.sum(axis=1), 1.0
                ),
                "Row normalization failed"
            )

            perturbation_cases += 1

print(
    f"Synthetic cases verified: {perturbation_cases}",
    flush=True
)


# --------------------------------------------------
# 5. Save audit report
# --------------------------------------------------

print("\n[5] Saving audit report...", flush=True)

result = {
    "phase": "4.3",
    "step": "4.3.2",
    "status": "PRE_EVALUATION_INTEGRITY_VERIFIED",
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "frozen_protocol_sha256": actual_hash,
    "freeze_report_sha256": sha256(FREEZE_REPORT),
    "checkpoint_hashes_verified": True,
    "verified_checkpoint_count": checkpoint_count,
    "synthetic_perturbation_cases": perturbation_cases,
    "synthetic_perturbations_reproducible": True,
    "frozen_test_design_verified": True,
    "graph_archive_opened": False,
    "test_features_loaded": False,
    "test_identifiers_loaded": False,
    "model_inference_performed": False,
    "training_performed": False,
    "checkpoint_selection_performed": False,
    "test_evaluated": False,
    "next_stage": "4.4.1_held_out_test_evaluation",
    "limitations": [
        "Synthetic checks verify specified graph properties",
        "Synthetic checks do not establish exact historical implementation equivalence",
        "Graph archive contents are intentionally not inspected",
        "Graph archive checksum must be verified during final evaluation",
        "Test results cannot trigger model changes",
    ],
}

write_new_json(OUTPUT, result)

print("\n" + "=" * 70)
print("PHASE 4.3.2 - INTEGRITY AUDIT COMPLETED")
print("=" * 70)

print("Protocol SHA256:", actual_hash)
print("Verified checkpoints:", checkpoint_count)
print("Synthetic perturbation cases:", perturbation_cases)
print("Saved:", OUTPUT)
print()
print("Training: NO")
print("Model inference: NO")
print("Graph archive opened: NO")
print("Test evaluated: NO")
print("Next: PHASE 4.4.1")
