
"""
PHASE 4.3.1 - FINAL EVALUATION PROTOCOL FREEZE

Creates a provenance-verified frozen protocol for the
existing residual GraphSAGE models.

No model inference.
No data partition loaded.
No test evaluation.
No training or checkpoint selection.

Outputs:
  configs/45_phase43_frozen_evaluation_protocol.json
  outputs/reports/45_phase43_protocol_freeze_report.json

Both outputs use exclusive creation and cannot
silently replace an existing freeze.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math


ROOT = (
    Path.home()
    / "MEDICAL_GNN_RESEARCH"
    / "phase4_main_research"
)

GRAPH = ROOT / "outputs/features/15_stage_a_spatial_graphs.npz"
GRAPH_REPORT = ROOT / "outputs/reports/15_graph_dataset_report.json"
PROTOCOL_16 = ROOT / "configs/16_initial_gnn_experimental_protocol.json"
ARCH_25 = ROOT / "configs/25_stability_aware_gnn_architecture_v1.json"

MODEL_REPORTS = {
    "ResidualGate05":
        ROOT / "outputs/reports/28_gate_init05_report.json",
    "TopologyConsistentResidual":
        ROOT / "outputs/reports/31_topology_consistent_report.json",
}

CHECKPOINT_FOLDERS = {
    "ResidualGate05": (
        "outputs/checkpoints/28_residual_gate_init05",
        "gate_init05_seed_{seed}.pt",
    ),
    "TopologyConsistentResidual": (
        "outputs/checkpoints/31_topology_consistent_residual",
        "topology_consistent_seed_{seed}.pt",
    ),
}

SOURCE_REPORTS = {
    "robustness_reference":
        ROOT / "outputs/reports/30_topology_robustness_report.json",
    "paired_robustness":
        ROOT / "outputs/reports/32_paired_robustness_audit.json",
    "representation_audit":
        ROOT / "outputs/reports/33_representation_preservation_report.json",
    "corrected_ig_audit":
        ROOT / "outputs/reports/41_corrected_ig_audit_report.json",
    "edge_mask_audit":
        ROOT / "outputs/reports/43_edge_mask_reliability_report.json",
    "research_synthesis":
        ROOT / "outputs/reports/44_research_synthesis_report.json",
}

FROZEN_CONFIG = (
    ROOT / "configs/45_phase43_frozen_evaluation_protocol.json"
)

FREEZE_REPORT = (
    ROOT / "outputs/reports/45_phase43_protocol_freeze_report.json"
)

SEEDS = (42, 123, 2026)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            h.update(block)

    return h.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def create_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            content,
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")


def relative(path):
    return str(path.relative_to(ROOT))


print("=" * 70)
print("PHASE 4.3.1 - FINAL EVALUATION PROTOCOL FREEZE")
print("=" * 70)

# ---------------------------------------------------
# 1. Validate all source artifacts
# ---------------------------------------------------

print("\n[1] Checking required research artifacts...", flush=True)

required = [
    GRAPH, GRAPH_REPORT, PROTOCOL_16, ARCH_25,
    *MODEL_REPORTS.values(),
    *SOURCE_REPORTS.values(),
]

for path in required:
    if not path.is_file():
        raise RuntimeError(f"Missing source: {path}")

for path in (FROZEN_CONFIG, FREEZE_REPORT):
    if path.exists():
        raise RuntimeError(
            f"Freeze already exists; refusing overwrite: {path}"
        )

graph_hash = sha256(GRAPH)
protocol_hash = sha256(PROTOCOL_16)
architecture_hash = sha256(ARCH_25)

if (
    load_json(GRAPH_REPORT)["graph_archive_sha256"]
    != graph_hash
):
    raise RuntimeError("Graph integrity mismatch.")

if (
    load_json(PROTOCOL_16)["source_graph_sha256"]
    != graph_hash
):
    raise RuntimeError("Original protocol mismatch.")

architecture = load_json(ARCH_25)

if (
    architecture["graph_archive_sha256"] != graph_hash
    or architecture["initial_protocol_sha256"]
    != protocol_hash
):
    raise RuntimeError("Architecture provenance mismatch.")

source = {
    name: load_json(path)
    for name, path in SOURCE_REPORTS.items()
}

if source["research_synthesis"]["step"] != "4.2.29":
    raise RuntimeError("Research synthesis not finalized.")

synthesis = source["research_synthesis"]

for name, path in SOURCE_REPORTS.items():
    item = source[name]

    if item.get("test_evaluated") is not False:
        raise RuntimeError(
            f"Historical test status invalid: {name}"
        )

    if item.get("graph_archive_sha256") != graph_hash:
        raise RuntimeError(
            f"Graph mismatch in {name}"
        )

    if item.get("protocol_sha256") != protocol_hash:
        raise RuntimeError(
            f"Protocol mismatch in {name}"
        )

# Phase 4.2.29 recorded the source hashes
# used during the final synthesis.
synthesis_sources = synthesis["source_sha256"]

required_synthesis_links = {
    "representation_report": SOURCE_REPORTS[
        "representation_audit"
    ],
    "ig_report": SOURCE_REPORTS[
        "corrected_ig_audit"
    ],
    "edge_report": SOURCE_REPORTS[
        "edge_mask_audit"
    ],
}

for key, path in required_synthesis_links.items():
    if synthesis_sources[key] != sha256(path):
        raise RuntimeError(
            f"Synthesis source changed: {key}"
        )

print("Historical graph, protocol, and XAI reports: VERIFIED")


# ---------------------------------------------------
# 2. Lock existing model checkpoints
# ---------------------------------------------------

print("\n[2] Verifying model checkpoints...", flush=True)

models = {}
model_references = {}

for model_name, report_path in MODEL_REPORTS.items():
    report = load_json(report_path)

    if (
        report["graph_archive_sha256"] != graph_hash
        or report["protocol_sha256"] != protocol_hash
        or report["parent_architecture_sha256"]
        != architecture_hash
        or report["test_evaluated"] is not False
    ):
        raise RuntimeError(
            f"Model provenance mismatch: {model_name}"
        )

    if tuple(report["seeds"]) != SEEDS:
        raise RuntimeError(
            f"Unexpected training seeds: {model_name}"
        )

    folder, template = CHECKPOINT_FOLDERS[model_name]

    checkpoints = {}

    for seed in SEEDS:
        path = ROOT / folder / template.format(seed=seed)

        if not path.is_file():
            raise RuntimeError(
                f"Missing checkpoint: {path}"
            )

        actual_hash = sha256(path)
        recorded_hash = report["checkpoint_hashes"][str(seed)]

        if actual_hash != recorded_hash:
            raise RuntimeError(
                f"Checkpoint altered: {path}"
            )

        checkpoints[str(seed)] = {
            "path": relative(path),
            "sha256": actual_hash,
        }

    models[model_name] = {
        "source_report": relative(report_path),
        "source_report_sha256": sha256(report_path),
        "training_seeds": list(SEEDS),
        "checkpoints": checkpoints,
        "architecture": "gated_residual_graphsage",
        "original_initial_gate": 0.5,
        "topology_consistency_lambda": (
            0.0 if model_name == "ResidualGate05"
            else 1.0
        ),
        "allow_retraining": False,
        "allow_checkpoint_reselection": False,
    }

    model_references[model_name] = report

    print(
        f"  {model_name}: 3 checkpoint hashes VERIFIED",
        flush=True,
    )


# ---------------------------------------------------
# 3. Verify synthesis-level reference measurements
# ---------------------------------------------------

print("\n[3] Recording validation references...", flush=True)

synthesis_models = {
    row["model"]: row
    for row in synthesis["models"]
}

if set(synthesis_models) != set(models):
    raise RuntimeError(
        "Final synthesis model list mismatch."
    )

validation_references = {}

for model_name in models:
    row = synthesis_models[model_name]

    normal = float(row["normal_recall_at_1"])
    discordant = float(row["discordant_recall_at_1"])

    if not (
        math.isfinite(normal)
        and math.isfinite(discordant)
        and 0 <= normal <= 1
        and 0 <= discordant <= 1
    ):
        raise RuntimeError(
            "Invalid synthesis retrieval metric."
        )

    validation_references[model_name] = {
        "normal_recall_at_1": normal,
        "scanner_discordant_recall_at_1": discordant,
    }

print("Validation references: VERIFIED")


# ---------------------------------------------------
# 4. Specify held-out evaluation protocol
# ---------------------------------------------------

print("\n[4] Constructing frozen test protocol...", flush=True)

freeze = {
    "phase": "4.3",
    "step": "4.3.1",
    "name": "frozen_cross_scanner_graph_retrieval_protocol",
    "protocol_status": "FROZEN",
    "created_utc":
        datetime.now(timezone.utc).isoformat(),

    "research_scope": {
        "task": "matched_tissue_tile_retrieval",
        "dataset": "Owkin PLISM subset",
        "scanners": ["AT2", "GT450"],
        "biological_scope":
            "single_tissue_condition_shared_source",
        "feature_extractor": "facebook/dinov2-small",
        "feature_dimension": 384,
        "nodes_per_graph": 36,
        "directed_edges_per_graph": 120,
        "undirected_edges_per_graph": 60,
        "graph_type": "6x6_four_neighbor_grid",
        "independent_patient_claim": False,
        "clinical_classification_claim": False,
    },

    "data_provenance": {
        "graph_archive_path": relative(GRAPH),
        "graph_archive_sha256": graph_hash,
        "initial_protocol_sha256": protocol_hash,
        "architecture_sha256": architecture_hash,
        "test_data_loaded_during_freeze": False,
    },

    "models": models,

    "test_partition": {
        "split_name": "test",
        "expected_matched_region_pairs": 10,
        "expected_scanner_graphs": 20,
        "region_pairing": "same_region_id",
        "node_correspondence": "exact_tile_id_order",
        "allow_checkpoint_selection": False,
        "allow_model_training": False,
        "allow_hyperparameter_tuning": False,
        "allow_repeated_adaptive_test_evaluation": False,
    },

    "inference_conditions": {
        "normal_spatial": {
            "adjacency": "original_degree_normalized_grid",
            "same_adjacency_for_both_scanners": True,
            "repetitions": 1,
        },

        "scanner_discordant_random": {
            "adjacency":
                "independent_scanner_permutations_of_original_grid",
            "same_adjacency_for_both_scanners": False,
            "repetitions": 5,
            "preserve_edge_count": True,
            "preserve_degree_multiset": True,
            "preserve_undirected_symmetry": True,
            "node_feature_order_fixed": True,
            "permutation_random_source":
                "numpy_default_rng",
            "seed_derivation": {
                "key_components": [
                    "phase4215",
                    "training_seed",
                    "repetition_index",
                    "region_id",
                    "scanner",
                ],
                "join_character": "|",
                "hash": "sha256",
                "seed_bytes": "first_8",
                "seed_byteorder": "little",
            },
            "permutation_retry_limit": 200,
            "exclude_original_adjacency": True,
            "edge_row_normalization": True,
        },
    },

    "metrics": {
        "primary": {
            "name":
                "scanner_discordant_recall_at_1_difference",
            "direction":
                "TopologyConsistentResidual_minus_ResidualGate05",
            "unit": "percentage_points",
            "aggregation":
                "mean_over_seeds_regions_and_five_repetitions",
        },
        "secondary": [
            "normal_topology_recall_at_1",
            "scanner_discordant_mrr",
            "normal_topology_mrr",
            "robustness_degradation_percentage_points",
        ],
        "retrieval_direction": "AT2_to_GT450",
        "candidate_gallery":
            "36_GT450_nodes_from_corresponding_region",
        "ranking": "descending_cosine_similarity",
        "target": "matching_node_index",
        "tie_handling": "torch_argmax_first_index",
    },

    "uncertainty": {
        "method": "paired_spatial_region_bootstrap",
        "bootstrap_repetitions": 5000,
        "random_seed": 43001,
        "resampling_unit": "spatial_region",
        "keep_all_seeds_within_region": True,
        "keep_all_perturbation_repetitions_within_region":
            True,
        "interval": "descriptive_95_percentile",
        "independent_patient_inference": False,
        "independent_slide_inference": False,
    },

    "xai_reporting": {
        "structural_node_perturbation":
            "exploratory_validation_evidence",
        "corrected_integrated_gradients":
            "exploratory_validation_evidence",
        "learned_edge_masks":
            "exploratory_validation_evidence",
        "selection_using_test_xai": False,
        "new_test_xai_optimization_planned": False,
        "claim_of_biological_explanation_validity": False,
    },

    "evaluation_rules": {
        "evaluate_only_saved_checkpoint_weights": True,
        "verify_checkpoint_hashes_before_inference": True,
        "verify_graph_archive_sha256_before_inference": True,
        "fail_on_region_or_tile_pairing_mismatch": True,
        "fail_on_partition_count_mismatch": True,
        "never_overwrite_evaluation_outputs": True,
        "no_retraining_after_test_inspection": True,
        "no_model_changes_after_test_inspection": True,
        "no_selective_reporting_of_seeds_or_regions": True,
        "report_all_prespecified_conditions": True,
        "report_negative_results": True,
    },

    "validation_reference_only": validation_references,

    "limitations": [
        "One biological tissue/source context",
        "Spatial regions are not independent patients",
        "Only two scanner views of matched tissue locations",
        "Synthetic topology disagreement is not real scanner shift",
        "Validation has been extensively used for development",
        "Test results will not establish clinical applicability",
        "Validation XAI is exploratory and method dependent",
    ],

    "test_evaluated": False,
}

# ---------------------------------------------------
# 5. Write frozen protocol and verification report
# ---------------------------------------------------

print("\n[5] Writing frozen protocol...", flush=True)

create_json(FROZEN_CONFIG, freeze)

frozen_hash = sha256(FROZEN_CONFIG)

report = {
    "phase": "4.3",
    "step": "4.3.1",
    "status": "PROTOCOL_FROZEN",
    "created_utc":
        datetime.now(timezone.utc).isoformat(),
    "frozen_protocol_path": relative(FROZEN_CONFIG),
    "frozen_protocol_sha256": frozen_hash,
    "source_report_hashes": {
        name: sha256(path)
        for name, path in SOURCE_REPORTS.items()
    },
    "model_checkpoint_hashes_verified": True,
    "model_count": len(models),
    "training_seeds": list(SEEDS),
    "planned_test_region_pairs": 10,
    "test_data_loaded": False,
    "test_evaluated": False,
    "model_training_performed": False,
    "model_inference_performed": False,
    "checkpoint_selection_performed": False,
    "next_stage":
        "4.3.2_pre_evaluation_protocol_integrity_audit",
}

create_json(FREEZE_REPORT, report)

print("\n" + "=" * 70)
print("PHASE 4.3.1 - PROTOCOL FREEZE COMPLETED")
print("=" * 70)

print("Frozen protocol:", FROZEN_CONFIG)
print("Protocol SHA256:", frozen_hash)
print("Verification report:", FREEZE_REPORT)

print("\nModels frozen:", len(models))
print("Seeds:", SEEDS)
print("Checkpoints verified:", len(models) * len(SEEDS))
print("Training: NO")
print("Model inference: NO")
print("Test data loaded: NO")
print("Test evaluated: NO")
print("\nSTEP 4.3.1 COMPLETED")
