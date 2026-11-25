
"""
PHASE 4.2.10 - STABILITY-AWARE GNN ARCHITECTURE

Freeze a versioned architecture proposal and planned
ablations before training.

No training.
No test evaluation.
No existing artifacts overwritten.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json


BASE = (
    Path.home()
    / "MEDICAL_GNN_RESEARCH"
    / "phase4_main_research"
)

GRAPH = BASE / "outputs/features/15_stage_a_spatial_graphs.npz"

PROTOCOL = (
    BASE / "configs"
    / "16_initial_gnn_experimental_protocol.json"
)

REPORT_NAMES = {
    "mlp": "17_mlp_baseline_report.json",
    "gcn": "18_gcn_baseline_report.json",
    "graphsage": "19_graphsage_baseline_report.json",
    "diagnostics": "20_graphsage_diagnostics.json",
    "fixed_random": "21_permuted_topology_report.json",
    "self_only": "22_self_only_graphsage_report.json",
    "global_retrieval": "23_cross_region_retrieval_report.json",
    "region_random": "24_region_topology_report.json",
}

CONFIG_OUT = (
    BASE / "configs"
    / "25_stability_aware_gnn_architecture_v1.json"
)

REPORT_OUT = (
    BASE / "outputs/reports"
    / "25_architecture_protocol_report.json"
)


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def save_new(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as stream:
        json.dump(content, stream, indent=2)


print("=" * 70)
print("PHASE 4.2.10 - STABILITY-AWARE ARCHITECTURE")
print("=" * 70)

print("\n[1] Verifying experiment provenance...")

if not GRAPH.exists() or not PROTOCOL.exists():
    raise RuntimeError("Missing graph or protocol.")

for path in (CONFIG_OUT, REPORT_OUT):
    if path.exists():
        raise RuntimeError(
            f"Refusing to overwrite: {path}"
        )

graph_hash = sha256(GRAPH)
protocol_hash = sha256(PROTOCOL)

initial_protocol = read_json(PROTOCOL)

if initial_protocol["source_graph_sha256"] != graph_hash:
    raise RuntimeError("Protocol graph mismatch.")

results = {}
source_hashes = {}

for name, filename in REPORT_NAMES.items():
    path = BASE / "outputs/reports" / filename

    if not path.is_file():
        raise RuntimeError(
            f"Missing prior experiment: {path}"
        )

    result = read_json(path)

    # Reports 20 and 23 use graph_sha256.
    reported_hash = result.get(
        "graph_archive_sha256",
        result.get("graph_sha256"),
    )

    if reported_hash != graph_hash:
        raise RuntimeError(
            f"Graph provenance mismatch: {name}"
        )

    if result.get("protocol_sha256") != protocol_hash:
        raise RuntimeError(
            f"Protocol provenance mismatch: {name}"
        )

    if result.get("test_evaluated", False):
        raise RuntimeError(
            f"Unexpected test evaluation: {name}"
        )

    results[name] = result
    source_hashes[name] = sha256(path)

print("Verified historical experiments:", len(results))
print("Graph provenance: VERIFIED")
print("Protocol provenance: VERIFIED")


print("\n[2] Recording baseline performance...")

scores = {
    "frozen_dinov2": float(
        next(
            r["mean_validation_recall_at_1"]
            for r in results["diagnostics"]["models"]
            if r["model"] == "Frozen DINOv2"
        )
    ),
    "mlp": float(
        results["mlp"]["mean_validation_recall_at_1"]
    ),
    "gcn": float(
        results["gcn"]["mean_validation_recall_at_1"]
    ),
    "graphsage": float(
        results["graphsage"]["mean_validation_recall_at_1"]
    ),
    "fixed_random": float(
        results["fixed_random"]["mean_permuted_recall_at_1"]
    ),
    "self_only": float(
        results["self_only"]["mean_self_only_recall_at_1"]
    ),
    "region_random": float(
        results["region_random"][
            "mean_region_specific_recall_at_1"
        ]
    ),
}

for name, score in scores.items():
    print(f"  {name}: {score:.4f}")


print("\n[3] Defining candidate architecture...")

architecture = {
    "phase": "4.2",
    "step": "4.2.10",
    "version": "candidate_v1",
    "status": "architecture_specification_not_trained",
    "graph_archive_sha256": graph_hash,
    "initial_protocol_sha256": protocol_hash,

    "research_goal": (
        "Cross-scanner representation stability with "
        "node discriminability and explainability"
    ),

    "dataset": {
        "scanner_pair": ["AT2", "GT450"],
        "train_regions": 40,
        "validation_regions": 10,
        "reserved_test_regions": 10,
        "nodes_per_graph": 36,
        "feature_dimension": 384,
        "encoder": "facebook/dinov2-small",
        "encoder_frozen": True,
        "biologically_independent_test": False,
    },

    "model": {
        "working_name": "GatedResidualGraphSAGE",
        "backbone": "mean_aggregation_graphsage",
        "dimensions": [384, 256, 384],
        "activation": "GELU",
        "shared_between_scanners": True,
        "identity_residual": True,
        "gate_type": "learned_global_scalar",
        "gate_initial_value": 0.1,
        "gate_range": [0.0, 1.0],
        "output_l2_normalization": True,
        "topology": "original_spatial_four_neighbor",
        "batch_normalization": False,
        "dropout": 0.0,
    },

    "initial_training": {
        "loss": "bidirectional_within_region_InfoNCE",
        "temperature": 0.07,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_regions": 8,
        "maximum_epochs": 100,
        "early_stopping_patience": 20,
        "seeds": [42, 123, 2026],
        "checkpoint_selection": (
            "validation_within_region_recall_at_1"
            "_then_validation_loss"
        ),
        "test_used_for_selection": False,
        "diversity_regularization_initial_weight": 0.0,
    },

    "required_ablations": [
        "full_gated_residual",
        "remove_identity_residual",
        "remove_neighbor_messages",
        "fixed_gate_vs_trainable_gate",
        "without_vs_with_diversity_regularization",
        "scanner_discordant_random_topology",
    ],

    "evaluation": {
        "primary": "validation_within_region_recall_at_1",
        "secondary": [
            "validation_within_region_mrr",
            "validation_360_candidate_recall_at_1",
            "validation_360_candidate_recall_at_5",
            "validation_360_candidate_mrr",
            "reverse_scanner_retrieval",
            "cross_scanner_positive_cosine",
            "off_diagonal_node_cosine",
            "per_region_retrieval_accuracy",
            "per_region_representation_diversity",
        ],
        "analysis_unit": "spatial_region",
        "report_multiple_seeds": True,
        "note": (
            "The 360-candidate evaluation was introduced "
            "after earlier validation experiments and "
            "is exploratory, not independent confirmation."
        ),
    },

    "xai": {
        "required": True,
        "implementation_stage": "model_evaluation",
        "methods": [
            "GNNExplainer",
            "Integrated_Gradients",
            "node_edge_perturbation",
        ],
        "metrics": [
            "explanation_fidelity",
            "cross_scanner_explanation_consistency",
            "explanation_sparsity",
        ],
        "planned_script_folder": "scripts/explainable_ai",
        "planned_output_folder": "outputs/xai",
        "clinical_validation_claims_allowed": False,
    },

    "scientific_restrictions": [
        "No test-set hyperparameter selection",
        "Do not claim independently sampled tissue",
        "Do not claim random topology is superior",
        "Do not claim anatomical topology is validated",
        "Near-ceiling retrieval requires careful interpretation",
        "Explicitly compare against stronger existing controls",
        "Report ablations before superiority claims",
        "No clinical interpretability claims from XAI alone",
    ],

    "historical_baseline_validation_recall_at_1": scores,
    "historical_report_hashes": source_hashes,
}

save_new(CONFIG_OUT, architecture)


print("\n[4] Saving protocol verification...")

verification = {
    "phase": "4.2",
    "step": "4.2.10",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "architecture_status": "specified_not_trained",
    "graph_archive_sha256": graph_hash,
    "initial_protocol_sha256": protocol_hash,
    "architecture_sha256": sha256(CONFIG_OUT),
    "historical_experiments_verified": len(results),
    "prior_results_modified": False,
    "test_evaluated": False,
    "biological_independence_verified": False,
    "xai_included": True,
    "next_experiment": (
        "Train initial gated residual GraphSAGE "
        "under the existing validation protocol"
    ),
}

save_new(REPORT_OUT, verification)

print("\n" + "=" * 70)
print("PHASE 4.2.10 - ARCHITECTURE SPECIFIED")
print("=" * 70)

print("Baseline reports verified:", len(results))
print("Architecture:", architecture["model"]["working_name"])
print("Initial gate:", architecture["model"]["gate_initial_value"])
print("Planned ablations:", len(architecture["required_ablations"]))

print("\nSaved:")
print(CONFIG_OUT)
print(REPORT_OUT)

print("\nModel training: NOT STARTED")
print("Held-out test: NOT EVALUATED")
print("Historical experiments: UNCHANGED")
print("XAI: INCLUDED IN ARCHITECTURE PLAN")
print("\nSTEP 4.2.10 COMPLETED")
