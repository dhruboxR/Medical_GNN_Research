
"""
PHASE 4.2.1 - GNN EXPERIMENTAL PROTOCOL

Validate the graph dataset and save a versioned
experimental protocol before beginning model training.

No model is trained.
No existing dataset or artifact is modified.
Test labels/metrics are not used for model selection.

The spatial splits share biological material.
They are not independent-patient evaluation splits.
"""

from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json

import numpy as np


ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "phase4_main_research"

GRAPH_FILE = (
    BASE / "outputs/features"
    / "15_stage_a_spatial_graphs.npz"
)
SOURCE_REPORT = (
    BASE / "outputs/reports"
    / "15_graph_dataset_report.json"
)

CONFIG_FILE = (
    BASE / "configs"
    / "16_initial_gnn_experimental_protocol.json"
)
REPORT_FILE = (
    BASE / "outputs/reports"
    / "16_gnn_protocol_verification.json"
)

EXPECTED_SPLITS = {
    "train": 80,
    "validation": 20,
    "test": 20,
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)


print("=" * 70)
print("PHASE 4.2.1 - GNN EXPERIMENTAL PROTOCOL")
print("=" * 70)

print("\n[1] Validating existing graph dataset...")

for path in (GRAPH_FILE, SOURCE_REPORT):
    if not path.is_file():
        raise RuntimeError(f"Missing input: {path}")

for path in (CONFIG_FILE, REPORT_FILE):
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite: {path}")

with SOURCE_REPORT.open(encoding="utf-8") as stream:
    source_report = json.load(stream)

graph_hash = sha256(GRAPH_FILE)

if graph_hash != source_report["graph_archive_sha256"]:
    raise RuntimeError("Graph archive SHA256 mismatch.")

with np.load(GRAPH_FILE, allow_pickle=False) as archive:
    required = {
        "node_features", "edge_index", "region_id",
        "spatial_split", "scanner", "tile_id",
        "x", "y", "topology",
        "original_feature_archive_sha256",
    }

    if not required.issubset(archive.files):
        raise RuntimeError("Incomplete graph archive.")

    features = archive["node_features"]
    edges = archive["edge_index"]

    regions = archive["region_id"].astype(str)
    splits = archive["spatial_split"].astype(str)
    scanners = archive["scanner"].astype(str)
    tiles = archive["tile_id"].astype(str)
    xs = archive["x"]
    ys = archive["y"]

    topology = str(archive["topology"])
    feature_hash = str(
        archive["original_feature_archive_sha256"]
    )

if features.shape != (120, 36, 384):
    raise RuntimeError("Invalid graph feature dimensions.")

if not np.isfinite(features).all():
    raise RuntimeError("Non-finite node features.")

if edges.shape != (2, 120):
    raise RuntimeError("Unexpected edge-index shape.")

if topology != "spatial_4_neighbor":
    raise RuntimeError("Unexpected graph topology.")

if feature_hash != source_report[
    "source_feature_archive_sha256"
]:
    raise RuntimeError("Feature archive provenance mismatch.")

if (
    tiles.shape != (120, 36)
    or xs.shape != (120, 36)
    or ys.shape != (120, 36)
):
    raise RuntimeError("Graph coordinates mismatch.")

if len(set(regions)) != 60:
    raise RuntimeError("Expected 60 distinct regions.")

if dict(Counter(splits)) != EXPECTED_SPLITS:
    raise RuntimeError("Unexpected graph split counts.")

if any(scanner not in ("AT2", "GT450") for scanner in scanners):
    raise RuntimeError("Unexpected scanner.")

edge_set = set(map(tuple, edges.T.tolist()))

if len(edge_set) != 120:
    raise RuntimeError("Duplicate directed edges.")

if any(
    source == target
    or (target, source) not in edge_set
    for source, target in edge_set
):
    raise RuntimeError("Graph edge reciprocity failed.")

if any(
    not (0 <= source < 36 and 0 <= target < 36)
    for source, target in edge_set
):
    raise RuntimeError("Invalid edge endpoint.")

print("Graphs:", len(features))
print("Node features:", features.shape)
print("Edge index:", edges.shape)
print("Spatial splits:", dict(Counter(splits)))


print("\n[2] Verifying graph pairs and spatial separation...")

by_region = defaultdict(list)

for i, region in enumerate(regions):
    by_region[region].append(i)

occupied = {}
region_boxes = {}
region_splits = {}

for region, indices in by_region.items():
    if len(indices) != 2:
        raise RuntimeError(
            f"Region lacks exactly two scanner graphs: {region}"
        )

    a, b = indices

    if set(scanners[indices]) != {"AT2", "GT450"}:
        raise RuntimeError(f"Invalid scanner pair: {region}")

    if splits[a] != splits[b]:
        raise RuntimeError(f"Pair crosses splits: {region}")

    if not np.array_equal(tiles[a], tiles[b]):
        raise RuntimeError(f"Pair tile order differs: {region}")

    if (
        not np.array_equal(xs[a], xs[b])
        or not np.array_equal(ys[a], ys[b])
    ):
        raise RuntimeError(f"Paired coordinates differ: {region}")

    x_set = sorted(set(xs[a].tolist()))
    y_set = sorted(set(ys[a].tolist()))

    if (
        len(x_set) != 6
        or len(y_set) != 6
        or x_set != list(range(x_set[0], x_set[0] + 6))
        or y_set != list(range(y_set[0], y_set[0] + 6))
    ):
        raise RuntimeError(f"Invalid spatial rectangle: {region}")

    coords = list(zip(xs[a].tolist(), ys[a].tolist()))

    if len(set(coords)) != 36:
        raise RuntimeError("Duplicate coordinates within graph.")

    for node, coordinate in enumerate(coords):
        if coordinate in occupied:
            raise RuntimeError(
                f"Tissue position reused across regions: {coordinate}"
            )

        occupied[coordinate] = region

        if tiles[a, node] != (
            f"tile_16_{coordinate[0]}_{coordinate[1]}"
        ):
            raise RuntimeError("Tile coordinate mismatch.")

    expected_edges = set()

    lookup = {
        coordinate: node
        for node, coordinate in enumerate(coords)
    }

    for node, (x, y) in enumerate(coords):
        for neighbor in (
            (x - 1, y), (x + 1, y),
            (x, y - 1), (x, y + 1),
        ):
            if neighbor in lookup:
                expected_edges.add((node, lookup[neighbor]))

    if edge_set != expected_edges:
        raise RuntimeError(
            f"Topology differs from spatial grid: {region}"
        )

    region_boxes[region] = (
        min(x_set), max(x_set),
        min(y_set), max(y_set),
    )

    region_splits[region] = str(splits[a])

# Verify the two-tile spatial buffer, rather than
# relying only on the old selection script.
names = sorted(region_boxes)

for i, first in enumerate(names):
    ax0, ax1, ay0, ay1 = region_boxes[first]

    for second in names[i + 1:]:
        bx0, bx1, by0, by1 = region_boxes[second]

        x_separated = (
            ax1 + 2 < bx0 or bx1 + 2 < ax0
        )
        y_separated = (
            ay1 + 2 < by0 or by1 + 2 < ay0
        )

        if not (x_separated or y_separated):
            raise RuntimeError(
                f"Spatial buffer violated: {first}, {second}"
            )

print("Verified matched graph pairs:", len(by_region))
print("Unique tissue positions:", len(occupied))
print("Minimum spatial buffer: 2 tiles")


print("\n[3] Preparing experimental protocol...")

protocol = {
    "phase": "4.2",
    "step": "4.2.1",
    "version": "initial_v1",
    "status": "initial_protocol_frozen_for_baselines",
    "source_graph_sha256": graph_hash,
    "source_feature_sha256": feature_hash,
    "dataset": {
        "source": "owkin/plism-dataset-tiles",
        "condition": "GIVH",
        "scanners": ["AT2", "GT450"],
        "regions": 60,
        "graph_pairs": 60,
        "graphs": 120,
        "nodes_per_graph": 36,
        "feature_dimension": 384,
        "topology": "spatial_4_neighbor",
        "directed_edges_per_graph": 120,
        "encoder": "facebook/dinov2-small",
        "encoder_training": "frozen",
        "pixel_level_registration_assumed": False,
    },
    "partition_policy": {
        "train_regions": 40,
        "validation_regions": 10,
        "test_regions": 10,
        "same_region_across_scanners_same_split": True,
        "minimum_spatial_buffer_tiles": 2,
        "train_used_for_gradient_updates": True,
        "validation_used_for_model_selection": True,
        "test_used_for_final_locked_evaluation": True,
        "test_used_for_hyperparameter_selection": False,
        "biologically_independent_test": False,
    },
    "initial_baselines": [
        "frozen_dinov2_identity",
        "mlp_projection",
        "spatial_gcn",
        "spatial_graphsage",
    ],
    "proposed_model": {
        "working_name": "stability_aware_gnn",
        "design_status": "not_implemented",
        "candidate_components": [
            "shared_scanner_encoder",
            "residual_graph_message_passing",
            "cross_scanner_representation_consistency",
            "representation_diversity_preservation",
        ],
        "loss_weights_status": "not_selected",
    },
    "evaluation": {
        "primary": "within_region_cross_scanner_recall_at_1",
        "secondary": [
            "within_region_reciprocal_rank",
            "cross_scanner_cosine_similarity",
            "cross_scanner_euclidean_distance",
            "node_representation_diversity",
            "oversmoothing_diagnostics",
        ],
        "paired_position_matching": True,
        "full_dataset_retrieval": "not_yet_evaluated",
        "uncertainty": "report_multiple_seeds_and_region_level_variation",
    },
    "xai": {
        "planned": True,
        "stage": "phase_4_2_model_evaluation",
        "folder_to_create_later": "scripts/explainable_ai",
        "methods_to_evaluate": [
            "GNNExplainer",
            "Integrated_Gradients",
            "node_edge_perturbation",
        ],
        "questions": [
            "explanation_fidelity",
            "cross_scanner_explanation_consistency",
        ],
        "implementation_status": "not_started",
    },
    "restrictions": [
        "No test-set tuning",
        "No biological-independence claims",
        "No assumption of pixel-perfect registration",
        "No final novelty or superiority claims before baselines",
        "No XAI clinical-validity claims without supporting evidence",
    ],
}

report = {
    "phase": "4.2",
    "step": "4.2.1",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "graphs_verified": 120,
    "matched_graph_pairs": 60,
    "unique_spatial_positions": len(occupied),
    "graph_tensor_shape": list(features.shape),
    "edge_index_shape": list(edges.shape),
    "graph_splits": dict(Counter(splits)),
    "region_splits": dict(Counter(region_splits.values())),
    "spatial_buffer_verified": True,
    "feature_source_sha256": feature_hash,
    "graph_archive_sha256": graph_hash,
    "biological_independence_verified": False,
    "pixel_level_registration_verified": False,
    "gnn_training_started": False,
    "xai_included_in_protocol": True,
}

save_json(CONFIG_FILE, protocol)

report["protocol_sha256"] = sha256(CONFIG_FILE)

save_json(REPORT_FILE, report)

print("\n" + "=" * 70)
print("PHASE 4.2.1 - PROTOCOL VERIFICATION COMPLETED")
print("=" * 70)

print("Graphs verified:", 120)
print("Matched graph pairs:", 60)
print("Unique spatial positions:", len(occupied))
print("Spatial buffer: VERIFIED")
print("Train/validation/test:", dict(Counter(region_splits.values())))

print("\nSaved:")
print(CONFIG_FILE)
print(REPORT_FILE)

print("\nGNN training: NOT STARTED")
print("Initial experiment protocol: SAVED")
print("Biological independence: UNVERIFIED")
print("XAI: INCLUDED IN FUTURE EVALUATION PLAN")
print("\nSTEP 4.2.1 COMPLETED")
