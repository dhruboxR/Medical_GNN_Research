
"""
PHASE 4.1.15 - GRAPH-READY DATASET AND FEATURE BASELINES

Inputs:
    14_stage_a_dinov2_features.npz
    14_feature_metadata.csv
    14_feature_extraction_report.json

Outputs:
    15_stage_a_spatial_graphs.npz
    15_graph_inventory.csv
    15_feature_retrieval_metrics.csv
    15_graph_dataset_report.json

Construct:
    120 scanner-specific graphs
    36 nodes per graph
    384 features per node
    4-neighbor spatial grid
    60 undirected / 120 directed edges per graph

Evaluate:
    Matched cross-scanner cosine
    Nonmatched within-region cosine
    Nonmatched cross-region cosine
    AT2-to-GT450 retrieval Recall@1 and MRR

No training, checkpoint updates, or PNG modifications.
Graph topology is a provisional baseline.
"""

from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json

import numpy as np


ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "phase4_main_research"

FEATURES = BASE / "outputs/features/14_stage_a_dinov2_features.npz"
METADATA = BASE / "outputs/tables/14_feature_metadata.csv"
PREVIOUS = BASE / "outputs/reports/14_feature_extraction_report.json"

GRAPH_FILE = BASE / "outputs/features/15_stage_a_spatial_graphs.npz"
INVENTORY = BASE / "outputs/tables/15_graph_inventory.csv"
RETRIEVAL = BASE / "outputs/tables/15_feature_retrieval_metrics.csv"
REPORT = BASE / "outputs/reports/15_graph_dataset_report.json"

SCANNERS = ("AT2", "GT450")
SPLITS = ("train", "validation", "test")
DIM = 384
REGION_SIZE = 6
EXPECTED_REGIONS = 60
EXPECTED_POSITIONS = 2160
EXPECTED_GRAPHS = 120
EXPECTED_EDGES = 120


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows, fields):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def cosine_matrix(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    a_norms = np.linalg.norm(a, axis=1, keepdims=True)
    b_norms = np.linalg.norm(b, axis=1, keepdims=True)

    if np.any(a_norms <= 1e-12) or np.any(b_norms <= 1e-12):
        raise RuntimeError("Zero-length feature detected.")

    a = a / a_norms
    b = b / b_norms

    return a @ b.T


def summarize(values):
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0 or not np.isfinite(values).all():
        raise RuntimeError("Invalid metric distribution.")

    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


print("=" * 70, flush=True)
print("PHASE 4.1.15 - FEATURE AND GRAPH DATASET AUDIT", flush=True)
print("=" * 70, flush=True)


# ---------------------------------------------------------
# 1. Validate Phase 4.1.14 feature provenance
# ---------------------------------------------------------

print("\n[1] Validating source archive and metadata...", flush=True)

for path in (FEATURES, METADATA, PREVIOUS):
    if not path.is_file():
        raise RuntimeError(f"Missing source: {path}")

for path in (GRAPH_FILE, INVENTORY, RETRIEVAL, REPORT):
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite: {path}")

with PREVIOUS.open(encoding="utf-8") as stream:
    previous = json.load(stream)

if sha256(FEATURES) != previous["feature_archive_sha256"]:
    raise RuntimeError("Feature archive SHA256 mismatch.")

if sha256(METADATA) != previous["metadata_sha256"]:
    raise RuntimeError("Feature metadata SHA256 mismatch.")

metadata = read_csv(METADATA)

if len(metadata) != 4320:
    raise RuntimeError("Expected 4320 metadata rows.")

with np.load(FEATURES, allow_pickle=False) as archive:
    required = {
        "AT2", "GT450", "region_id",
        "spatial_split", "tile_id", "x", "y",
        "weight_sha256", "processor_sha256",
        "source_manifest_sha256",
    }

    if not required.issubset(archive.files):
        raise RuntimeError("Missing feature archive fields.")

    at2 = archive["AT2"].copy()
    gt450 = archive["GT450"].copy()

    region_id = archive["region_id"].astype(str)
    spatial_split = archive["spatial_split"].astype(str)
    tile_id = archive["tile_id"].astype(str)
    x = archive["x"].copy()
    y = archive["y"].copy()

    weight_hash = str(archive["weight_sha256"])
    processor_hash = str(archive["processor_sha256"])
    source_hash = str(archive["source_manifest_sha256"])

if at2.shape != (EXPECTED_POSITIONS, DIM):
    raise RuntimeError("AT2 feature shape mismatch.")

if gt450.shape != (EXPECTED_POSITIONS, DIM):
    raise RuntimeError("GT450 feature shape mismatch.")

if not np.isfinite(at2).all() or not np.isfinite(gt450).all():
    raise RuntimeError("Non-finite features.")

if weight_hash != previous["model_weight_sha256"]:
    raise RuntimeError("Model weight hash mismatch.")

if processor_hash != previous["processor_sha256"]:
    raise RuntimeError("Processor configuration mismatch.")

if source_hash != previous["source_audit_sha256"]:
    raise RuntimeError("Source dataset hash mismatch.")

if any(
    len(a) != EXPECTED_POSITIONS
    for a in (region_id, spatial_split, tile_id, x, y)
):
    raise RuntimeError("Feature metadata array length mismatch.")

# Each feature index must correspond to exactly two
# metadata rows, one from each scanner.

by_index = defaultdict(dict)

for row in metadata:
    index = int(row["feature_index"])
    scanner = row["scanner"]

    if not 0 <= index < EXPECTED_POSITIONS:
        raise RuntimeError("Feature index outside archive.")

    if scanner not in SCANNERS:
        raise RuntimeError("Unknown scanner in metadata.")

    if scanner in by_index[index]:
        raise RuntimeError("Duplicate scanner/index metadata.")

    by_index[index][scanner] = row

if len(by_index) != EXPECTED_POSITIONS:
    raise RuntimeError("Missing feature indices.")

for i in range(EXPECTED_POSITIONS):
    pair = by_index[i]

    if set(pair) != set(SCANNERS):
        raise RuntimeError("Incomplete feature metadata pair.")

    for scanner in SCANNERS:
        row = pair[scanner]

        expected = (
            str(region_id[i]),
            str(spatial_split[i]),
            str(tile_id[i]),
            int(x[i]),
            int(y[i]),
        )

        observed = (
            row["region_id"],
            row["spatial_split"],
            row["tile_id"],
            int(row["x"]),
            int(row["y"]),
        )

        if observed != expected:
            raise RuntimeError(
                f"Feature ordering mismatch at index {i}."
            )

        if row["tile_id"] != (
            f"tile_16_{row['x']}_{row['y']}"
        ):
            raise RuntimeError("Tile ID/coordinates mismatch.")

print("Verified feature positions:", EXPECTED_POSITIONS)
print("Verified feature vectors:", EXPECTED_POSITIONS * 2)
print("Encoder dimension:", DIM)


# ---------------------------------------------------------
# 2. Define 4-neighbor spatial topology
# ---------------------------------------------------------

print("\n[2] Preparing spatial reference topology...", flush=True)

edge_pairs = []

for row in range(REGION_SIZE):
    for col in range(REGION_SIZE):
        node = row * REGION_SIZE + col

        if col + 1 < REGION_SIZE:
            right = node + 1
            edge_pairs.extend([
                (node, right),
                (right, node),
            ])

        if row + 1 < REGION_SIZE:
            below = node + REGION_SIZE
            edge_pairs.extend([
                (node, below),
                (below, node),
            ])

edge_index = np.array(
    edge_pairs,
    dtype=np.int64,
).T

if edge_index.shape != (2, EXPECTED_EDGES):
    raise RuntimeError("Unexpected spatial topology.")

if len({tuple(e) for e in edge_index.T}) != EXPECTED_EDGES:
    raise RuntimeError("Duplicate graph edges.")

print("Nodes per graph:", REGION_SIZE ** 2)
print("Undirected edges:", EXPECTED_EDGES // 2)
print("Directed edges:", EXPECTED_EDGES)


# ---------------------------------------------------------
# 3. Construct all scanner-specific graphs
# ---------------------------------------------------------

print("\n[3] Building graph-ready feature arrays...", flush=True)

region_names = sorted(set(region_id))

if len(region_names) != EXPECTED_REGIONS:
    raise RuntimeError("Expected 60 regions.")

graph_features = []
graph_regions = []
graph_splits = []
graph_scanners = []
graph_tile_ids = []
graph_x = []
graph_y = []

inventory_rows = []

region_indices = {}

for name in region_names:
    indices = np.where(region_id == name)[0]

    if len(indices) != 36:
        raise RuntimeError(f"Region does not contain 36 nodes: {name}")

    # Enforce row-major ordering using actual coordinates.
    order = sorted(
        indices,
        key=lambda i: (int(y[i]), int(x[i])),
    )

    indices = np.array(order, dtype=np.int64)

    xs = sorted(set(int(x[i]) for i in indices))
    ys = sorted(set(int(y[i]) for i in indices))

    if len(xs) != 6 or len(ys) != 6:
        raise RuntimeError("Invalid spatial grid size.")

    if xs != list(range(xs[0], xs[0] + 6)):
        raise RuntimeError("Noncontiguous X coordinates.")

    if ys != list(range(ys[0], ys[0] + 6)):
        raise RuntimeError("Noncontiguous Y coordinates.")

    expected_coords = [
        (px, py)
        for py in ys
        for px in xs
    ]

    observed_coords = [
        (int(x[i]), int(y[i]))
        for i in indices
    ]

    if expected_coords != observed_coords:
        raise RuntimeError("Invalid row-major spatial ordering.")

    splits = set(spatial_split[indices])

    if len(splits) != 1:
        raise RuntimeError("Region belongs to multiple splits.")

    split = next(iter(splits))

    if split not in SPLITS:
        raise RuntimeError("Unexpected spatial split.")

    region_indices[name] = indices

    for scanner, matrix in (
        ("AT2", at2),
        ("GT450", gt450),
    ):
        graph_id = len(graph_features)

        graph_features.append(
            matrix[indices]
        )

        graph_regions.append(name)
        graph_splits.append(split)
        graph_scanners.append(scanner)
        graph_tile_ids.append(tile_id[indices])
        graph_x.append(x[indices])
        graph_y.append(y[indices])

        inventory_rows.append({
            "graph_index": graph_id,
            "region_id": name,
            "spatial_split": split,
            "scanner": scanner,
            "node_count": 36,
            "undirected_edges": 60,
            "directed_edges": 120,
            "origin_x": xs[0],
            "origin_y": ys[0],
        })

graph_features = np.stack(
    graph_features
).astype(np.float32)

graph_tile_ids = np.stack(graph_tile_ids)
graph_x = np.stack(graph_x).astype(np.int32)
graph_y = np.stack(graph_y).astype(np.int32)

if graph_features.shape != (120, 36, 384):
    raise RuntimeError("Graph feature tensor shape mismatch.")

for i in range(0, EXPECTED_GRAPHS, 2):
    if not np.array_equal(
        graph_tile_ids[i],
        graph_tile_ids[i + 1],
    ):
        raise RuntimeError("Paired graph tile order differs.")

graph_split_counts = Counter(graph_splits)

if dict(graph_split_counts) != {
    "train": 80,
    "validation": 20,
    "test": 20,
}:
    raise RuntimeError("Graph split counts mismatch.")

print("Constructed graphs:", len(graph_features))
print("Graph tensor:", graph_features.shape)
print("Matched graph pairs:", len(graph_features) // 2)


# ---------------------------------------------------------
# 4. Evaluate feature discriminability
# ---------------------------------------------------------

print("\n[4] Evaluating cross-scanner feature baselines...", flush=True)

similarity_rows = []

matched_values = []
within_region_negative_values = []
cross_region_negative_values = []

recall_at_1 = []
reciprocal_ranks = []

for name in region_names:
    indices = region_indices[name]

    a = at2[indices]
    b = gt450[indices]

    similarity = cosine_matrix(a, b)

    for local_index, global_index in enumerate(indices):
        positive = float(
            similarity[local_index, local_index]
        )

        # Deterministic nonmatching tile in the same
        # 36-tile region.
        negative_local = (
            local_index + 1
        ) % len(indices)

        within_negative = float(
            similarity[local_index, negative_local]
        )

        # Different-region negative selected by cycling
        # to the next region in sorted order.
        other_name = region_names[
            (region_names.index(name) + 1) % len(region_names)
        ]

        other_indices = region_indices[other_name]

        other_feature = gt450[
            other_indices[local_index]
        ]

        other_cosine = float(
            cosine_matrix(
                at2[global_index:global_index + 1],
                other_feature[None, :],
            )[0, 0]
        )

        # Within-region AT2 -> GT450 retrieval.
        candidates = similarity[local_index]
        rank = 1 + int(
            np.sum(candidates > candidates[local_index])
        )

        top_index = int(np.argmax(candidates))

        correct = top_index == local_index

        recall_at_1.append(correct)
        reciprocal_ranks.append(1.0 / rank)

        matched_values.append(positive)
        within_region_negative_values.append(
            within_negative
        )
        cross_region_negative_values.append(
            other_cosine
        )

        similarity_rows.append({
            "region_id": name,
            "spatial_split": str(
                spatial_split[global_index]
            ),
            "tile_id": str(tile_id[global_index]),
            "matched_cosine": positive,
            "within_region_negative_cosine":
                within_negative,
            "cross_region_negative_cosine":
                other_cosine,
            "retrieval_rank": rank,
            "retrieval_top1_correct": correct,
        })

print(
    "Median matched cosine:",
    f"{np.median(matched_values):.6f}",
)

print(
    "Median within-region negative cosine:",
    f"{np.median(within_region_negative_values):.6f}",
)

print(
    "Median cross-region negative cosine:",
    f"{np.median(cross_region_negative_values):.6f}",
)

print(
    "Within-region retrieval Recall@1:",
    f"{np.mean(recall_at_1):.4f}",
)

print(
    "Within-region retrieval MRR:",
    f"{np.mean(reciprocal_ranks):.4f}",
)


# ---------------------------------------------------------
# 5. Save verified graph-ready artifacts
# ---------------------------------------------------------

print("\n[5] Saving graph dataset and audit...", flush=True)

GRAPH_FILE.parent.mkdir(parents=True, exist_ok=True)
INVENTORY.parent.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

with GRAPH_FILE.open("xb") as stream:
    np.savez_compressed(
        stream,
        node_features=graph_features,
        edge_index=edge_index,
        region_id=np.array(graph_regions),
        spatial_split=np.array(graph_splits),
        scanner=np.array(graph_scanners),
        tile_id=graph_tile_ids,
        x=graph_x,
        y=graph_y,
        topology=np.array("spatial_4_neighbor"),
        original_feature_archive_sha256=np.array(
            sha256(FEATURES)
        ),
    )

save_csv(
    INVENTORY,
    inventory_rows,
    [
        "graph_index",
        "region_id",
        "spatial_split",
        "scanner",
        "node_count",
        "undirected_edges",
        "directed_edges",
        "origin_x",
        "origin_y",
    ],
)

save_csv(
    RETRIEVAL,
    similarity_rows,
    [
        "region_id",
        "spatial_split",
        "tile_id",
        "matched_cosine",
        "within_region_negative_cosine",
        "cross_region_negative_cosine",
        "retrieval_rank",
        "retrieval_top1_correct",
    ],
)

split_retrieval = {}

for split in SPLITS:
    subset = [
        row for row in similarity_rows
        if row["spatial_split"] == split
    ]

    split_retrieval[split] = {
        "positions": len(subset),
        "recall_at_1": float(np.mean([
            row["retrieval_top1_correct"]
            for row in subset
        ])),
        "mrr": float(np.mean([
            1.0 / row["retrieval_rank"]
            for row in subset
        ])),
        "median_matched_cosine": float(np.median([
            row["matched_cosine"]
            for row in subset
        ])),
    }

report = {
    "phase": "4",
    "step": "4.1.15",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "source_feature_archive_sha256": sha256(FEATURES),
    "source_feature_metadata_sha256": sha256(METADATA),
    "model_weight_sha256": weight_hash,
    "processor_sha256": processor_hash,
    "graphs": len(graph_features),
    "paired_graphs": len(graph_features) // 2,
    "nodes_per_graph": 36,
    "feature_dimension": DIM,
    "undirected_edges_per_graph": 60,
    "directed_edges_per_graph": EXPECTED_EDGES,
    "graph_shape": list(graph_features.shape),
    "graph_topology": "spatial_4_neighbor",
    "split_graph_counts": dict(graph_split_counts),
    "matched_cosine": summarize(matched_values),
    "within_region_negative_cosine": summarize(
        within_region_negative_values
    ),
    "cross_region_negative_cosine": summarize(
        cross_region_negative_values
    ),
    "retrieval_recall_at_1": float(
        np.mean(recall_at_1)
    ),
    "retrieval_mrr": float(
        np.mean(reciprocal_ranks)
    ),
    "split_retrieval_metrics": split_retrieval,
    "graph_archive_sha256": sha256(GRAPH_FILE),
    "inventory_sha256": sha256(INVENTORY),
    "retrieval_metrics_sha256": sha256(RETRIEVAL),
    "dataset_status": "provisional_graph_ready",
    "pixel_registration_assumed": False,
    "biological_independence_verified": False,
    "gnn_training_protocol_frozen": False,
    "xai_planned": True,
    "original_images_modified": False,
    "phase3b_modified": False,
    "limitations": [
        "Only two scanners used in Stage A",
        "The spatial 4-neighbor graph is a reference topology",
        "Retrieval negatives are deterministic examples",
        "High cosine values may reflect embedding anisotropy",
        "Spatial partitions share the same biological material",
        "No GNN training has yet occurred",
        "No independent biological validation established",
    ],
}

save_json(REPORT, report)

print("\n" + "=" * 70)
print("PHASE 4.1.15 - GRAPH DATASET COMPLETED")
print("=" * 70)

print("Graphs:", len(graph_features))
print("Node features:", graph_features.shape)
print("Edges per graph:", edge_index.shape[1])

print(
    "Retrieval Recall@1:",
    f"{report['retrieval_recall_at_1']:.4f}",
)

print(
    "Retrieval MRR:",
    f"{report['retrieval_mrr']:.4f}",
)

print("\nSaved:")
print(GRAPH_FILE)
print(INVENTORY)
print(RETRIEVAL)
print(REPORT)

print("\nGNN training: NOT STARTED")
print("Final training protocol: NOT FROZEN")
print("Original PNGs: UNCHANGED")
print("Phase 3B: UNCHANGED")
print("XAI: PLANNED FOR MODEL EVALUATION")
print("\nSTEP 4.1.15 COMPLETED")
