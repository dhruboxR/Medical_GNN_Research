
"""
PHASE 4.1.14 - STAGE A DINOV2 FEATURE EXTRACTION

Replicates the documented Phase 3B extraction procedure:
    facebook/dinov2-small
    AutoImageProcessor
    AutoModel
    RGB images
    raw last_hidden_state[:, 0, :]
    float32
    batch size 8

Inputs:
    4,320 verified original PNGs
    60 spatial regions
    AT2 / GT450 paired images

Outputs:
    14_stage_a_dinov2_features.npz
    14_feature_metadata.csv
    14_paired_feature_similarity.csv
    14_feature_extraction_report.json

Safety:
    Verify provenance and original image hashes
    Save resumable region-level checkpoints
    Preserve raw, unnormalized CLS features
    Detect missing or duplicate features
    Never overwrite completed results
    Never modify original PNG files
    Record model/processor configuration and weight hash
"""

from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
from io import BytesIO
import csv
import hashlib
import json

import numpy as np
import torch

from PIL import Image
from transformers import AutoImageProcessor, AutoModel


ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "phase4_main_research"

INPUT_CSV = (
    BASE / "outputs/tables"
    / "10_stage_a_original_png_audit.csv"
)

INPUT_REPORT = (
    BASE / "outputs/reports"
    / "10_stage_a_download_report.json"
)

CHECKPOINT_DIR = (
    BASE / "outputs/checkpoints"
    / "14_dinov2_region_checkpoints"
)

FEATURE_FILE = (
    BASE / "outputs/features"
    / "14_stage_a_dinov2_features.npz"
)

METADATA_FILE = (
    BASE / "outputs/tables"
    / "14_feature_metadata.csv"
)

SIMILARITY_FILE = (
    BASE / "outputs/tables"
    / "14_paired_feature_similarity.csv"
)

REPORT_FILE = (
    BASE / "outputs/reports"
    / "14_feature_extraction_report.json"
)

MODEL_NAME = "facebook/dinov2-small"

BATCH_SIZE = 8
FEATURE_DIM = 384

SCANNERS = ("AT2", "GT450")
SPLITS = ("train", "validation", "test")

EXPECTED_REGIONS = 60
EXPECTED_POSITIONS = 2160
EXPECTED_IMAGES = 4320

EXPECTED_SPLIT_REGIONS = {
    "train": 40,
    "validation": 10,
    "test": 10,
}

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_csv(path):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as stream:
        return list(csv.DictReader(stream))


def save_csv_new(path, rows, fields):
    with path.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def save_json_new(path, data):
    with path.open(
        "x",
        encoding="utf-8",
    ) as stream:
        json.dump(data, stream, indent=2)


def safe_image_path(relative_name):
    relative = Path(relative_name)

    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("Unsafe image destination.")

    path = (BASE / relative).resolve()

    root = (
        BASE / "datasets/expanded_plism/stage_a"
    ).resolve()

    if not path.is_relative_to(root):
        raise RuntimeError(
            "Image outside Stage A directory."
        )

    return path


def load_image_verified(record):
    path = safe_image_path(
        record["destination_relative_path"]
    )

    raw = path.read_bytes()

    if sha256_bytes(raw) != record["sha256"]:
        raise RuntimeError(
            f"Image SHA256 mismatch: {path}"
        )

    if len(raw) != int(record["size_bytes"]):
        raise RuntimeError(
            f"Unexpected image size in bytes: {path}"
        )

    with Image.open(BytesIO(raw)) as image:
        if image.format != "PNG":
            raise RuntimeError("Not an original PNG.")

        if image.size != (224, 224):
            raise RuntimeError(
                f"Unexpected dimensions: {path}"
            )

        rgb = image.convert("RGB")
        rgb.load()

    return rgb


def model_weight_sha256(model):
    """
    Hash the actual model weights to record precisely
    which checkpoint values produced these features.
    """

    digest = hashlib.sha256()

    for name, tensor in sorted(
        model.state_dict().items()
    ):
        digest.update(name.encode("utf-8"))
        array = (
            tensor.detach()
            .cpu()
            .contiguous()
            .numpy()
        )

        digest.update(
            str(array.shape).encode("utf-8")
        )
        digest.update(
            str(array.dtype).encode("utf-8")
        )
        digest.update(array.tobytes())

    return digest.hexdigest()


def processor_sha256(processor):
    config = processor.to_dict()

    content = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")

    return sha256_bytes(content)


def extract_features(records, processor, model):
    pieces = []

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]

        images = [
            load_image_verified(record)
            for record in batch
        ]

        inputs = processor(
            images=images,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = model(**inputs)

            # Exactly the Phase 3B CLS-token selection.
            embeddings = (
                outputs.last_hidden_state[:, 0, :]
            )

        result = (
            embeddings.detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

        if result.shape != (len(batch), FEATURE_DIM):
            raise RuntimeError(
                f"Unexpected feature shape: {result.shape}"
            )

        if not np.isfinite(result).all():
            raise RuntimeError("Non-finite DINOv2 features.")

        pieces.append(result)

    return np.concatenate(pieces, axis=0)


def region_checkpoint_path(region_id):
    return CHECKPOINT_DIR / f"{region_id}.npz"


def checkpoint_signature(
    region_id,
    records,
    manifest_hash,
    weight_hash,
    processor_hash,
):
    digest = hashlib.sha256()

    digest.update(region_id.encode())
    digest.update(manifest_hash.encode())
    digest.update(weight_hash.encode())
    digest.update(processor_hash.encode())

    for record in sorted(
        records,
        key=lambda r: (
            r["scanner"],
            r["tile_id"],
        ),
    ):
        digest.update(
            record["scanner"].encode()
        )
        digest.update(
            record["tile_id"].encode()
        )
        digest.update(
            record["sha256"].encode()
        )

    return digest.hexdigest()


def save_checkpoint(
    path,
    region_id,
    signature,
    tile_ids,
    at2,
    gt450,
):
    # Exclusive creation prevents overwriting a
    # previously completed research checkpoint.
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            region_id=np.array(region_id),
            signature=np.array(signature),
            tile_ids=np.array(tile_ids),
            AT2=at2,
            GT450=gt450,
        )


def load_checkpoint(path, region_id, signature, tile_ids):
    with np.load(
        path,
        allow_pickle=False,
    ) as archive:
        saved_region = str(archive["region_id"])
        saved_signature = str(archive["signature"])
        saved_ids = archive["tile_ids"].astype(str)
        at2 = archive["AT2"].copy()
        gt450 = archive["GT450"].copy()

    if saved_region != region_id:
        raise RuntimeError(
            f"Checkpoint region mismatch: {path}"
        )

    if saved_signature != signature:
        raise RuntimeError(
            f"Checkpoint provenance mismatch: {path}"
        )

    if not np.array_equal(
        saved_ids,
        np.array(tile_ids),
    ):
        raise RuntimeError(
            f"Checkpoint tile order mismatch: {path}"
        )

    for array in (at2, gt450):
        if array.shape != (36, FEATURE_DIM):
            raise RuntimeError(
                f"Unexpected checkpoint shape: {path}"
            )

        if array.dtype != np.float32:
            raise RuntimeError(
                "Checkpoint dtype mismatch."
            )

        if not np.isfinite(array).all():
            raise RuntimeError(
                "Non-finite checkpoint features."
            )

    return at2, gt450


print("=" * 70, flush=True)
print("PHASE 4.1.14 - DINOV2 FEATURE EXTRACTION", flush=True)
print("=" * 70, flush=True)


# ------------------------------------------------------------
# 1. Verify source provenance
# ------------------------------------------------------------

print("\n[1] Verifying Stage A dataset...", flush=True)

for path in (INPUT_CSV, INPUT_REPORT):
    if not path.is_file():
        raise RuntimeError(
            f"Missing input: {path}"
        )

for path in (
    FEATURE_FILE,
    METADATA_FILE,
    SIMILARITY_FILE,
    REPORT_FILE,
):
    if path.exists():
        raise RuntimeError(
            f"Refusing to overwrite: {path}"
        )

with INPUT_REPORT.open(encoding="utf-8") as stream:
    previous = json.load(stream)

manifest_hash = sha256_file(INPUT_CSV)

if manifest_hash != previous["final_audit_sha256"]:
    raise RuntimeError(
        "Source audit SHA256 mismatch."
    )

records = read_csv(INPUT_CSV)

if len(records) != EXPECTED_IMAGES:
    raise RuntimeError(
        "Expected 4320 original images."
    )

by_region = defaultdict(
    lambda: defaultdict(dict)
)

region_split = {}
seen_images = set()
split_region_counts = Counter()

for record in records:
    region = record["region_id"]
    split = record["spatial_split"]
    scanner = record["scanner"]
    tile_id = record["tile_id"]

    if split not in SPLITS or scanner not in SCANNERS:
        raise RuntimeError(
            "Unexpected split or scanner."
        )

    if region in region_split and region_split[region] != split:
        raise RuntimeError(
            "Region spans multiple partitions."
        )

    region_split[region] = split

    key = (region, scanner, tile_id)

    if key in seen_images:
        raise RuntimeError("Duplicate image record.")

    seen_images.add(key)
    by_region[region][scanner][tile_id] = record

if len(by_region) != EXPECTED_REGIONS:
    raise RuntimeError("Expected 60 regions.")

for region, scanner_data in by_region.items():
    if set(scanner_data) != set(SCANNERS):
        raise RuntimeError(
            f"Missing scanner for {region}"
        )

    if (
        set(scanner_data["AT2"])
        != set(scanner_data["GT450"])
    ):
        raise RuntimeError(
            f"Unmatched tile positions: {region}"
        )

    if len(scanner_data["AT2"]) != 36:
        raise RuntimeError(
            f"Expected 36 positions: {region}"
        )

for split in region_split.values():
    split_region_counts[split] += 1

if dict(split_region_counts) != EXPECTED_SPLIT_REGIONS:
    raise RuntimeError(
        "Spatial partition counts mismatch."
    )

print("Original images:", len(records))
print("Paired positions:", len(records) // 2)
print("Spatial regions:", len(by_region))
print("Device:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
        flush=True,
    )


# ------------------------------------------------------------
# 2. Load Phase 3B-compatible encoder
# ------------------------------------------------------------

print("\n[2] Loading DINOv2-Small...", flush=True)

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
).to(device)

model.eval()

weight_hash = model_weight_sha256(model)
proc_hash = processor_sha256(processor)

model_revision = getattr(
    model.config,
    "_commit_hash",
    None,
)

processor_revision = getattr(
    processor,
    "_commit_hash",
    None,
)

print("Model:", MODEL_NAME)
print("Batch size:", BATCH_SIZE)
print("Weight SHA256:", weight_hash)
print("Processor SHA256:", proc_hash)
print("Model revision:", model_revision)
print("Processor revision:", processor_revision)


# ------------------------------------------------------------
# 3. Extract one region at a time
# ------------------------------------------------------------

print("\n[3] Extracting paired CLS features...", flush=True)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

region_outputs = {}
new_checkpoints = 0
reused_checkpoints = 0

region_order = sorted(by_region)

for index, region in enumerate(region_order, 1):
    scanner_data = by_region[region]

    tile_ids = sorted(
        scanner_data["AT2"]
    )

    region_records = [
        scanner_data[scanner][tile]
        for scanner in SCANNERS
        for tile in tile_ids
    ]

    signature = checkpoint_signature(
        region,
        region_records,
        manifest_hash,
        weight_hash,
        proc_hash,
    )

    checkpoint = region_checkpoint_path(region)

    if checkpoint.exists():
        at2, gt450 = load_checkpoint(
            checkpoint,
            region,
            signature,
            tile_ids,
        )

        reused_checkpoints += 1

    else:
        at2_records = [
            scanner_data["AT2"][tile]
            for tile in tile_ids
        ]

        gt450_records = [
            scanner_data["GT450"][tile]
            for tile in tile_ids
        ]

        at2 = extract_features(
            at2_records,
            processor,
            model,
        )

        gt450 = extract_features(
            gt450_records,
            processor,
            model,
        )

        save_checkpoint(
            checkpoint,
            region,
            signature,
            tile_ids,
            at2,
            gt450,
        )

        new_checkpoints += 1

    region_outputs[region] = {
        "tile_ids": tile_ids,
        "AT2": at2,
        "GT450": gt450,
    }

    if index % 5 == 0 or index == len(region_order):
        print(
            f"  Regions {index}/{len(region_order)} | "
            f"New: {new_checkpoints} | "
            f"Reused: {reused_checkpoints}",
            flush=True,
        )


# ------------------------------------------------------------
# 4. Assemble reproducible paired feature arrays
# ------------------------------------------------------------

print("\n[4] Assembling paired feature archive...", flush=True)

all_at2 = []
all_gt450 = []

region_column = []
split_column = []
tile_column = []
x_column = []
y_column = []

metadata_rows = []

feature_index = 0

for region in region_order:
    result = region_outputs[region]

    tile_ids = result["tile_ids"]
    split = region_split[region]

    all_at2.append(result["AT2"])
    all_gt450.append(result["GT450"])

    for tile in tile_ids:
        parts = tile.split("_")

        if len(parts) != 4 or parts[0] != "tile":
            raise RuntimeError(
                f"Invalid tile ID: {tile}"
            )

        level, x, y = map(int, parts[1:])

        if level != 16:
            raise RuntimeError(
                "Unexpected tile level."
            )

        region_column.append(region)
        split_column.append(split)
        tile_column.append(tile)
        x_column.append(x)
        y_column.append(y)

        for scanner in SCANNERS:
            source = by_region[region][scanner][tile]

            metadata_rows.append({
                "feature_index": feature_index,
                "region_id": region,
                "spatial_split": split,
                "tile_id": tile,
                "x": x,
                "y": y,
                "scanner": scanner,
                "image_sha256": source["sha256"],
                "image_relative_path":
                    source["destination_relative_path"],
            })

        feature_index += 1

features_at2 = np.concatenate(
    all_at2,
    axis=0,
)

features_gt450 = np.concatenate(
    all_gt450,
    axis=0,
)

expected_shape = (
    EXPECTED_POSITIONS,
    FEATURE_DIM,
)

if (
    features_at2.shape != expected_shape
    or features_gt450.shape != expected_shape
):
    raise RuntimeError(
        "Unexpected assembled feature shape."
    )

if (
    not np.isfinite(features_at2).all()
    or not np.isfinite(features_gt450).all()
):
    raise RuntimeError(
        "Non-finite assembled features."
    )

if len(metadata_rows) != EXPECTED_IMAGES:
    raise RuntimeError(
        "Incorrect feature metadata count."
    )

print("AT2:", features_at2.shape)
print("GT450:", features_gt450.shape)


# ------------------------------------------------------------
# 5. Calculate descriptive paired feature similarity
# ------------------------------------------------------------

print("\n[5] Measuring paired feature similarity...", flush=True)

at2_norm = features_at2 / np.maximum(
    np.linalg.norm(
        features_at2,
        axis=1,
        keepdims=True,
    ),
    1e-12,
)

gt450_norm = features_gt450 / np.maximum(
    np.linalg.norm(
        features_gt450,
        axis=1,
        keepdims=True,
    ),
    1e-12,
)

paired_cosine = np.sum(
    at2_norm * gt450_norm,
    axis=1,
)

paired_distance = np.linalg.norm(
    features_at2 - features_gt450,
    axis=1,
)

if (
    not np.isfinite(paired_cosine).all()
    or not np.isfinite(paired_distance).all()
):
    raise RuntimeError(
        "Non-finite paired similarity values."
    )

similarity_rows = []

for index in range(EXPECTED_POSITIONS):
    similarity_rows.append({
        "feature_index": index,
        "region_id": region_column[index],
        "spatial_split": split_column[index],
        "tile_id": tile_column[index],
        "cosine_similarity": float(
            paired_cosine[index]
        ),
        "euclidean_distance": float(
            paired_distance[index]
        ),
    })

print(
    "Median paired cosine similarity:",
    f"{np.median(paired_cosine):.6f}",
)

print(
    "Median paired Euclidean distance:",
    f"{np.median(paired_distance):.6f}",
)


# ------------------------------------------------------------
# 6. Save immutable research artifacts
# ------------------------------------------------------------

print("\n[6] Saving reproducible feature outputs...", flush=True)

FEATURE_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

METADATA_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with FEATURE_FILE.open("xb") as stream:
    np.savez_compressed(
        stream,
        AT2=features_at2,
        GT450=features_gt450,
        region_id=np.array(region_column),
        spatial_split=np.array(split_column),
        tile_id=np.array(tile_column),
        x=np.array(x_column, dtype=np.int32),
        y=np.array(y_column, dtype=np.int32),
        model_name=np.array(MODEL_NAME),
        weight_sha256=np.array(weight_hash),
        processor_sha256=np.array(proc_hash),
        source_manifest_sha256=np.array(manifest_hash),
    )

save_csv_new(
    METADATA_FILE,
    metadata_rows,
    [
        "feature_index",
        "region_id",
        "spatial_split",
        "tile_id",
        "x",
        "y",
        "scanner",
        "image_sha256",
        "image_relative_path",
    ],
)

save_csv_new(
    SIMILARITY_FILE,
    similarity_rows,
    [
        "feature_index",
        "region_id",
        "spatial_split",
        "tile_id",
        "cosine_similarity",
        "euclidean_distance",
    ],
)

report = {
    "phase": "4",
    "step": "4.1.14",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "encoder": MODEL_NAME,
    "model_revision": model_revision,
    "processor_revision": processor_revision,
    "model_weight_sha256": weight_hash,
    "processor_sha256": proc_hash,
    "feature_type": "raw CLS token",
    "feature_dimension": FEATURE_DIM,
    "feature_dtype": "float32",
    "feature_normalization": "none",
    "model_eval_mode": True,
    "inference_mode": True,
    "batch_size": BATCH_SIZE,
    "device": str(device),
    "gpu_name": (
        torch.cuda.get_device_name(0)
        if device.type == "cuda"
        else None
    ),
    "pytorch_version": torch.__version__,
    "original_images": EXPECTED_IMAGES,
    "paired_positions": EXPECTED_POSITIONS,
    "spatial_regions": EXPECTED_REGIONS,
    "region_split_counts": dict(split_region_counts),
    "new_region_checkpoints": new_checkpoints,
    "reused_region_checkpoints": reused_checkpoints,
    "feature_shape_per_scanner": list(expected_shape),
    "median_paired_cosine_similarity": float(
        np.median(paired_cosine)
    ),
    "mean_paired_cosine_similarity": float(
        np.mean(paired_cosine)
    ),
    "median_paired_euclidean_distance": float(
        np.median(paired_distance)
    ),
    "source_audit_sha256": manifest_hash,
    "feature_archive_sha256": sha256_file(FEATURE_FILE),
    "metadata_sha256": sha256_file(METADATA_FILE),
    "similarity_sha256": sha256_file(SIMILARITY_FILE),
    "phase3b_modified": False,
    "original_images_modified": False,
    "pixel_registration_assumed": False,
    "biological_independence_verified": False,
    "training_protocol_frozen": False,
    "xai_planned": True,
    "next_step": (
        "Validate feature archive and paired ordering; "
        "construct graph-ready region datasets."
    ),
}

save_json_new(
    REPORT_FILE,
    report,
)

print("\n" + "=" * 70)
print("PHASE 4.1.14 - DINOV2 EXTRACTION COMPLETED")
print("=" * 70)

print("Original PNGs:", EXPECTED_IMAGES)
print("Regions:", EXPECTED_REGIONS)
print("Paired positions:", EXPECTED_POSITIONS)
print("AT2 feature shape:", features_at2.shape)
print("GT450 feature shape:", features_gt450.shape)

print(
    "Median paired cosine:",
    f"{np.median(paired_cosine):.6f}",
)

print("New checkpoints:", new_checkpoints)
print("Reused checkpoints:", reused_checkpoints)

print("\nSaved:")
print(FEATURE_FILE)
print(METADATA_FILE)
print(SIMILARITY_FILE)
print(REPORT_FILE)

print("\nOriginal images: UNCHANGED")
print("Phase 3B: UNCHANGED")
print("Final GNN training protocol: NOT FROZEN")
print("XAI integration: PLANNED")
print("\nSTEP 4.1.14 COMPLETED")
