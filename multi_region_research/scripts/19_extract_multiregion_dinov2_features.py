
"""
PHASE 3B.19 - MULTI-REGION DINOV2 FEATURE EXTRACTION

Reproduces the Phase 1 encoder procedure:
    Model: facebook/dinov2-small
    Batch size: 8
    Processor: Hugging Face default
    Embedding: last_hidden_state[:, 0, :]
    Saved features: raw CLS embeddings, float32

Input:
    360 frozen matched tile positions
    720 validated original PNG images

Outputs:
    Feature NPZ archive
    Paired feature-similarity CSV
    Per-region feature-similarity summary
    JSON extraction and provenance report

Does not modify original PNGs or frozen manifest.
Does not overwrite an existing feature archive.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "multi_region_research"

IMAGE_ROOT = ROOT / "datasets/PLISM/phase3b_original_png"

MANIFEST = (
    BASE / "outputs/validated/"
    "phase3b_frozen_tile_manifest.csv"
)

FEATURE_DIR = BASE / "outputs/features"
TABLE_DIR = BASE / "outputs/tables"
REPORT_DIR = BASE / "outputs/reports"

for directory in (FEATURE_DIR, TABLE_DIR, REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

FEATURE_FILE = (
    FEATURE_DIR / "plism_phase3b_dinov2_features.npz"
)

PAIR_CSV = (
    TABLE_DIR / "plism_phase3b_dinov2_pair_similarity.csv"
)

REGION_CSV = (
    TABLE_DIR / "plism_phase3b_dinov2_region_summary.csv"
)

REPORT_FILE = (
    REPORT_DIR / "19_multiregion_dinov2_extraction.json"
)

EXPECTED_MANIFEST_SHA256 = (
    "69d11e1a0dd1107f2e6534632727e2c2291ff83d6b17"
    "cf953ed5056bea9020f9"
)

MODEL_NAME = "facebook/dinov2-small"
BATCH_SIZE = 8
SCANNERS = ("AT2", "GT450")

device = "cuda" if torch.cuda.is_available() else "cpu"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


# ============================================================
# STEP 1: VERIFY FROZEN MANIFEST
# ============================================================

print("=" * 70, flush=True)
print("PHASE 3B.19 - MULTI-REGION DINOV2 FEATURES", flush=True)
print("=" * 70, flush=True)

print("\n[1] Verifying frozen manifest...", flush=True)

if not MANIFEST.is_file():
    raise RuntimeError("Frozen manifest missing.")

manifest_hash = sha256(MANIFEST.read_bytes())

if manifest_hash != EXPECTED_MANIFEST_SHA256:
    raise RuntimeError(
        "Frozen manifest SHA256 mismatch."
    )

df = pd.read_csv(
    MANIFEST,
    dtype={
        "region_id": str,
        "tile_id": str,
    },
)

required = {
    "region_id", "tile_id", "level",
    "x", "y", "at2_slide_id", "gt450_slide_id",
}

if required - set(df.columns):
    raise RuntimeError(
        "Frozen manifest missing required columns."
    )

df = df.sort_values(
    ["region_id", "level", "x", "y", "tile_id"]
).reset_index(drop=True)

if len(df) != 360:
    raise RuntimeError("Expected 360 paired locations.")

if df["tile_id"].duplicated().any():
    raise RuntimeError("Duplicate tile identifiers.")

counts = df.groupby("region_id").size()

if len(counts) != 10 or not (counts == 36).all():
    raise RuntimeError(
        "Expected 10 regions of 36 positions."
    )

expected_slides = {
    "at2_slide_id": "GIVH_AT2_to_GMH_S60",
    "gt450_slide_id": "GIVH_GT450_to_GMH_S60",
}

for column, expected in expected_slides.items():
    if not (df[column] == expected).all():
        raise RuntimeError(
            f"Acquisition provenance mismatch: {column}"
        )

for row in df.itertuples(index=False):
    expected_id = (
        f"tile_{int(row.level)}_{int(row.x)}_{int(row.y)}"
    )
    if row.tile_id != expected_id:
        raise RuntimeError(
            f"Tile-coordinate mismatch: {row.tile_id}"
        )

print("Frozen manifest: VERIFIED")
print("Regions:", len(counts))
print("Matched tile positions:", len(df))


# ============================================================
# STEP 2: VERIFY ORIGINAL PNG INPUTS
# ============================================================

print("\n[2] Checking image inventory...", flush=True)

tile_ids = df["tile_id"].tolist()

for scanner in SCANNERS:

    missing = [
        tile_id
        for tile_id in tile_ids
        if not (
            IMAGE_ROOT / scanner / f"{tile_id}.png"
        ).is_file()
    ]

    if missing:
        raise RuntimeError(
            f"{scanner}: {len(missing)} images missing. "
            f"Example: {missing[:3]}"
        )

    print(
        f"{scanner}: {len(tile_ids)} image paths verified",
        flush=True,
    )

# Protect previous experimental results.
for output in (FEATURE_FILE, PAIR_CSV, REGION_CSV, REPORT_FILE):
    if output.exists():
        raise RuntimeError(
            f"Output already exists: {output}\n"
            "Refusing to overwrite previous results."
        )


# ============================================================
# STEP 3: LOAD PHASE 1 ENCODER
# ============================================================

print("\n[3] Loading DINOv2-Small...", flush=True)

print("Device:", device, flush=True)
print("Model:", MODEL_NAME, flush=True)
print("Batch size:", BATCH_SIZE, flush=True)

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
).to(device)

model.eval()

print("Model loaded successfully.", flush=True)


# ============================================================
# STEP 4: EXTRACT UNNORMALIZED CLS FEATURES
# ============================================================

print("\n[4] Extracting features...", flush=True)

features = {}

for scanner in SCANNERS:

    print(
        f"\nScanner: {scanner}",
        flush=True
    )

    scanner_features = []

    for start in tqdm(
        range(0, len(tile_ids), BATCH_SIZE),
        desc=scanner,
    ):

        batch_ids = tile_ids[
            start:start + BATCH_SIZE
        ]

        images = []

        for tile_id in batch_ids:

            path = (
                IMAGE_ROOT / scanner / f"{tile_id}.png"
            )

            with Image.open(path) as img:

                if img.format != "PNG":
                    raise RuntimeError(
                        f"Not PNG: {path}"
                    )

                if img.size != (224, 224):
                    raise RuntimeError(
                        f"Unexpected image size: {path}"
                    )

                images.append(
                    img.convert("RGB")
                )

        # Identical processing call to Phase 1.
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

            # Same CLS-token selection as Phase 1.
            embeddings = (
                outputs.last_hidden_state[:, 0, :]
            )

        batch_features = (
            embeddings.cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

        if batch_features.shape != (
            len(batch_ids), 384
        ):
            raise RuntimeError(
                "Unexpected embedding shape: "
                f"{batch_features.shape}"
            )

        if not np.isfinite(
            batch_features
        ).all():
            raise RuntimeError(
                "Non-finite feature values detected."
            )

        scanner_features.append(
            batch_features
        )

    features[scanner] = np.concatenate(
        scanner_features,
        axis=0,
    )

    print(
        f"{scanner} features:",
        features[scanner].shape,
        flush=True,
    )


# ============================================================
# STEP 5: PAIRED FEATURE SIMILARITY
# ============================================================

print("\n[5] Calculating paired similarity...", flush=True)

a = features["AT2"]
b = features["GT450"]

# Normalize ONLY for cosine similarity.
# Preserve raw CLS vectors in the NPZ archive.

a_norm = a / np.maximum(
    np.linalg.norm(a, axis=1, keepdims=True),
    1e-12,
)

b_norm = b / np.maximum(
    np.linalg.norm(b, axis=1, keepdims=True),
    1e-12,
)

cosine_similarity = np.sum(
    a_norm * b_norm,
    axis=1,
)

euclidean_distance = np.linalg.norm(
    a - b,
    axis=1,
)

pair_df = pd.DataFrame({
    "region_id": df["region_id"],
    "tile_id": df["tile_id"],
    "grid_row": df["x"],
    "grid_col": df["y"],
    "cosine_similarity": cosine_similarity,
    "euclidean_distance": euclidean_distance,
})

region_df = (
    pair_df.groupby(
        "region_id", as_index=False
    )
    .agg(
        paired_positions=("tile_id", "count"),
        mean_cosine_similarity=(
            "cosine_similarity", "mean"
        ),
        std_cosine_similarity=(
            "cosine_similarity", "std"
        ),
        min_cosine_similarity=(
            "cosine_similarity", "min"
        ),
        max_cosine_similarity=(
            "cosine_similarity", "max"
        ),
        mean_euclidean_distance=(
            "euclidean_distance", "mean"
        ),
    )
)

print(
    "Overall mean cosine similarity:",
    f"{cosine_similarity.mean():.6f}",
    flush=True,
)

print(
    "Overall mean Euclidean distance:",
    f"{euclidean_distance.mean():.6f}",
    flush=True,
)


# ============================================================
# STEP 6: SAVE FEATURES
# ============================================================

print("\n[6] Saving feature archive...", flush=True)

np.savez_compressed(
    FEATURE_FILE,
    at2=a,
    gt450=b,
    tile_ids=df["tile_id"].to_numpy(dtype=str),
    grid_rows=df["x"].to_numpy(dtype=np.int64),
    grid_cols=df["y"].to_numpy(dtype=np.int64),
    region_ids=df["region_id"].to_numpy(dtype=str),
)

pair_df.to_csv(
    PAIR_CSV, index=False
)

region_df.to_csv(
    REGION_CSV, index=False
)

# Independently reopen saved archive to verify it.
with np.load(
    FEATURE_FILE,
    allow_pickle=False,
) as saved:

    if saved["at2"].shape != (360, 384):
        raise RuntimeError(
            "Saved AT2 feature shape mismatch."
        )

    if saved["gt450"].shape != (360, 384):
        raise RuntimeError(
            "Saved GT450 feature shape mismatch."
        )

    if not np.array_equal(
        saved["tile_ids"],
        df["tile_id"].to_numpy(dtype=str),
    ):
        raise RuntimeError(
            "Saved tile ordering mismatch."
        )


# ============================================================
# STEP 7: SAVE REPRODUCIBILITY REPORT
# ============================================================

print("\n[7] Saving reproducibility report...", flush=True)

report = {
    "phase": "3B",
    "step": "3B.19",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "manifest_sha256": manifest_hash,
    "feature_archive_sha256": sha256(
        FEATURE_FILE.read_bytes()
    ),
    "model": MODEL_NAME,
    "batch_size": BATCH_SIZE,
    "device": device,
    "image_source": "original_plism_parquet_png",
    "processor": "AutoImageProcessor.from_pretrained",
    "processor_settings": processor.to_dict(),
    "model_output": "last_hidden_state[:, 0, :]",
    "model_eval": True,
    "torch_inference_mode": True,
    "stored_embeddings_l2_normalized": False,
    "embedding_dtype": "float32",
    "embedding_dimension": 384,
    "regions": 10,
    "matched_positions": 360,
    "images_processed": 720,
    "overall_mean_paired_cosine": float(
        cosine_similarity.mean()
    ),
    "overall_std_paired_cosine": float(
        cosine_similarity.std(ddof=1)
    ),
    "overall_mean_paired_euclidean": float(
        euclidean_distance.mean()
    ),
    "region_results": region_df.to_dict(
        orient="records"
    ),
    "manual_visual_review_completed": False,
    "biological_independence_verified": False,
    "original_wsi_mapping_verified": False,
    "phase1_feature_archive_modified": False,
    "frozen_manifest_modified": False,
    "limitations": [
        "Pretrained generic image encoder",
        "One registered scanner-acquisition pair",
        "Ten spatial regions are not proven independent specimens",
        "Manual visual registration assessment remains pending",
        "GPU and CPU calculations may have small numerical differences",
    ],
}

REPORT_FILE.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3B.19 - FEATURE EXTRACTION COMPLETED")
print("=" * 70)

print("Model:", MODEL_NAME)
print("Device:", device)
print("Matched positions:", len(df))
print("Regions:", len(counts))
print("AT2 feature shape:", a.shape)
print("GT450 feature shape:", b.shape)
print("Feature dtype:", a.dtype)

print(
    "Mean paired cosine:",
    f"{cosine_similarity.mean():.6f}"
)

print(
    "Mean paired Euclidean distance:",
    f"{euclidean_distance.mean():.6f}"
)

print("\nRegion-level similarity:")
print(
    region_df.to_string(index=False)
)

print("\nSaved:")
print(FEATURE_FILE)
print(PAIR_CSV)
print(REGION_CSV)
print(REPORT_FILE)

print("\nOriginal images: UNCHANGED")
print("Frozen manifest: UNCHANGED")
print("Phase 1 feature archive: UNCHANGED")
print("\nSTEP 3B.19 COMPLETED")

