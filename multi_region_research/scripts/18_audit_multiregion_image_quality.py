
"""
PHASE 3B.18 - MULTI-REGION IMAGE QUALITY AUDIT

Inputs:
    Frozen Phase 3B tile manifest
    Original PNG images from Step 3B.17
    Original PNG download SHA256 audit

Outputs:
    Per-image quality metrics
    Per-pair quality metrics
    Region-level summary
    Representative scanner-pair contact sheet
    JSON validation report

No network requests.
No image modification.
No changes to the frozen manifest.

IMPORTANT:
    Image-quality flags are exploratory.
    Pixel differences do not measure registration
    accuracy or establish scanner causality.
"""

from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "multi_region_research"

TABLES = BASE / "outputs/tables"
REPORTS = BASE / "outputs/reports"
FIGURES = BASE / "outputs/figures"

IMAGE_ROOT = ROOT / "datasets/PLISM/phase3b_original_png"

MANIFEST_PATH = (
    BASE / "outputs/validated/"
    "phase3b_frozen_tile_manifest.csv"
)

DOWNLOAD_AUDIT_PATH = (
    TABLES / "plism_original_png_download_audit.csv"
)

EXPECTED_MANIFEST_HASH = (
    "69d11e1a0dd1107f2e6534632727e2c2291ff83d6b17"
    "cf953ed5056bea9020f9"
)

SCANNERS = ("AT2", "GT450")
EXPECTED_SIZE = (224, 224)

for directory in (TABLES, REPORTS, FIGURES):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_png(path):
    raw = path.read_bytes()

    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError(
            f"Missing PNG signature: {path}"
        )

    with Image.open(path) as image:
        if image.format != "PNG":
            raise RuntimeError(
                f"Unexpected encoding: {path}"
            )
        image.verify()

    with Image.open(path) as image:
        image.load()

        if image.size != EXPECTED_SIZE:
            raise RuntimeError(
                f"Incorrect dimensions: {path}"
            )

        if image.mode not in ("RGB", "RGBA"):
            raise RuntimeError(
                f"Unexpected color mode: {path}"
            )

        rgb = np.asarray(
            image.convert("RGB"),
            dtype=np.uint8,
        ).copy()

    return raw, rgb


def image_metrics(rgb):
    pixels = rgb.astype(np.float32)

    # Descriptive brightness measure, not a
    # scanner calibration or clinical metric.
    gray = (
        0.299 * pixels[:, :, 0]
        + 0.587 * pixels[:, :, 1]
        + 0.114 * pixels[:, :, 2]
    )

    # Bright/white-pixel fraction is a simple
    # background proxy, not tissue segmentation.
    near_white_fraction = float(
        np.mean(np.all(pixels >= 240, axis=2))
    )

    near_black_fraction = float(
        np.mean(np.all(pixels <= 15, axis=2))
    )

    # Elementary spatial variation measure.
    horizontal_difference = np.abs(
        gray[:, 1:] - gray[:, :-1]
    )

    vertical_difference = np.abs(
        gray[1:, :] - gray[:-1, :]
    )

    texture_difference = float(
        (
            horizontal_difference.mean()
            + vertical_difference.mean()
        ) / 2
    )

    return {
        "mean_rgb": float(pixels.mean()),
        "std_rgb": float(pixels.std()),
        "mean_gray": float(gray.mean()),
        "std_gray": float(gray.std()),
        "near_white_fraction": near_white_fraction,
        "near_black_fraction": near_black_fraction,
        "texture_difference": texture_difference,
    }


def quality_flags(metrics):
    flags = []

    if metrics["near_white_fraction"] >= 0.95:
        flags.append("HIGH_NEAR_WHITE_FRACTION")

    if metrics["near_black_fraction"] >= 0.95:
        flags.append("HIGH_NEAR_BLACK_FRACTION")

    if metrics["std_gray"] < 5:
        flags.append("LOW_GRAYSCALE_VARIATION")

    if metrics["texture_difference"] < 2:
        flags.append("LOW_LOCAL_PIXEL_VARIATION")

    return flags


# ============================================================
# STEP 1: LOAD AND VERIFY INPUTS
# ============================================================

print("=" * 70, flush=True)
print("PHASE 3B.18 - MULTI-REGION IMAGE QUALITY AUDIT", flush=True)
print("=" * 70, flush=True)

print("\n[1] Validating input files...", flush=True)

if not MANIFEST_PATH.is_file():
    raise RuntimeError("Frozen manifest missing.")

if not DOWNLOAD_AUDIT_PATH.is_file():
    raise RuntimeError("Download audit missing.")

manifest_hash = sha256(
    MANIFEST_PATH.read_bytes()
)

if manifest_hash != EXPECTED_MANIFEST_HASH:
    raise RuntimeError(
        "Frozen manifest fingerprint mismatch."
    )

manifest = pd.read_csv(
    MANIFEST_PATH,
    dtype={"region_id": str, "tile_id": str},
)

audit = pd.read_csv(
    DOWNLOAD_AUDIT_PATH,
    dtype={"tile_id": str, "scanner": str},
)

if len(manifest) != 360:
    raise RuntimeError(
        "Expected 360 matched tile positions."
    )

if manifest["tile_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate tile IDs in frozen manifest."
    )

if len(audit) != 720:
    raise RuntimeError(
        f"Expected 720 audit records, found {len(audit)}."
    )

if audit.duplicated(
    ["tile_id", "scanner"]
).any():
    raise RuntimeError(
        "Duplicate tile/scanner entries in audit."
    )

required_audit_columns = {
    "tile_id", "scanner", "sha256"
}

if required_audit_columns - set(audit.columns):
    raise RuntimeError(
        "Download audit missing required fields."
    )

audit_index = {
    (str(row.tile_id), str(row.scanner)): row
    for row in audit.itertuples(index=False)
}

expected_keys = {
    (str(row.tile_id), scanner)
    for row in manifest.itertuples(index=False)
    for scanner in SCANNERS
}

if set(audit_index) != expected_keys:
    raise RuntimeError(
        "Download audit does not exactly match "
        "the frozen cohort."
    )

region_counts = manifest.groupby(
    "region_id"
).size()

if len(region_counts) != 10 or not (
    region_counts == 36
).all():
    raise RuntimeError(
        "Expected 10 regions with 36 positions each."
    )

print("Frozen manifest: VERIFIED")
print("Matched positions: 360")
print("Download audit: 720/720")
print("Region geometry counts: PASSED")


# ============================================================
# STEP 2: VALIDATE ALL PNG IMAGES
# ============================================================

print("\n[2] Validating images and SHA256 hashes...", flush=True)

image_records = []
pair_records = []
representatives = {}

for region_id, group in manifest.groupby(
    "region_id", sort=True
):

    # Use a consistent grid position near the center.
    ordered = group.sort_values(["x", "y"])
    representative_tile = str(
        ordered.iloc[len(ordered) // 2]["tile_id"]
    )

    for row in group.itertuples(index=False):

        tile_id = str(row.tile_id)
        arrays = {}

        for scanner in SCANNERS:

            path = (
                IMAGE_ROOT
                / scanner
                / f"{tile_id}.png"
            )

            if not path.is_file():
                raise RuntimeError(
                    f"Missing image: {path}"
                )

            raw, rgb = read_png(path)

            digest = sha256(raw)

            audited_hash = str(
                audit_index[
                    (tile_id, scanner)
                ].sha256
            )

            if digest != audited_hash:
                raise RuntimeError(
                    f"SHA256 mismatch: {path}"
                )

            metrics = image_metrics(rgb)
            flags = quality_flags(metrics)

            image_records.append({
                "region_id": region_id,
                "tile_id": tile_id,
                "scanner": scanner,
                "image_bytes": len(raw),
                "sha256": digest,
                **metrics,
                "quality_flag_count": len(flags),
                "quality_flags": ";".join(flags),
            })

            arrays[scanner] = rgb

        if tile_id == representative_tile:
            representatives[region_id] = {
                "tile_id": tile_id,
                "AT2": arrays["AT2"],
                "GT450": arrays["GT450"],
            }

        at2 = arrays["AT2"].astype(np.float32)
        gt450 = arrays["GT450"].astype(np.float32)

        diff = at2 - gt450

        pair_records.append({
            "region_id": region_id,
            "tile_id": tile_id,
            "pixel_mae": float(
                np.abs(diff).mean()
            ),
            "pixel_rmse": float(
                np.sqrt(np.mean(diff ** 2))
            ),
            "identical_rgb": bool(
                np.array_equal(
                    arrays["AT2"],
                    arrays["GT450"]
                )
            ),
        })

    print(
        f"  {region_id}: 36 matched pairs verified",
        flush=True
    )


# ============================================================
# STEP 3: SUMMARIZE IMAGE QUALITY
# ============================================================

print("\n[3] Computing region-level statistics...", flush=True)

images_df = pd.DataFrame(image_records)
pairs_df = pd.DataFrame(pair_records)

regions_df = (
    images_df.groupby(
        ["region_id", "scanner"], as_index=False
    )
    .agg(
        image_count=("tile_id", "count"),
        mean_brightness=("mean_gray", "mean"),
        mean_grayscale_std=("std_gray", "mean"),
        mean_near_white_fraction=(
            "near_white_fraction", "mean"
        ),
        mean_texture_difference=(
            "texture_difference", "mean"
        ),
        flagged_images=(
            "quality_flag_count",
            lambda values: int((values > 0).sum())
        ),
    )
)

pair_summary = (
    pairs_df.groupby(
        "region_id", as_index=False
    )
    .agg(
        paired_positions=("tile_id", "count"),
        mean_pixel_mae=("pixel_mae", "mean"),
        mean_pixel_rmse=("pixel_rmse", "mean"),
        max_pixel_mae=("pixel_mae", "max"),
    )
)

regions_df = regions_df.merge(
    pair_summary,
    on="region_id",
    how="left",
    validate="many_to_one",
)

print(
    "Verified images:",
    len(images_df)
)

print(
    "Verified scanner pairs:",
    len(pairs_df)
)

print(
    "Images with exploratory quality flags:",
    int((images_df["quality_flag_count"] > 0).sum())
)


# ============================================================
# STEP 4: CREATE REPRESENTATIVE CONTACT SHEET
# ============================================================

print("\n[4] Creating scanner-pair contact sheet...", flush=True)

region_ids = sorted(representatives)

fig, axes = plt.subplots(
    len(region_ids),
    2,
    figsize=(7, 3.0 * len(region_ids)),
)

for index, region_id in enumerate(region_ids):

    sample = representatives[region_id]

    for column, scanner in enumerate(SCANNERS):

        ax = axes[index, column]

        ax.imshow(sample[scanner])

        ax.set_title(
            f"{region_id} | {scanner}\n"
            f"{sample['tile_id']}",
            fontsize=9
        )

        ax.axis("off")

fig.suptitle(
    "Phase 3B: Representative Original PNG Scanner Pairs",
    fontsize=13,
    y=1.002
)

fig.tight_layout()

figure_path = (
    FIGURES /
    "phase3b_10_region_scanner_pair_contact_sheet.png"
)

fig.savefig(
    figure_path,
    dpi=180,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# STEP 5: SAVE OUTPUTS
# ============================================================

print("\n[5] Saving reports and tables...", flush=True)

images_path = (
    TABLES /
    "plism_phase3b_image_quality_metrics.csv"
)

pairs_path = (
    TABLES /
    "plism_phase3b_pair_quality_metrics.csv"
)

regions_path = (
    TABLES /
    "plism_phase3b_region_quality_summary.csv"
)

report_path = (
    REPORTS /
    "18_multiregion_image_quality_audit.json"
)

images_df.to_csv(images_path, index=False)
pairs_df.to_csv(pairs_path, index=False)
regions_df.to_csv(regions_path, index=False)

flag_counts = Counter()

for flags in images_df["quality_flags"]:
    for flag in str(flags).split(";"):
        if flag:
            flag_counts[flag] += 1

report = {
    "phase": "3B",
    "step": "3B.18",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "manifest_sha256": manifest_hash,
    "regions": len(region_ids),
    "matched_positions": len(pairs_df),
    "validated_png_images": len(images_df),
    "download_audit_hashes_verified": True,
    "scanner_pairs_complete": len(pairs_df) == 360,
    "image_dimensions": [224, 224],
    "exploratory_quality_flag_counts": dict(
        flag_counts
    ),
    "images_with_quality_flags": int(
        (images_df["quality_flag_count"] > 0).sum()
    ),
    "mean_cross_scanner_pixel_mae": float(
        pairs_df["pixel_mae"].mean()
    ),
    "mean_cross_scanner_pixel_rmse": float(
        pairs_df["pixel_rmse"].mean()
    ),
    "figure": str(
        figure_path.relative_to(ROOT)
    ),
    "images_excluded": 0,
    "manual_visual_review_completed": False,
    "registration_quality_verified": False,
    "tissue_type_verified": False,
    "biological_independence_verified": False,
    "original_wsi_mapping_verified": False,
    "limitations": [
        "Quality thresholds are exploratory",
        "No images were automatically excluded",
        "White-pixel fraction is not tissue segmentation",
        "Pixel differences do not verify registration",
        "Image hashes establish consistency with download audit",
        "Biological specimen independence is not established",
    ],
}

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3B.18 - IMAGE QUALITY AUDIT COMPLETED")
print("=" * 70)

print(f"Validated regions: {len(region_ids)}")
print(f"Validated PNG images: {len(images_df)}")
print(f"Matched scanner pairs: {len(pairs_df)}")

print(
    "Images with quality flags:",
    report["images_with_quality_flags"]
)

print(
    "Mean paired pixel MAE:",
    f"{report['mean_cross_scanner_pixel_mae']:.4f}"
)

print(
    "Mean paired pixel RMSE:",
    f"{report['mean_cross_scanner_pixel_rmse']:.4f}"
)

print("\nRegion-level results:")
print(
    regions_df[
        [
            "region_id",
            "scanner",
            "image_count",
            "mean_brightness",
            "mean_near_white_fraction",
            "flagged_images",
            "mean_pixel_mae",
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print(images_path)
print(pairs_path)
print(regions_path)
print(figure_path)
print(report_path)

print("\nOriginal PNGs modified: NONE")
print("Frozen manifest: UNCHANGED")
print("Manual visual review: PENDING")
print("Biological independence: UNVERIFIED")

print("\nSTEP 3B.18 COMPLETED")
