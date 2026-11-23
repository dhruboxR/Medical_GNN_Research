
"""
PHASE 3B - STEP 3B.10 (CORRECTED)
Paired Dataset Viewer Image Validation

Fix:
    Detect actual encoded image format.
    Do not assume the 'png' metadata field contains PNG bytes.

Downloads exactly one AT2/GT450 pair.

Checks:
    - Frozen manifest fingerprint
    - Exact scanner/stainer/tile metadata
    - Actual image encoding and integrity
    - RGB image dimensions
    - Image hashes
    - Cross-scanner pixel differences

IMPORTANT:
    Dataset Viewer images are NOT assumed to be
    byte-identical to original source PNG tiles.
"""

from pathlib import Path
from urllib.parse import urlparse
from io import BytesIO
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd
import requests
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "multi_region_research"

MANIFEST = (
    BASE / "outputs/validated/"
    "phase3b_frozen_tile_manifest.csv"
)

IMAGE_DIR = (
    ROOT / "datasets/PLISM/"
    "phase3b_pilot_validation"
)

REPORT_DIR = BASE / "outputs/reports"
FIGURE_DIR = BASE / "outputs/figures"

for path in (IMAGE_DIR, REPORT_DIR, FIGURE_DIR):
    path.mkdir(parents=True, exist_ok=True)

EXPECTED_MANIFEST_SHA256 = (
    "69d11e1a0dd1107f2e6534632727e2c2291ff83d6b17"
    "cf953ed5056bea9020f9"
)

API = "https://datasets-server.huggingface.co/filter"
DATASET = "owkin/plism-dataset-tiles"

TILE_ID = "tile_16_105_76"
EXPECTED_SIZE = (224, 224)

ACQUISITIONS = {
    "AT2": "GIVH_AT2_to_GMH_S60",
    "GT450": "GIVH_GT450_to_GMH_S60",
}

MAX_METADATA_BYTES = 100_000
MAX_IMAGE_BYTES = 5 * 1024 * 1024

FORMAT_EXTENSIONS = {
    "PNG": ".png",
    "JPEG": ".jpg",
    "WEBP": ".webp",
}


# ============================================================
# HELPERS
# ============================================================

def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_limited(response, max_bytes):
    data = bytearray()

    for chunk in response.iter_content(16384):
        data.extend(chunk)

        if len(data) > max_bytes:
            raise RuntimeError(
                f"Response exceeded {max_bytes} bytes"
            )

    return bytes(data)


def fetch_metadata(session, scanner):
    where = (
        f'"tile_id"=\'{TILE_ID}\' '
        f'AND "stainer"=\'GIVH\' '
        f'AND "scanner"=\'{scanner}\''
    )

    params = {
        "dataset": DATASET,
        "config": "default",
        "split": "train",
        "where": where,
        "offset": 0,
        "length": 2,
    }

    with session.get(
        API,
        params=params,
        stream=True,
        timeout=(10, 25),
    ) as response:

        response.raise_for_status()

        raw = read_limited(
            response, MAX_METADATA_BYTES
        )

    data = json.loads(raw)

    if int(data.get("num_rows_total", -1)) != 1:
        raise RuntimeError(
            f"{scanner}: expected one matching record"
        )

    rows = data.get("rows", [])

    if len(rows) != 1:
        raise RuntimeError(
            f"{scanner}: unexpected row count"
        )

    record = rows[0]["row"]

    expected = {
        "tile_id": TILE_ID,
        "stainer": "GIVH",
        "scanner": scanner,
        "slide_id": ACQUISITIONS[scanner],
    }

    for field, value in expected.items():
        if record.get(field) != value:
            raise RuntimeError(
                f"{scanner}: {field} mismatch"
            )

    asset = record.get("png")

    if not isinstance(asset, dict):
        raise RuntimeError(
            f"{scanner}: missing image metadata"
        )

    if (
        asset.get("width") != EXPECTED_SIZE[0]
        or asset.get("height") != EXPECTED_SIZE[1]
    ):
        raise RuntimeError(
            f"{scanner}: unexpected metadata dimensions"
        )

    src = asset.get("src")

    if not isinstance(src, str) or not src:
        raise RuntimeError(
            f"{scanner}: missing image URL"
        )

    parsed = urlparse(src)

    if (
        parsed.scheme != "https"
        or parsed.hostname
        != "datasets-server.huggingface.co"
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise RuntimeError(
            f"{scanner}: unexpected image URL"
        )

    return src


def fetch_and_validate_image(session, src, scanner):
    with session.get(
        src,
        stream=True,
        timeout=(10, 40),
        allow_redirects=False,
    ) as response:

        if response.is_redirect:
            raise RuntimeError(
                f"{scanner}: unexpected redirect"
            )

        response.raise_for_status()

        raw = read_limited(
            response, MAX_IMAGE_BYTES
        )

        content_type = response.headers.get(
            "Content-Type", ""
        )

    # Use actual decoder detection rather than
    # the HTTP content type or metadata field name.
    with Image.open(BytesIO(raw)) as im:
        image_format = im.format
        im.verify()

    if image_format not in FORMAT_EXTENSIONS:
        raise RuntimeError(
            f"{scanner}: unsupported encoding "
            f"{image_format}"
        )

    with Image.open(BytesIO(raw)) as im:
        im.load()

        actual_size = im.size
        original_mode = im.mode

        if actual_size != EXPECTED_SIZE:
            raise RuntimeError(
                f"{scanner}: wrong dimensions "
                f"{actual_size}"
            )

        if original_mode not in ("RGB", "RGBA"):
            raise RuntimeError(
                f"{scanner}: unexpected mode "
                f"{original_mode}"
            )

        rgb = np.asarray(
            im.convert("RGB"),
            dtype=np.uint8
        ).copy()

    extension = FORMAT_EXTENSIONS[image_format]

    destination = (
        IMAGE_DIR / f"{TILE_ID}_{scanner}{extension}"
    )

    # Avoid leaving an incorrectly named file around
    # if an earlier attempt saved different encoding.
    for other_extension in FORMAT_EXTENSIONS.values():
        other = IMAGE_DIR / (
            f"{TILE_ID}_{scanner}{other_extension}"
        )

        if other != destination and other.exists():
            raise RuntimeError(
                f"Conflicting image file exists: {other}"
            )

    digest = sha256(raw)

    if destination.exists():
        existing_digest = sha256(
            destination.read_bytes()
        )

        if existing_digest != digest:
            raise RuntimeError(
                "Previously saved image differs: "
                f"{destination}"
            )
    else:
        destination.write_bytes(raw)

    pixels = rgb.astype(np.float32)

    stats = {
        "scanner": scanner,
        "tile_id": TILE_ID,
        "source_type": "huggingface_dataset_viewer",
        "actual_encoding": image_format,
        "http_content_type": content_type,
        "width": actual_size[0],
        "height": actual_size[1],
        "original_color_mode": original_mode,
        "stored_bytes": len(raw),
        "sha256": digest,
        "mean_pixel_intensity": float(
            pixels.mean()
        ),
        "pixel_standard_deviation": float(
            pixels.std()
        ),
        "relative_path": str(
            destination.relative_to(ROOT)
        ),
        "original_source_bytes_verified": False,
    }

    return rgb, stats


# ============================================================
# STEP 1: FROZEN MANIFEST VALIDATION
# ============================================================

print("=" * 70, flush=True)
print("PHASE 3B - CORRECTED IMAGE VALIDATION", flush=True)
print("=" * 70, flush=True)

print("\n[1] Checking frozen manifest...", flush=True)

if not MANIFEST.is_file():
    raise RuntimeError(
        f"Missing manifest: {MANIFEST}"
    )

manifest_hash = sha256(
    MANIFEST.read_bytes()
)

if manifest_hash != EXPECTED_MANIFEST_SHA256:
    raise RuntimeError(
        "Frozen manifest hash mismatch"
    )

manifest = pd.read_csv(MANIFEST)

selected = manifest[
    manifest["tile_id"] == TILE_ID
]

if len(selected) != 1:
    raise RuntimeError(
        "Pilot tile missing or duplicated"
    )

region_id = str(
    selected.iloc[0]["region_id"]
)

for scanner, column in [
    ("AT2", "at2_slide_id"),
    ("GT450", "gt450_slide_id"),
]:
    if selected.iloc[0][column] != ACQUISITIONS[scanner]:
        raise RuntimeError(
            "Frozen manifest acquisition mismatch"
        )

print("Manifest SHA256: VERIFIED")
print("Region:", region_id)
print("Tile:", TILE_ID)


# ============================================================
# STEP 2: RETRIEVE EXACT IMAGE PAIR
# ============================================================

print("\n[2] Retrieving scanner pair...", flush=True)

arrays = {}
statistics = []

with requests.Session() as session:

    for scanner in ACQUISITIONS:

        print(
            f"\nScanner: {scanner}",
            flush=True
        )

        src = fetch_metadata(
            session, scanner
        )

        pixels, stats = fetch_and_validate_image(
            session, src, scanner
        )

        arrays[scanner] = pixels
        statistics.append(stats)

        print(
            "  Detected encoding:",
            stats["actual_encoding"]
        )
        print(
            "  Image size:",
            f"{stats['width']} x {stats['height']}"
        )
        print(
            "  Bytes:",
            stats["stored_bytes"]
        )
        print(
            "  SHA256:",
            stats["sha256"]
        )
        print(
            "  Image integrity: PASSED"
        )


# ============================================================
# STEP 3: CROSS-SCANNER PIXEL METRICS
# ============================================================

print("\n[3] Computing pixel statistics...", flush=True)

at2 = arrays["AT2"].astype(np.float32)
gt450 = arrays["GT450"].astype(np.float32)

difference = np.abs(at2 - gt450)

mae = float(difference.mean())
rmse = float(
    np.sqrt(
        np.mean((at2 - gt450) ** 2)
    )
)

identical_pixels = bool(
    np.array_equal(
        arrays["AT2"],
        arrays["GT450"]
    )
)

print(f"Pixel MAE: {mae:.4f}")
print(f"Pixel RMSE: {rmse:.4f}")
print("Identical decoded pixels:", identical_pixels)

# These are raw pixel differences, not registration
# scores or scanner-specific causal effects.


# ============================================================
# STEP 4: COMPARISON FIGURE
# ============================================================

print("\n[4] Generating comparison figure...", flush=True)

fig, axes = plt.subplots(
    1, 3,
    figsize=(12, 4)
)

axes[0].imshow(
    arrays["AT2"]
)
axes[0].set_title("AT2")

axes[1].imshow(
    arrays["GT450"]
)
axes[1].set_title("GT450")

axes[2].imshow(
    difference.mean(axis=2),
    cmap="magma",
    vmin=0,
    vmax=255,
)
axes[2].set_title(
    "Mean absolute RGB difference"
)

for ax in axes:
    ax.axis("off")

fig.suptitle(
    f"PLISM Dataset Viewer Pair: {TILE_ID}"
)

fig.tight_layout()

figure_path = (
    FIGURE_DIR /
    "phase3b_first_pair_validation.png"
)

fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# STEP 5: REPORT
# ============================================================

print("\n[5] Saving validation report...", flush=True)

report = {
    "phase": "3B",
    "step": "3B.10",
    "status": "completed",
    "timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "manifest_sha256": manifest_hash,
    "region_id": region_id,
    "tile_id": TILE_ID,
    "image_count": len(statistics),
    "image_statistics": statistics,
    "pixel_mae": mae,
    "pixel_rmse": rmse,
    "decoded_pixels_identical": identical_pixels,
    "image_integrity_verified": True,
    "scanner_metadata_verified": True,
    "original_source_image_bytes_verified": False,
    "original_png_equivalence_verified": False,
    "visual_correspondence_manually_verified": False,
    "remaining_images_not_retrieved": 718,
    "limitations": [
        "Only one paired tile was examined",
        "Dataset Viewer encoding may differ from source",
        "JPEG is a potentially lossy encoding",
        "Raw pixel differences do not verify registration",
        "Biological specimen independence unverified",
    ],
}

report_path = (
    REPORT_DIR /
    "10_paired_image_pixel_validation.json"
)

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3B.10 - IMAGE VALIDATION COMPLETED")
print("=" * 70)

print("Validated scanner pairs: 1")
print("Validated images: 2")

for stats in statistics:
    print(
        f"{stats['scanner']}: "
        f"{stats['actual_encoding']} | "
        f"{stats['stored_bytes']} bytes"
    )

print(f"Pixel MAE: {mae:.4f}")
print(f"Pixel RMSE: {rmse:.4f}")

print("\nSaved images:")
for stats in statistics:
    print(stats["relative_path"])

print("\nComparison figure:")
print(figure_path)

print("\nReport:")
print(report_path)

print("\nOriginal source bytes: UNVERIFIED")
print("Manual visual correspondence: PENDING")
print("Remaining images: 718")

print("\nSTEP 3B.10 COMPLETED")
