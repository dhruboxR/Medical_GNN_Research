
"""
PHASE 3B - STEP 3B.11B

Compare Phase 1 local PNG images against
Hugging Face Dataset Viewer images.

Tile: tile_16_100_127
Scanners: AT2 and GT450

No changes to existing PNGs or frozen sampling manifest.
"""

from pathlib import Path
from io import BytesIO
from urllib.parse import urlparse
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import requests
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "multi_region_research"

ORIGINAL_DIR = ROOT / "datasets/PLISM/GIVH_AT2_GT450"
OUTPUT_DIR = ROOT / "datasets/PLISM/phase3b_source_comparison"

REPORT_DIR = BASE / "outputs/reports"
FIGURE_DIR = BASE / "outputs/figures"

MANIFEST = (
    BASE / "outputs/validated/"
    "phase3b_frozen_tile_manifest.csv"
)

EXPECTED_MANIFEST_HASH = (
    "69d11e1a0dd1107f2e6534632727e2c2291ff83d6b17"
    "cf953ed5056bea9020f9"
)

TILE_ID = "tile_16_100_127"
DATASET = "owkin/plism-dataset-tiles"
API = "https://datasets-server.huggingface.co/filter"

SCANNERS = {
    "AT2": "GIVH_AT2_to_GMH_S60",
    "GT450": "GIVH_GT450_to_GMH_S60",
}

MAX_METADATA_BYTES = 100_000
MAX_IMAGE_BYTES = 5_000_000

for directory in (OUTPUT_DIR, REPORT_DIR, FIGURE_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_limited(response, limit):
    result = bytearray()

    for chunk in response.iter_content(16384):
        result.extend(chunk)

        if len(result) > limit:
            raise RuntimeError("Response size limit exceeded")

    return bytes(result)


def decode_image(raw):
    # Verify encoded image integrity first.
    with Image.open(BytesIO(raw)) as image:
        image_format = image.format
        image.verify()

    with Image.open(BytesIO(raw)) as image:
        image.load()
        size = image.size
        mode = image.mode

        rgb = np.asarray(
            image.convert("RGB"),
            dtype=np.uint8
        ).copy()

    return rgb, image_format, size, mode


def get_viewer_image(session, scanner):
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
        timeout=(10, 25),
        stream=True,
    ) as response:
        response.raise_for_status()
        metadata = json.loads(
            read_limited(response, MAX_METADATA_BYTES)
        )

    if (
        metadata.get("num_rows_total") != 1
        or len(metadata.get("rows", [])) != 1
    ):
        raise RuntimeError(
            f"{scanner}: exact lookup was not unique"
        )

    record = metadata["rows"][0]["row"]

    for field, expected in {
        "tile_id": TILE_ID,
        "stainer": "GIVH",
        "scanner": scanner,
        "slide_id": SCANNERS[scanner],
    }.items():
        if record.get(field) != expected:
            raise RuntimeError(
                f"{scanner}: invalid {field}"
            )

    asset = record.get("png", {})

    if (
        asset.get("width") != 224
        or asset.get("height") != 224
    ):
        raise RuntimeError("Unexpected image metadata dimensions")

    src = asset.get("src")

    if not isinstance(src, str):
        raise RuntimeError("Missing image reference")

    parsed = urlparse(src)

    if (
        parsed.scheme != "https"
        or parsed.hostname != "datasets-server.huggingface.co"
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise RuntimeError("Unexpected image host")

    with session.get(
        src,
        timeout=(10, 40),
        stream=True,
        allow_redirects=False,
    ) as response:

        if response.is_redirect:
            raise RuntimeError("Unexpected image redirect")

        response.raise_for_status()

        raw = read_limited(
            response, MAX_IMAGE_BYTES
        )

    return raw


print("=" * 70, flush=True)
print("PHASE 3B.11B - ORIGINAL VS VIEWER COMPARISON", flush=True)
print("=" * 70, flush=True)

print("\n[1] Checking frozen manifest...", flush=True)

if not MANIFEST.is_file():
    raise RuntimeError("Frozen manifest is missing")

manifest_hash = sha256(MANIFEST.read_bytes())

if manifest_hash != EXPECTED_MANIFEST_HASH:
    raise RuntimeError("Frozen manifest fingerprint changed")

print("Frozen manifest: VERIFIED")
print("Comparison tile:", TILE_ID)
print("Note: This pilot tile is outside the frozen cohort.")

results = {}
images = {}

with requests.Session() as session:

    for scanner in SCANNERS:
        print(
            f"\n[2] Comparing {scanner} images...",
            flush=True
        )

        original_path = (
            ORIGINAL_DIR / scanner / f"{TILE_ID}.png"
        )

        if not original_path.is_file():
            raise RuntimeError(
                f"Original file missing: {original_path}"
            )

        original_bytes = original_path.read_bytes()

        original, original_format, original_size, original_mode = (
            decode_image(original_bytes)
        )

        if original_format != "PNG":
            raise RuntimeError("Local source is not a PNG")

        viewer_bytes = get_viewer_image(
            session, scanner
        )

        viewer, viewer_format, viewer_size, viewer_mode = (
            decode_image(viewer_bytes)
        )

        if original_size != (224, 224):
            raise RuntimeError("Unexpected original dimensions")

        if viewer_size != original_size:
            raise RuntimeError("Image dimensions mismatch")

        if viewer_format not in ("PNG", "JPEG", "WEBP"):
            raise RuntimeError(
                f"Unsupported viewer encoding: {viewer_format}"
            )

        extension = {
            "PNG": ".png",
            "JPEG": ".jpg",
            "WEBP": ".webp",
        }[viewer_format]

        viewer_path = OUTPUT_DIR / (
            f"{TILE_ID}_{scanner}_viewer{extension}"
        )

        if viewer_path.exists():
            if sha256(viewer_path.read_bytes()) != sha256(
                viewer_bytes
            ):
                raise RuntimeError(
                    f"Viewer asset changed: {viewer_path}"
                )
        else:
            viewer_path.write_bytes(viewer_bytes)

        a = original.astype(np.float32)
        b = viewer.astype(np.float32)

        diff = np.abs(a - b)

        exact = bool(np.array_equal(original, viewer))
        mae = float(diff.mean())
        rmse = float(np.sqrt(np.mean((a - b) ** 2)))
        max_diff = int(diff.max())
        changed_fraction = float(
            np.mean(np.any(original != viewer, axis=2))
        )

        images[scanner] = {
            "original": original,
            "viewer": viewer,
        }

        results[scanner] = {
            "tile_id": TILE_ID,
            "original_file": str(original_path.relative_to(ROOT)),
            "viewer_file": str(viewer_path.relative_to(ROOT)),
            "original_format": original_format,
            "viewer_format": viewer_format,
            "original_mode": original_mode,
            "viewer_mode": viewer_mode,
            "dimensions": [224, 224],
            "original_sha256": sha256(original_bytes),
            "viewer_sha256": sha256(viewer_bytes),
            "original_bytes": len(original_bytes),
            "viewer_bytes": len(viewer_bytes),
            "decoded_rgb_exact_match": exact,
            "pixel_mae": mae,
            "pixel_rmse": rmse,
            "maximum_channel_difference": max_diff,
            "fraction_of_changed_pixels": changed_fraction,
        }

        print(f"  Original encoding: {original_format}")
        print(f"  Viewer encoding: {viewer_format}")
        print(f"  Original bytes: {len(original_bytes)}")
        print(f"  Viewer bytes: {len(viewer_bytes)}")
        print(f"  Exact RGB equality: {exact}")
        print(f"  Pixel MAE: {mae:.6f}")
        print(f"  Pixel RMSE: {rmse:.6f}")
        print(f"  Maximum difference: {max_diff}")
        print(
            "  Fraction changed pixels:",
            f"{changed_fraction:.6f}"
        )


print("\n[3] Creating comparison figure...", flush=True)

fig, axes = plt.subplots(
    2, 3,
    figsize=(12, 8),
    constrained_layout=True,
)

for row_index, scanner in enumerate(SCANNERS):
    a = images[scanner]["original"]
    b = images[scanner]["viewer"]

    difference = np.abs(
        a.astype(np.float32) -
        b.astype(np.float32)
    ).mean(axis=2)

    axes[row_index, 0].imshow(a)
    axes[row_index, 0].set_title(
        f"{scanner}: Original PNG"
    )

    axes[row_index, 1].imshow(b)
    axes[row_index, 1].set_title(
        f"{scanner}: Dataset Viewer"
    )

    axes[row_index, 2].imshow(
        difference,
        cmap="magma",
        vmin=0,
        vmax=255
    )
    axes[row_index, 2].set_title(
        f"{scanner}: Absolute RGB difference"
    )

    for ax in axes[row_index]:
        ax.axis("off")

fig.suptitle(
    f"Original vs Dataset Viewer | {TILE_ID}"
)

figure_path = (
    FIGURE_DIR /
    "phase3b_original_vs_viewer_comparison.png"
)

fig.savefig(figure_path, dpi=250)
plt.close(fig)


print("\n[4] Saving report...", flush=True)

report = {
    "phase": "3B",
    "step": "3B.11B",
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "tile_id": TILE_ID,
    "frozen_manifest_sha256": manifest_hash,
    "comparisons": results,
    "both_exact_rgb_matches": all(
        item["decoded_rgb_exact_match"]
        for item in results.values()
    ),
    "interpretation_limitations": [
        "One tile per scanner was compared",
        "Local PNG origin is based on Phase 1 file provenance",
        "Pixel differences alone do not identify their cause",
        "A match does not prove equivalence for all dataset tiles",
        "Original WSI tissue attribution remains unverified",
    ]
}

report_path = (
    REPORT_DIR /
    "11b_original_vs_viewer_comparison.json"
)

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("PHASE 3B.11B - COMPARISON COMPLETED")
print("=" * 70)

for scanner, result in results.items():
    print(
        f"{scanner}: "
        f"exact={result['decoded_rgb_exact_match']}, "
        f"MAE={result['pixel_mae']:.6f}, "
        f"RMSE={result['pixel_rmse']:.6f}"
    )

print("\nComparison figure:", figure_path)
print("Report:", report_path)
print("Frozen manifest: UNCHANGED")
print("\nSTEP 3B.11B COMPLETED")

