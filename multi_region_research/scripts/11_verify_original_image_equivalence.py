
"""
PHASE 3B - STEP 3B.11
Original Source vs Dataset Viewer Image Equivalence

Compare one matched tile for each scanner:
    Original Phase 1 PNG
    Dataset Viewer JPEG

Checks:
    - Exact tile ID and scanner
    - Dimensions and image format
    - Decoded RGB pixel equality
    - Pixel MAE / RMSE / maximum difference
    - File SHA256 and provenance paths

No network requests.
No image modifications.
No changes to the frozen manifest.
"""

from pathlib import Path
from datetime import datetime, timezone
from PIL import Image
import hashlib
import json

import numpy as np


ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"

TILE_ID = "tile_16_105_76"
SCANNERS = ("AT2", "GT450")

ORIGINAL_ROOT = (
    ROOT / "datasets/PLISM/GIVH_AT2_GT450"
)

VIEWER_ROOT = (
    ROOT / "datasets/PLISM/phase3b_pilot_validation"
)

REPORT_PATH = (
    ROOT / "multi_region_research/outputs/reports/"
    "11_original_image_equivalence.json"
)

MANIFEST = (
    ROOT / "multi_region_research/outputs/validated/"
    "phase3b_frozen_tile_manifest.csv"
)

EXPECTED_MANIFEST_SHA256 = (
    "69d11e1a0dd1107f2e6534632727e2c2291ff83d6b17"
    "cf953ed5056bea9020f9"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_image(path):
    with Image.open(path) as image:
        image.load()

        image_format = image.format
        image_size = image.size
        original_mode = image.mode

        pixels = np.asarray(
            image.convert("RGB"),
            dtype=np.uint8
        ).copy()

    return {
        "format": image_format,
        "size": image_size,
        "mode": original_mode,
        "pixels": pixels,
    }


def find_original(scanner):
    scanner_dir = ORIGINAL_ROOT / scanner

    if not scanner_dir.is_dir():
        return [], scanner_dir

    # Match the complete tile filename stem only.
    candidates = [
        path
        for path in scanner_dir.rglob("*")
        if (
            path.is_file()
            and path.stem == TILE_ID
            and path.suffix.lower() == ".png"
        )
    ]

    return sorted(candidates), scanner_dir


print("=" * 70)
print("PHASE 3B - ORIGINAL IMAGE EQUIVALENCE")
print("=" * 70)

print("\n[1] Verifying research manifest...")

if not MANIFEST.is_file():
    raise RuntimeError("Frozen manifest not found.")

if sha256(MANIFEST) != EXPECTED_MANIFEST_SHA256:
    raise RuntimeError("Manifest SHA256 mismatch.")

print("Frozen manifest: VERIFIED")

results = {}

for scanner in SCANNERS:
    print(f"\n[2] Scanner: {scanner}")

    viewer_candidates = [
        VIEWER_ROOT / f"{TILE_ID}_{scanner}.jpg",
        VIEWER_ROOT / f"{TILE_ID}_{scanner}.jpeg",
        VIEWER_ROOT / f"{TILE_ID}_{scanner}.png",
        VIEWER_ROOT / f"{TILE_ID}_{scanner}.webp",
    ]

    viewer_found = [
        p for p in viewer_candidates if p.is_file()
    ]

    if len(viewer_found) != 1:
        raise RuntimeError(
            f"{scanner}: expected one viewer image, "
            f"found {len(viewer_found)}"
        )

    viewer_path = viewer_found[0]
    original_candidates, searched_dir = find_original(
        scanner
    )

    print("Viewer image:", viewer_path)
    print("Original directory:", searched_dir)
    print(
        "Exact original PNG candidates:",
        len(original_candidates)
    )

    if len(original_candidates) != 1:
        status = (
            "ORIGINAL_NOT_FOUND"
            if not original_candidates
            else "AMBIGUOUS_ORIGINAL_MATCH"
        )

        results[scanner] = {
            "status": status,
            "viewer_path": str(viewer_path),
            "original_candidates": [
                str(p) for p in original_candidates
            ],
        }

        print("Status:", status)
        continue

    original_path = original_candidates[0]

    original = read_image(original_path)
    viewer = read_image(viewer_path)

    print("Original format:", original["format"])
    print("Viewer format:", viewer["format"])
    print("Original dimensions:", original["size"])
    print("Viewer dimensions:", viewer["size"])

    if original["size"] != viewer["size"]:
        results[scanner] = {
            "status": "DIMENSION_MISMATCH",
            "original_path": str(original_path),
            "viewer_path": str(viewer_path),
        }
        print("Status: DIMENSION_MISMATCH")
        continue

    a = original["pixels"].astype(np.float32)
    b = viewer["pixels"].astype(np.float32)

    difference = np.abs(a - b)

    mae = float(difference.mean())
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    maximum = float(difference.max())

    exact_match = bool(
        np.array_equal(
            original["pixels"],
            viewer["pixels"]
        )
    )

    results[scanner] = {
        "status": "COMPARED",
        "original_path": str(original_path),
        "viewer_path": str(viewer_path),
        "original_format": original["format"],
        "viewer_format": viewer["format"],
        "dimensions": list(original["size"]),
        "original_sha256": sha256(original_path),
        "viewer_sha256": sha256(viewer_path),
        "decoded_rgb_exact_match": exact_match,
        "pixel_mae": mae,
        "pixel_rmse": rmse,
        "maximum_channel_difference": maximum,
    }

    print("Exact decoded RGB match:", exact_match)
    print(f"Original vs Viewer MAE: {mae:.6f}")
    print(f"Original vs Viewer RMSE: {rmse:.6f}")
    print(f"Maximum channel difference: {maximum:.0f}")


print("\n[3] Saving comparison report...")

REPORT_PATH.parent.mkdir(
    parents=True, exist_ok=True
)

report = {
    "phase": "3B",
    "step": "3B.11",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "tile_id": TILE_ID,
    "manifest_sha256": EXPECTED_MANIFEST_SHA256,
    "scanner_results": results,
    "all_scanners_compared": all(
        results[s]["status"] == "COMPARED"
        for s in SCANNERS
    ),
    "original_source_provenance_assumption": (
        "Files under the Phase 1 dataset directory "
        "are treated as original PNG candidates. "
        "Their download provenance must be confirmed "
        "from Phase 1 acquisition records."
    ),
}

REPORT_PATH.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("PHASE 3B.11 - COMPARISON FINISHED")
print("=" * 70)

for scanner in SCANNERS:
    r = results[scanner]

    print(f"\n{scanner}: {r['status']}")

    if r["status"] == "COMPARED":
        print(
            "  Exact RGB equality:",
            r["decoded_rgb_exact_match"]
        )
        print(
            "  Pixel MAE:",
            f"{r['pixel_mae']:.6f}"
        )

print("\nReport:")
print(REPORT_PATH)

print("\nNo images downloaded.")
print("Frozen manifest unchanged.")
