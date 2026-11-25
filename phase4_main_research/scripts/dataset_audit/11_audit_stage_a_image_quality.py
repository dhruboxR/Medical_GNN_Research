
"""
PHASE 4.1.11 - STAGE A IMAGE QUALITY AUDIT

Inputs:
    4,320 original PLISM PNG images
    Phase 4.1.10 provenance CSV and report

Outputs:
    Per-image quality metrics CSV
    Per-pair similarity metrics CSV
    Three paired contact sheets (train/validation/test)
    Quality-audit JSON report

Read-only access to original images.

Important:
    Quality metrics are descriptive, not diagnostic.
    Pixel correlation does not establish registration.
    No images are rejected automatically.
    No evaluation samples are used to fit a model.
"""

from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
from io import BytesIO
import csv
import hashlib
import json
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "phase4_main_research"

AUDIT = BASE / "outputs/tables/10_stage_a_original_png_audit.csv"
PREVIOUS = BASE / "outputs/reports/10_stage_a_download_report.json"

IMAGE_METRICS = BASE / "outputs/tables/11_image_quality_metrics.csv"
PAIR_METRICS = BASE / "outputs/tables/11_scanner_pair_metrics.csv"
REPORT = BASE / "outputs/reports/11_image_quality_report.json"

FIGURES = BASE / "outputs/figures"
SPLITS = ("train", "validation", "test")
SCANNERS = ("AT2", "GT450")
PATTERN = re.compile(r"^tile_16_(-?\d+)_(-?\d+)$")

THUMB = 112
GRID = 6
MARGIN = 22
TOP = 86
GAP = 32

IMAGE_FIELDS = [
    "region_id", "spatial_split", "scanner", "tile_id",
    "sha256", "size_bytes", "mean_gray", "std_gray",
    "near_white_fraction", "near_black_fraction",
    "edge_variance",
]

PAIR_FIELDS = [
    "region_id", "spatial_split", "tile_id",
    "gray_correlation", "gray_mean_absolute_difference",
]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def safe_image_path(name):
    relative = Path(name)

    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe image path: {name}")

    destination = (BASE / relative).resolve()
    allowed = (BASE / "datasets/expanded_plism/stage_a").resolve()

    if not destination.is_relative_to(allowed):
        raise RuntimeError("Image path leaves Stage A directory.")

    return destination


def tile_xy(tile_id):
    match = PATTERN.fullmatch(tile_id)

    if match is None:
        raise RuntimeError(f"Unexpected tile identifier: {tile_id}")

    return tuple(map(int, match.groups()))


def describe(values):
    values = np.asarray(values, dtype=np.float64)

    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def gray_correlation(first, second):
    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()

    a -= a.mean()
    b -= b.mean()

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator < 1e-12:
        return None

    return float(np.dot(a, b) / denominator)


print("=" * 70, flush=True)
print("PHASE 4.1.11 - IMAGE QUALITY AND PAIR REVIEW", flush=True)
print("=" * 70, flush=True)

print("\n[1] Validating source provenance...", flush=True)

for path in (AUDIT, PREVIOUS):
    if not path.is_file():
        raise RuntimeError(f"Missing input: {path}")

with PREVIOUS.open(encoding="utf-8") as stream:
    previous = json.load(stream)

if hashlib.sha256(AUDIT.read_bytes()).hexdigest() != previous[
    "final_audit_sha256"
]:
    raise RuntimeError("Phase 4.1.10 audit hash mismatch.")

rows = read_csv(AUDIT)

if len(rows) != 4320:
    raise RuntimeError(f"Expected 4320 records, got {len(rows)}.")

selected_figures = {
    split: FIGURES / f"11_{split}_at2_gt450_contact_sheet.png"
    for split in SPLITS
}

outputs = [IMAGE_METRICS, PAIR_METRICS, REPORT, *selected_figures.values()]

for path in outputs:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite: {path}")

by_pair = defaultdict(dict)
by_region = defaultdict(list)
seen_paths = set()

for row in rows:
    if row["spatial_split"] not in SPLITS:
        raise RuntimeError("Unexpected split.")

    if row["scanner"] not in SCANNERS:
        raise RuntimeError("Unexpected scanner.")

    tile_xy(row["tile_id"])

    key = (row["region_id"], row["tile_id"])
    scanner = row["scanner"]

    if scanner in by_pair[key]:
        raise RuntimeError(f"Duplicate scanner/tile: {key}")

    destination = safe_image_path(row["destination_relative_path"])

    if destination in seen_paths:
        raise RuntimeError("Duplicate image path.")

    seen_paths.add(destination)
    by_pair[key][scanner] = row
    by_region[row["region_id"]].append(row)

if len(by_pair) != 2160:
    raise RuntimeError("Unexpected number of paired positions.")

if any(set(pair) != set(SCANNERS) for pair in by_pair.values()):
    raise RuntimeError("Incomplete AT2/GT450 pairing.")

print("Images:", len(rows))
print("Matched pairs:", len(by_pair))
print("Regions:", len(by_region))


print("\n[2] Measuring image quality and rechecking integrity...", flush=True)

quality_rows = []
metric_lookup = {}
preview_cache = {}

for index, row in enumerate(rows, 1):
    destination = safe_image_path(row["destination_relative_path"])
    raw = destination.read_bytes()

    if sha256_bytes(raw) != row["sha256"]:
        raise RuntimeError(f"SHA256 mismatch: {destination}")

    if len(raw) != int(row["size_bytes"]):
        raise RuntimeError(f"File size mismatch: {destination}")

    with Image.open(BytesIO(raw)) as image:
        if image.format != "PNG" or image.size != (224, 224):
            raise RuntimeError(f"Unexpected PNG: {destination}")

        rgb_image = image.convert("RGB")
        rgb = np.asarray(rgb_image, dtype=np.uint8)

        gray_image = rgb_image.convert("L")
        gray = np.asarray(gray_image, dtype=np.float32)

        edge = np.asarray(
            gray_image.filter(ImageFilter.FIND_EDGES),
            dtype=np.float32,
        )

    metrics = {
        "region_id": row["region_id"],
        "spatial_split": row["spatial_split"],
        "scanner": row["scanner"],
        "tile_id": row["tile_id"],
        "sha256": row["sha256"],
        "size_bytes": len(raw),
        "mean_gray": float(gray.mean()),
        "std_gray": float(gray.std()),
        "near_white_fraction": float(
            np.all(rgb >= 235, axis=2).mean()
        ),
        "near_black_fraction": float(
            np.all(rgb <= 20, axis=2).mean()
        ),
        "edge_variance": float(edge.var()),
    }

    quality_rows.append(metrics)
    metric_lookup[(row["scanner"], row["tile_id"])] = gray

    if index % 500 == 0 or index == len(rows):
        print(
            f"  Validated {index}/{len(rows)} images",
            flush=True,
        )


print("\n[3] Measuring matched-pair image similarity...", flush=True)

pair_rows = []

for (region_id, tile_id), scanner_records in sorted(by_pair.items()):
    at2 = metric_lookup[("AT2", tile_id)]
    gt450 = metric_lookup[("GT450", tile_id)]

    # Downsample to make these global similarity metrics
    # less sensitive to tiny pixel-level details.
    a = np.asarray(
        Image.fromarray(at2.astype(np.uint8)).resize(
            (64, 64), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    b = np.asarray(
        Image.fromarray(gt450.astype(np.uint8)).resize(
            (64, 64), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )

    pair_rows.append({
        "region_id": region_id,
        "spatial_split": scanner_records["AT2"]["spatial_split"],
        "tile_id": tile_id,
        "gray_correlation": gray_correlation(a, b),
        "gray_mean_absolute_difference": float(
            np.mean(np.abs(a - b))
        ),
    })

print("Paired comparisons:", len(pair_rows))


print("\n[4] Creating scanner-paired contact sheets...", flush=True)

FIGURES.mkdir(parents=True, exist_ok=True)

font = ImageFont.load_default()
panel_width = GRID * THUMB
width = 2 * panel_width + GAP + 2 * MARGIN
height = TOP + GRID * THUMB + MARGIN

selected_regions = {}

for split in SPLITS:
    candidates = sorted({
        row["region_id"]
        for row in rows
        if row["spatial_split"] == split
    })

    if not candidates:
        raise RuntimeError(f"No regions for split: {split}")

    region_id = candidates[0]
    selected_regions[split] = region_id

    region_rows = by_region[region_id]

    if len(region_rows) != 72:
        raise RuntimeError(f"Invalid region: {region_id}")

    x_values = sorted({
        tile_xy(r["tile_id"])[0]
        for r in region_rows
    })
    y_values = sorted({
        tile_xy(r["tile_id"])[1]
        for r in region_rows
    })

    if (
        len(x_values) != 6
        or len(y_values) != 6
        or x_values != list(range(x_values[0], x_values[0] + 6))
        or y_values != list(range(y_values[0], y_values[0] + 6))
    ):
        raise RuntimeError(f"Invalid grid coordinates: {region_id}")

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text(
        (MARGIN, 14),
        f"{split.upper()} | {region_id}",
        fill="black",
        font=font,
    )
    draw.text(
        (MARGIN, 45),
        "AT2 scanner",
        fill="black",
        font=font,
    )
    draw.text(
        (MARGIN + panel_width + GAP, 45),
        "GT450 scanner",
        fill="black",
        font=font,
    )

    for row in region_rows:
        x, y = tile_xy(row["tile_id"])

        col = x - x_values[0]
        grid_row = y - y_values[0]

        scanner_offset = (
            0 if row["scanner"] == "AT2"
            else panel_width + GAP
        )

        left = MARGIN + scanner_offset + col * THUMB
        top = TOP + grid_row * THUMB

        destination = safe_image_path(
            row["destination_relative_path"]
        )

        with Image.open(destination) as img:
            thumb = img.convert("RGB").resize(
                (THUMB, THUMB), Image.Resampling.BILINEAR
            )

            canvas.paste(thumb, (left, top))

    canvas.save(
        selected_figures[split],
        format="PNG",
    )

    print(
        f"  {split}: {region_id}",
        flush=True,
    )

# Free substantial intermediate image arrays before report.
del metric_lookup
del preview_cache


print("\n[5] Saving quality tables and report...", flush=True)

IMAGE_METRICS.parent.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

write_csv(IMAGE_METRICS, quality_rows, IMAGE_FIELDS)
write_csv(PAIR_METRICS, pair_rows, PAIR_FIELDS)

scanner_summary = {}

for scanner in SCANNERS:
    subset = [
        row for row in quality_rows
        if row["scanner"] == scanner
    ]

    scanner_summary[scanner] = {
        "images": len(subset),
        "mean_gray": describe(
            [r["mean_gray"] for r in subset]
        ),
        "std_gray": describe(
            [r["std_gray"] for r in subset]
        ),
        "near_white_fraction": describe(
            [r["near_white_fraction"] for r in subset]
        ),
        "edge_variance": describe(
            [r["edge_variance"] for r in subset]
        ),
    }

valid_correlations = [
    row["gray_correlation"]
    for row in pair_rows
    if row["gray_correlation"] is not None
]

report = {
    "phase": "4",
    "step": "4.1.11",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "source_phase4_1_10_audit_sha256": previous[
        "final_audit_sha256"
    ],
    "verified_images": len(quality_rows),
    "matched_scanner_pairs": len(pair_rows),
    "regions": len(by_region),
    "split_counts": dict(Counter(
        row["spatial_split"] for row in quality_rows
    )),
    "scanner_quality_summary": scanner_summary,
    "gray_correlation": (
        describe(valid_correlations)
        if valid_correlations else None
    ),
    "undefined_gray_correlations": (
        len(pair_rows) - len(valid_correlations)
    ),
    "gray_absolute_difference": describe(
        [
            row["gray_mean_absolute_difference"]
            for row in pair_rows
        ]
    ),
    "contact_sheet_regions": selected_regions,
    "contact_sheets": {
        split: str(path.relative_to(BASE))
        for split, path in selected_figures.items()
    },
    "image_metrics_sha256": hashlib.sha256(
        IMAGE_METRICS.read_bytes()
    ).hexdigest(),
    "pair_metrics_sha256": hashlib.sha256(
        PAIR_METRICS.read_bytes()
    ).hexdigest(),
    "manual_registration_review_completed": False,
    "automatic_quality_exclusions": 0,
    "biological_independence_verified": False,
    "phase3b_modified": False,
    "limitations": [
        "Image-level metrics are not diagnostic labels",
        "High similarity does not prove exact registration",
        "Scanner color and intensity differences affect correlation",
        "Contact sheets cover one representative region per split",
        "Human image-quality and registration review remains pending",
    ],
}

with REPORT.open("x", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)

print("\n" + "=" * 70, flush=True)
print("PHASE 4.1.11 - IMAGE QUALITY AUDIT COMPLETED", flush=True)
print("=" * 70)

print("Images verified:", len(quality_rows))
print("Scanner pairs assessed:", len(pair_rows))
print("Regions:", len(by_region))

if valid_correlations:
    print(
        "Median paired grayscale correlation:",
        f"{np.median(valid_correlations):.4f}",
    )

print("\nSaved:")
print(IMAGE_METRICS)
print(PAIR_METRICS)
print(REPORT)

for path in selected_figures.values():
    print(path)

print("\nOriginal PNG images: UNCHANGED")
print("Manual registration review: PENDING")
print("Biological independence: UNVERIFIED")
print("\nSTEP 4.1.11 COMPLETED")
