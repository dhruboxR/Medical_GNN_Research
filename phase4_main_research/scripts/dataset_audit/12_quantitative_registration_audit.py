
"""
PHASE 4.1.12 - QUANTITATIVE REGISTRATION AUDIT

Evaluate all 2,160 AT2/GT450 matched image pairs.

Metrics:
  - Raw grayscale SSIM
  - Contrast-normalized SSIM
  - Edge-map correlation
  - Estimated small translation from phase correlation
  - Pre/post-translation edge correlation on overlapping pixels
  - Translation distance and improvement

Outputs:
  - Per-pair metrics CSV
  - Per-region summary CSV
  - Worst-pair contact sheets for each spatial split
  - JSON audit report

Scientific limitations:
  - SSIM and edge correlation are registration proxies.
  - Phase correlation proposes a shift, not ground truth.
  - This audit does not prove pixel-perfect correspondence.
  - No pair is excluded automatically.
  - No training or image modification occurs.
"""

from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
from io import BytesIO
import csv
import hashlib
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, sobel
from skimage.metrics import structural_similarity
from skimage.registration import phase_cross_correlation


ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "phase4_main_research"

SOURCE_CSV = (
    BASE / "outputs/tables"
    / "10_stage_a_original_png_audit.csv"
)

SOURCE_REPORT = (
    BASE / "outputs/reports"
    / "10_stage_a_download_report.json"
)

PREVIOUS_REPORT = (
    BASE / "outputs/reports"
    / "11_image_quality_report.json"
)

PAIR_OUTPUT = (
    BASE / "outputs/tables"
    / "12_registration_pair_metrics.csv"
)

REGION_OUTPUT = (
    BASE / "outputs/tables"
    / "12_registration_region_summary.csv"
)

REPORT_OUTPUT = (
    BASE / "outputs/reports"
    / "12_registration_audit_report.json"
)

FIGURE_DIR = BASE / "outputs/figures"

SPLITS = ("train", "validation", "test")
SCANNERS = ("AT2", "GT450")

EXPECTED_PAIRS = 2160
EXPECTED_REGIONS = 60

MAX_SHIFT = 8
MIN_IMPROVEMENT = 0.02
WORST_PAIRS_PER_SPLIT = 4

PAIR_COLUMNS = [
    "region_id",
    "spatial_split",
    "tile_id",
    "raw_ssim",
    "normalized_ssim",
    "edge_correlation",
    "estimated_shift_y",
    "estimated_shift_x",
    "estimated_shift_magnitude",
    "shift_within_limit",
    "edge_correlation_same_crop",
    "edge_correlation_shifted",
    "edge_improvement",
    "translation_improvement_flag",
]

REGION_COLUMNS = [
    "region_id",
    "spatial_split",
    "pair_count",
    "median_raw_ssim",
    "median_normalized_ssim",
    "median_edge_correlation",
    "median_estimated_shift_magnitude",
    "within_limit_shift_count",
    "improved_translation_count",
]


def digest_file(path):
    h = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


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


def clean_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def median_or_none(values):
    usable = [
        float(value)
        for value in values
        if value is not None and np.isfinite(value)
    ]

    if not usable:
        return None

    return float(np.median(usable))


def distribution(values):
    usable = np.array(
        [
            float(value)
            for value in values
            if value is not None and np.isfinite(value)
        ],
        dtype=np.float64,
    )

    if len(usable) == 0:
        return None

    return {
        "count": len(usable),
        "min": float(np.min(usable)),
        "p05": float(np.percentile(usable, 5)),
        "median": float(np.median(usable)),
        "mean": float(np.mean(usable)),
        "p95": float(np.percentile(usable, 95)),
        "max": float(np.max(usable)),
    }


def safe_image_path(relative_name):
    relative = Path(relative_name)

    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(
            f"Unsafe relative image path: {relative_name}"
        )

    result = (BASE / relative).resolve()

    root = (
        BASE / "datasets/expanded_plism/stage_a"
    ).resolve()

    if not result.is_relative_to(root):
        raise RuntimeError(
            "Image is outside the Stage A directory."
        )

    return result


def load_grayscale_verified(record):
    path = safe_image_path(
        record["destination_relative_path"]
    )

    if not path.is_file():
        raise RuntimeError(f"Missing PNG: {path}")

    raw = path.read_bytes()

    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise RuntimeError(
            f"Image hash mismatch: {path}"
        )

    with Image.open(BytesIO(raw)) as image:
        if image.format != "PNG":
            raise RuntimeError("Non-PNG image found.")

        if image.size != (224, 224):
            raise RuntimeError(
                f"Unexpected image size: {image.size}"
            )

        image = image.convert("L")
        gray = np.asarray(
            image,
            dtype=np.float32,
        ) / 255.0

    return gray


def normalize_contrast(gray):
    """
    Percentile contrast normalization for
    diagnostic comparison only.

    Does not modify the saved original image.
    """

    low, high = np.percentile(gray, [2, 98])

    if high - low < 1e-6:
        return np.zeros_like(gray)

    return np.clip(
        (gray - low) / (high - low),
        0.0,
        1.0,
    ).astype(np.float32)


def edge_map(gray):
    smoothed = gaussian_filter(
        gray.astype(np.float32),
        sigma=1.0,
    )

    gx = sobel(smoothed, axis=1)
    gy = sobel(smoothed, axis=0)

    return np.hypot(
        gx, gy
    ).astype(np.float32)


def correlation(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    a = a - np.mean(a)
    b = b - np.mean(b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator < 1e-12:
        return None

    return clean_number(
        np.dot(a, b) / denominator
    )


def shifted_overlap(reference, moving, dy, dx):
    """
    Return corresponding overlapping regions after
    moving the second image by (dy, dx) pixels.

    No warping, interpolation, or image saving.
    """

    h, w = reference.shape

    if abs(dy) >= h or abs(dx) >= w:
        raise RuntimeError("Translation exceeds image dimensions.")

    if dy >= 0:
        ref_y = slice(dy, h)
        mov_y = slice(0, h - dy)
    else:
        ref_y = slice(0, h + dy)
        mov_y = slice(-dy, h)

    if dx >= 0:
        ref_x = slice(dx, w)
        mov_x = slice(0, w - dx)
    else:
        ref_x = slice(0, w + dx)
        mov_x = slice(-dx, w)

    return (
        reference[ref_y, ref_x],
        moving[mov_y, mov_x],
        moving[ref_y, ref_x],
    )


def estimate_translation(reference_edge, moving_edge):
    """
    Proposal only. Large or uncertain shifts
    are not used for adjusted comparison.
    """

    if (
        np.std(reference_edge) < 1e-8
        or np.std(moving_edge) < 1e-8
    ):
        return None

    shift, error, phase = phase_cross_correlation(
        reference_edge,
        moving_edge,
        upsample_factor=1,
        normalization=None,
    )

    if not np.all(np.isfinite(shift)):
        return None

    return (
        int(np.rint(shift[0])),
        int(np.rint(shift[1])),
    )


def inspect_pair(at2_record, gt450_record):
    at2 = load_grayscale_verified(at2_record)
    gt450 = load_grayscale_verified(gt450_record)

    raw_ssim = structural_similarity(
        at2,
        gt450,
        data_range=1.0,
    )

    norm_at2 = normalize_contrast(at2)
    norm_gt450 = normalize_contrast(gt450)

    normalized_ssim = structural_similarity(
        norm_at2,
        norm_gt450,
        data_range=1.0,
    )

    edge_at2 = edge_map(norm_at2)
    edge_gt450 = edge_map(norm_gt450)

    original_edge_corr = correlation(
        edge_at2,
        edge_gt450,
    )

    estimated = estimate_translation(
        edge_at2,
        edge_gt450,
    )

    dy = None
    dx = None
    magnitude = None
    within_limit = False

    same_crop_corr = None
    shifted_corr = None
    improvement = None
    improvement_flag = False

    if estimated is not None:
        dy, dx = estimated

        magnitude = math.hypot(dy, dx)

        within_limit = (
            abs(dy) <= MAX_SHIFT
            and abs(dx) <= MAX_SHIFT
        )

        if within_limit:
            ref, shifted_moving, original_crop = (
                shifted_overlap(
                    edge_at2,
                    edge_gt450,
                    dy,
                    dx,
                )
            )

            # Both comparisons use the same reference
            # area, avoiding unfair comparisons against
            # a full-image baseline.
            same_crop_corr = correlation(
                ref,
                original_crop,
            )

            shifted_corr = correlation(
                ref,
                shifted_moving,
            )

            if (
                same_crop_corr is not None
                and shifted_corr is not None
            ):
                improvement = (
                    shifted_corr - same_crop_corr
                )

                improvement_flag = (
                    improvement >= MIN_IMPROVEMENT
                )

    return {
        "region_id": at2_record["region_id"],
        "spatial_split": at2_record["spatial_split"],
        "tile_id": at2_record["tile_id"],
        "raw_ssim": clean_number(raw_ssim),
        "normalized_ssim": clean_number(normalized_ssim),
        "edge_correlation": original_edge_corr,
        "estimated_shift_y": dy,
        "estimated_shift_x": dx,
        "estimated_shift_magnitude": magnitude,
        "shift_within_limit": within_limit,
        "edge_correlation_same_crop": same_crop_corr,
        "edge_correlation_shifted": shifted_corr,
        "edge_improvement": improvement,
        "translation_improvement_flag": improvement_flag,
    }


def create_review_sheet(split, selected, pair_lookup, destination):
    """
    Four lowest-SSIM image pairs in each split.
    Each row shows the original AT2 and GT450 PNGs.
    """

    thumb = 224
    margin = 18
    gap = 18
    row_height = thumb + 55

    width = margin * 2 + thumb * 2 + gap
    height = 45 + len(selected) * row_height

    canvas = Image.new(
        "RGB",
        (width, height),
        "white",
    )

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text(
        (margin, 12),
        f"{split.upper()} - lowest raw SSIM pairs",
        fill="black",
        font=font,
    )

    for index, result in enumerate(selected):
        key = (
            result["region_id"],
            result["tile_id"],
        )

        pair = pair_lookup[key]
        top = 45 + index * row_height

        raw_score = result["raw_ssim"]

        label = (
            f"{result['region_id']} | "
            f"{result['tile_id']} | "
            f"SSIM={raw_score:.3f}"
        )

        draw.text(
            (margin, top),
            label,
            fill="black",
            font=font,
        )

        for scanner_index, scanner in enumerate(SCANNERS):
            path = safe_image_path(
                pair[scanner]["destination_relative_path"]
            )

            left = (
                margin
                + scanner_index * (thumb + gap)
            )

            with Image.open(path) as source:
                rgb = source.convert("RGB")
                canvas.paste(
                    rgb,
                    (left, top + 18),
                )

            draw.text(
                (left, top + 18 + thumb + 2),
                scanner,
                fill="black",
                font=font,
            )

    with destination.open("xb") as stream:
        canvas.save(stream, format="PNG")


print("=" * 70, flush=True)
print("PHASE 4.1.12 - CROSS-SCANNER REGISTRATION AUDIT", flush=True)
print("=" * 70, flush=True)

print("\n[1] Validating research inputs...", flush=True)

for path in (SOURCE_CSV, SOURCE_REPORT, PREVIOUS_REPORT):
    if not path.is_file():
        raise RuntimeError(f"Missing input: {path}")

figures = {
    split: FIGURE_DIR / f"12_{split}_worst_scanner_pairs.png"
    for split in SPLITS
}

outputs = [
    PAIR_OUTPUT,
    REGION_OUTPUT,
    REPORT_OUTPUT,
    *figures.values(),
]

for output in outputs:
    if output.exists():
        raise RuntimeError(
            f"Refusing to overwrite: {output}"
        )

with SOURCE_REPORT.open(encoding="utf-8") as stream:
    source_report = json.load(stream)

with PREVIOUS_REPORT.open(encoding="utf-8") as stream:
    previous = json.load(stream)

if digest_file(SOURCE_CSV) != source_report[
    "final_audit_sha256"
]:
    raise RuntimeError(
        "Stage A provenance hash mismatch."
    )

if previous["verified_images"] != 4320:
    raise RuntimeError(
        "Previous image-quality audit mismatch."
    )

records = read_csv(SOURCE_CSV)

if len(records) != 4320:
    raise RuntimeError(
        "Expected 4320 original image records."
    )

pair_lookup = defaultdict(dict)
region_splits = {}

for record in records:
    if record["scanner"] not in SCANNERS:
        raise RuntimeError("Unexpected scanner.")

    if record["spatial_split"] not in SPLITS:
        raise RuntimeError("Unexpected partition.")

    region = record["region_id"]
    split = record["spatial_split"]

    if region in region_splits and region_splits[region] != split:
        raise RuntimeError("Region appears across splits.")

    region_splits[region] = split

    key = (region, record["tile_id"])

    if record["scanner"] in pair_lookup[key]:
        raise RuntimeError("Duplicate paired image.")

    pair_lookup[key][record["scanner"]] = record

if len(pair_lookup) != EXPECTED_PAIRS:
    raise RuntimeError(
        "Incorrect scanner-pair count."
    )

if len(region_splits) != EXPECTED_REGIONS:
    raise RuntimeError("Incorrect region count.")

for key, pair in pair_lookup.items():
    if set(pair) != set(SCANNERS):
        raise RuntimeError(
            f"Missing scanner in pair: {key}"
        )

    if pair["AT2"]["spatial_split"] != pair["GT450"]["spatial_split"]:
        raise RuntimeError("Pair split mismatch.")

print("Regions:", len(region_splits))
print("Matched image pairs:", len(pair_lookup))
print("Original PNGs:", len(records))


print("\n[2] Computing quantitative registration metrics...", flush=True)

pair_metrics = []

for index, key in enumerate(sorted(pair_lookup), 1):
    pair = pair_lookup[key]

    result = inspect_pair(
        pair["AT2"],
        pair["GT450"],
    )

    pair_metrics.append(result)

    if index % 250 == 0 or index == EXPECTED_PAIRS:
        print(
            f"  Assessed {index}/{EXPECTED_PAIRS} pairs",
            flush=True,
        )


print("\n[3] Summarizing results by region...", flush=True)

region_groups = defaultdict(list)

for result in pair_metrics:
    region_groups[
        (result["region_id"], result["spatial_split"])
    ].append(result)

region_metrics = []

for (region_id, split), group in sorted(region_groups.items()):
    if len(group) != 36:
        raise RuntimeError(
            f"Expected 36 pairs in {region_id}."
        )

    region_metrics.append({
        "region_id": region_id,
        "spatial_split": split,
        "pair_count": len(group),
        "median_raw_ssim": median_or_none(
            [r["raw_ssim"] for r in group]
        ),
        "median_normalized_ssim": median_or_none(
            [r["normalized_ssim"] for r in group]
        ),
        "median_edge_correlation": median_or_none(
            [r["edge_correlation"] for r in group]
        ),
        "median_estimated_shift_magnitude": median_or_none(
            [
                r["estimated_shift_magnitude"]
                for r in group
            ]
        ),
        "within_limit_shift_count": sum(
            r["shift_within_limit"]
            for r in group
        ),
        "improved_translation_count": sum(
            r["translation_improvement_flag"]
            for r in group
        ),
    })

print("Region summaries:", len(region_metrics))


print("\n[4] Creating worst-pair visual review sheets...", flush=True)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

review_selections = {}

for split in SPLITS:
    candidates = sorted(
        [
            r for r in pair_metrics
            if r["spatial_split"] == split
        ],
        key=lambda r: (
            r["raw_ssim"],
            r["region_id"],
            r["tile_id"],
        ),
    )

    if len(candidates) < WORST_PAIRS_PER_SPLIT:
        raise RuntimeError(
            f"Not enough pairs in {split}."
        )

    selected = candidates[:WORST_PAIRS_PER_SPLIT]

    create_review_sheet(
        split,
        selected,
        pair_lookup,
        figures[split],
    )

    review_selections[split] = [
        {
            "region_id": r["region_id"],
            "tile_id": r["tile_id"],
            "raw_ssim": r["raw_ssim"],
        }
        for r in selected
    ]

    print(
        f"  {split}: {len(selected)} lowest-SSIM pairs",
        flush=True,
    )


print("\n[5] Saving registration audit...", flush=True)

PAIR_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

save_csv_new(
    PAIR_OUTPUT,
    pair_metrics,
    PAIR_COLUMNS,
)

save_csv_new(
    REGION_OUTPUT,
    region_metrics,
    REGION_COLUMNS,
)

overall = {
    "raw_ssim": distribution(
        [r["raw_ssim"] for r in pair_metrics]
    ),
    "normalized_ssim": distribution(
        [r["normalized_ssim"] for r in pair_metrics]
    ),
    "edge_correlation": distribution(
        [r["edge_correlation"] for r in pair_metrics]
    ),
    "estimated_shift_magnitude": distribution(
        [
            r["estimated_shift_magnitude"]
            for r in pair_metrics
        ]
    ),
    "edge_improvement_bounded": distribution(
        [
            r["edge_improvement"]
            for r in pair_metrics
            if r["shift_within_limit"]
        ]
    ),
}

partition_summary = {}

for split in SPLITS:
    subset = [
        r for r in pair_metrics
        if r["spatial_split"] == split
    ]

    partition_summary[split] = {
        "pairs": len(subset),
        "median_raw_ssim": median_or_none(
            [r["raw_ssim"] for r in subset]
        ),
        "median_normalized_ssim": median_or_none(
            [r["normalized_ssim"] for r in subset]
        ),
        "median_edge_correlation": median_or_none(
            [r["edge_correlation"] for r in subset]
        ),
        "bounded_translation_proposals": sum(
            r["shift_within_limit"]
            for r in subset
        ),
        "improved_translation_proposals": sum(
            r["translation_improvement_flag"]
            for r in subset
        ),
    }

report = {
    "phase": "4",
    "step": "4.1.12",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "dataset": "owkin/plism-dataset-tiles",
    "source_audit_sha256": digest_file(SOURCE_CSV),
    "pairs_assessed": len(pair_metrics),
    "regions_assessed": len(region_metrics),
    "maximum_shift_per_axis_pixels": MAX_SHIFT,
    "minimum_edge_improvement": MIN_IMPROVEMENT,
    "overall_metrics": overall,
    "partition_metrics": partition_summary,
    "pairs_with_bounded_shift": sum(
        r["shift_within_limit"]
        for r in pair_metrics
    ),
    "pairs_with_bounded_improvement": sum(
        r["translation_improvement_flag"]
        for r in pair_metrics
    ),
    "worst_pair_reviews": review_selections,
    "review_figures": {
        split: str(path.relative_to(BASE))
        for split, path in figures.items()
    },
    "pair_metrics_sha256": digest_file(PAIR_OUTPUT),
    "region_metrics_sha256": digest_file(REGION_OUTPUT),
    "original_images_modified": False,
    "pairs_excluded": 0,
    "manual_registration_review_completed": False,
    "dataset_approved_for_training": False,
    "biological_independence_verified": False,
    "limitations": [
        "SSIM is affected by scanner appearance",
        "Contrast normalization is diagnostic only",
        "Edge correlation is not proof of registration",
        "Estimated translations may be unreliable",
        "Only shifts within eight pixels per axis are evaluated",
        "Worst-pair sheets are based on raw SSIM",
        "Independent biological validation is unavailable",
    ],
}

with REPORT_OUTPUT.open(
    "x",
    encoding="utf-8",
) as stream:
    json.dump(
        report,
        stream,
        indent=2,
    )

print("\n" + "=" * 70)
print("PHASE 4.1.12 - REGISTRATION AUDIT COMPLETED")
print("=" * 70)

print("Pairs assessed:", len(pair_metrics))
print("Regions assessed:", len(region_metrics))

for name in (
    "raw_ssim",
    "normalized_ssim",
    "edge_correlation",
    "estimated_shift_magnitude",
):
    value = overall[name]

    print(
        f"Median {name}: "
        f"{value['median']:.4f}"
        if value else
        f"Median {name}: undefined"
    )

print(
    "Pairs with bounded translation proposals:",
    report["pairs_with_bounded_shift"],
)

print(
    "Pairs with improved edge agreement:",
    report["pairs_with_bounded_improvement"],
)

print("\nSaved:")
print(PAIR_OUTPUT)
print(REGION_OUTPUT)
print(REPORT_OUTPUT)

for path in figures.values():
    print(path)

print("\nOriginal images: UNCHANGED")
print("Image pairs excluded: 0")
print("Biological independence: UNVERIFIED")
print("Training approval: PENDING REVIEW")
print("\nSTEP 4.1.12 COMPLETED")
