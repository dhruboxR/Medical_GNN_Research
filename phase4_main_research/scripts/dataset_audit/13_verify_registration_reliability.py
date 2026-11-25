
"""
PHASE 4.1.13 - REGISTRATION RELIABILITY VERIFICATION

Research objectives
-------------------
1. Test whether a phase-correlation registration
   estimator recovers known synthetic image shifts.
2. Estimate AT2/GT450 displacement using the left
   half of each image.
3. Evaluate the proposed displacement using the
   independent right half of each image.
4. Summarize results for all 60 spatial regions.

Important:
- All original images remain unchanged.
- No translated images are saved.
- No image pairs are excluded.
- No training protocol is modified.
- Synthetic recovery does NOT prove real registration.
- The left/right partitions are evaluation regions
  within the same image, not biological independence.
- Validation/test metrics are descriptive only.
"""

from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
from io import BytesIO
import csv
import hashlib
import json
import math
import random

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel, shift as ndi_shift
from skimage.registration import phase_cross_correlation


ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "phase4_main_research"

SOURCE = (
    BASE / "outputs/tables"
    / "10_stage_a_original_png_audit.csv"
)

SOURCE_REPORT = (
    BASE / "outputs/reports"
    / "10_stage_a_download_report.json"
)

PREVIOUS_REPORT = (
    BASE / "outputs/reports"
    / "12_registration_audit_report.json"
)

REAL_OUTPUT = (
    BASE / "outputs/tables"
    / "13_independent_registration_metrics.csv"
)

SYNTHETIC_OUTPUT = (
    BASE / "outputs/tables"
    / "13_synthetic_shift_recovery.csv"
)

REGION_OUTPUT = (
    BASE / "outputs/tables"
    / "13_registration_region_summary.csv"
)

REPORT_OUTPUT = (
    BASE / "outputs/reports"
    / "13_registration_reliability_report.json"
)

SPLITS = ("train", "validation", "test")
SCANNERS = ("AT2", "GT450")

SEED = 42

EXPECTED_PAIRS = 2160
EXPECTED_IMAGES = 4320
EXPECTED_REGIONS = 60

IMAGE_SIZE = 224
HALF_WIDTH = IMAGE_SIZE // 2

MAX_REAL_SHIFT = 8
SYNTHETIC_MAX_SHIFT = 6

MIN_HELDOUT_IMPROVEMENT = 0.02

# Synthetic samples are balanced across scanners.
SYNTHETIC_SAMPLE_COUNTS = {
    "train": 40,
    "validation": 10,
    "test": 10,
}

REAL_FIELDS = [
    "region_id",
    "spatial_split",
    "tile_id",
    "estimated_shift_y",
    "estimated_shift_x",
    "estimated_shift_magnitude",
    "shift_within_limit",
    "heldout_original_correlation",
    "heldout_shifted_correlation",
    "heldout_improvement",
    "heldout_improvement_flag",
]

SYNTHETIC_FIELDS = [
    "region_id",
    "spatial_split",
    "scanner",
    "tile_id",
    "applied_shift_y",
    "applied_shift_x",
    "expected_recovery_y",
    "expected_recovery_x",
    "estimated_recovery_y",
    "estimated_recovery_x",
    "absolute_error_y",
    "absolute_error_x",
    "euclidean_error",
    "recovered_within_one_pixel",
]

REGION_FIELDS = [
    "region_id",
    "spatial_split",
    "pair_count",
    "bounded_shift_count",
    "median_shift_magnitude",
    "median_heldout_original_correlation",
    "median_heldout_shifted_correlation",
    "median_heldout_improvement",
    "heldout_improvement_count",
]


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_csv(path):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as stream:
        return list(csv.DictReader(stream))


def save_csv(path, records, fields):
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
        writer.writerows(records)


def save_json(path, value):
    with path.open(
        "x",
        encoding="utf-8",
    ) as stream:
        json.dump(value, stream, indent=2)


def safe_path(relative_name):
    relative = Path(relative_name)

    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("Unsafe image path.")

    path = (BASE / relative).resolve()

    allowed_root = (
        BASE / "datasets/expanded_plism/stage_a"
    ).resolve()

    if not path.is_relative_to(allowed_root):
        raise RuntimeError(
            "Source image outside Stage A."
        )

    return path


def load_gray(record):
    path = safe_path(
        record["destination_relative_path"]
    )

    raw = path.read_bytes()

    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise RuntimeError(
            f"Original image SHA256 mismatch: {path}"
        )

    with Image.open(BytesIO(raw)) as image:
        if image.format != "PNG":
            raise RuntimeError("Unexpected file format.")

        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError("Unexpected image dimensions.")

        gray = np.asarray(
            image.convert("L"),
            dtype=np.float32,
        ) / 255.0

    return gray


def normalize(gray):
    low, high = np.percentile(
        gray,
        [2, 98],
    )

    if high - low < 1e-6:
        return np.zeros_like(
            gray,
            dtype=np.float32,
        )

    return np.clip(
        (gray - low) / (high - low),
        0.0,
        1.0,
    ).astype(np.float32)


def gradient_magnitude(gray):
    image = gaussian_filter(
        normalize(gray),
        sigma=1.0,
    )

    gx = sobel(
        image,
        axis=1,
    )

    gy = sobel(
        image,
        axis=0,
    )

    magnitude = np.hypot(gx, gy)

    return magnitude.astype(np.float32)


def correlation(first, second):
    a = np.asarray(
        first,
        dtype=np.float64,
    ).ravel()

    b = np.asarray(
        second,
        dtype=np.float64,
    ).ravel()

    a = a - a.mean()
    b = b - b.mean()

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator < 1e-12:
        return None

    value = float(
        np.dot(a, b) / denominator
    )

    return value if np.isfinite(value) else None


def median(values):
    usable = [
        float(v)
        for v in values
        if v is not None and np.isfinite(v)
    ]

    if not usable:
        return None

    return float(np.median(usable))


def distribution(values):
    usable = np.array(
        [
            float(v)
            for v in values
            if v is not None and np.isfinite(v)
        ],
        dtype=np.float64,
    )

    if not len(usable):
        return None

    return {
        "count": int(len(usable)),
        "median": float(np.median(usable)),
        "mean": float(np.mean(usable)),
        "p05": float(np.percentile(usable, 5)),
        "p95": float(np.percentile(usable, 95)),
        "min": float(usable.min()),
        "max": float(usable.max()),
    }


def estimate_shift(reference, moving):
    """
    Phase correlation on gradient magnitudes.

    A Hann window limits boundary discontinuities.

    The returned shift is the displacement required
    to move 'moving' into the reference frame.
    """

    if (
        reference.shape != moving.shape
        or reference.ndim != 2
    ):
        raise RuntimeError(
            "Invalid estimator input dimensions."
        )

    if (
        np.std(reference) < 1e-8
        or np.std(moving) < 1e-8
    ):
        return None

    height, width = reference.shape

    window = np.outer(
        np.hanning(height),
        np.hanning(width),
    ).astype(np.float32)

    a = (
        reference - reference.mean()
    ) * window

    b = (
        moving - moving.mean()
    ) * window

    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return None

    shift, _, _ = phase_cross_correlation(
        a,
        b,
        upsample_factor=1,
        normalization=None,
    )

    if not np.all(np.isfinite(shift)):
        return None

    return (
        int(np.rint(shift[0])),
        int(np.rint(shift[1])),
    )


def overlap_crops(reference, moving, dy, dx):
    """
    Compare original and translated moving images
    against the exact same reference area.

    No image interpolation is performed.

    Returns:
        reference crop,
        original moving crop,
        shifted moving crop.
    """

    height, width = reference.shape

    if (
        reference.shape != moving.shape
        or abs(dy) >= height
        or abs(dx) >= width
    ):
        raise RuntimeError("Invalid overlap parameters.")

    if dy >= 0:
        ref_y = slice(dy, height)
        shifted_y = slice(0, height - dy)
    else:
        ref_y = slice(0, height + dy)
        shifted_y = slice(-dy, height)

    if dx >= 0:
        ref_x = slice(dx, width)
        shifted_x = slice(0, width - dx)
    else:
        ref_x = slice(0, width + dx)
        shifted_x = slice(-dx, width)

    return (
        reference[ref_y, ref_x],
        moving[ref_y, ref_x],
        moving[shifted_y, shifted_x],
    )


def inspect_real_pair(at2, gt450):
    """
    Estimate from the left half.
    Independently evaluate on the right half.
    """

    at2_gray = load_gray(at2)
    gt_gray = load_gray(gt450)

    at2_edges = gradient_magnitude(at2_gray)
    gt_edges = gradient_magnitude(gt_gray)

    estimated = estimate_shift(
        at2_edges[:, :HALF_WIDTH],
        gt_edges[:, :HALF_WIDTH],
    )

    result = {
        "region_id": at2["region_id"],
        "spatial_split": at2["spatial_split"],
        "tile_id": at2["tile_id"],
        "estimated_shift_y": None,
        "estimated_shift_x": None,
        "estimated_shift_magnitude": None,
        "shift_within_limit": False,
        "heldout_original_correlation": None,
        "heldout_shifted_correlation": None,
        "heldout_improvement": None,
        "heldout_improvement_flag": False,
    }

    if estimated is None:
        return result

    dy, dx = estimated

    magnitude = math.hypot(
        dy,
        dx,
    )

    bounded = (
        abs(dy) <= MAX_REAL_SHIFT
        and abs(dx) <= MAX_REAL_SHIFT
    )

    result.update({
        "estimated_shift_y": dy,
        "estimated_shift_x": dx,
        "estimated_shift_magnitude": magnitude,
        "shift_within_limit": bounded,
    })

    if not bounded:
        return result

    reference_right = at2_edges[:, HALF_WIDTH:]
    moving_right = gt_edges[:, HALF_WIDTH:]

    ref, original, shifted = overlap_crops(
        reference_right,
        moving_right,
        dy,
        dx,
    )

    before = correlation(
        ref,
        original,
    )

    after = correlation(
        ref,
        shifted,
    )

    improvement = None

    if before is not None and after is not None:
        improvement = after - before

    result.update({
        "heldout_original_correlation": before,
        "heldout_shifted_correlation": after,
        "heldout_improvement": improvement,
        "heldout_improvement_flag": (
            improvement is not None
            and improvement >= MIN_HELDOUT_IMPROVEMENT
        ),
    })

    return result


def inspect_synthetic(record, dy, dx):
    """
    Apply a known translation to an original image.

    Estimate its reversal using only the left half.
    """

    gray = load_gray(record)

    translated = ndi_shift(
        gray,
        shift=(dy, dx),
        order=1,
        mode="nearest",
        prefilter=False,
    ).astype(np.float32)

    reference_edges = gradient_magnitude(gray)
    translated_edges = gradient_magnitude(translated)

    estimated = estimate_shift(
        reference_edges[:, :HALF_WIDTH],
        translated_edges[:, :HALF_WIDTH],
    )

    estimate_y = None
    estimate_x = None

    error_y = None
    error_x = None
    error = None

    recovered = False

    if estimated is not None:
        estimate_y, estimate_x = estimated

        error_y = abs(
            estimate_y + dy
        )

        error_x = abs(
            estimate_x + dx
        )

        error = math.hypot(
            error_y,
            error_x,
        )

        recovered = (
            error_y <= 1
            and error_x <= 1
        )

    return {
        "region_id": record["region_id"],
        "spatial_split": record["spatial_split"],
        "scanner": record["scanner"],
        "tile_id": record["tile_id"],
        "applied_shift_y": dy,
        "applied_shift_x": dx,
        "expected_recovery_y": -dy,
        "expected_recovery_x": -dx,
        "estimated_recovery_y": estimate_y,
        "estimated_recovery_x": estimate_x,
        "absolute_error_y": error_y,
        "absolute_error_x": error_x,
        "euclidean_error": error,
        "recovered_within_one_pixel": recovered,
    }


print("=" * 70, flush=True)
print("PHASE 4.1.13 - REGISTRATION RELIABILITY", flush=True)
print("=" * 70, flush=True)

print("\n[1] Validating source files...", flush=True)

for path in (SOURCE, SOURCE_REPORT, PREVIOUS_REPORT):
    if not path.is_file():
        raise RuntimeError(
            f"Missing input: {path}"
        )

outputs = (
    REAL_OUTPUT,
    SYNTHETIC_OUTPUT,
    REGION_OUTPUT,
    REPORT_OUTPUT,
)

for path in outputs:
    if path.exists():
        raise RuntimeError(
            f"Refusing to overwrite: {path}"
        )

with SOURCE_REPORT.open(encoding="utf-8") as stream:
    source_report = json.load(stream)

with PREVIOUS_REPORT.open(encoding="utf-8") as stream:
    previous_report = json.load(stream)

if sha256_file(SOURCE) != source_report["final_audit_sha256"]:
    raise RuntimeError(
        "Source provenance SHA256 mismatch."
    )

if previous_report["pairs_assessed"] != EXPECTED_PAIRS:
    raise RuntimeError(
        "Previous registration audit mismatch."
    )

records = read_csv(SOURCE)

if len(records) != EXPECTED_IMAGES:
    raise RuntimeError(
        "Unexpected number of source images."
    )

pairs = defaultdict(dict)
regions = defaultdict(set)
records_by_split_scanner = defaultdict(list)

for record in records:
    split = record["spatial_split"]
    scanner = record["scanner"]

    if split not in SPLITS or scanner not in SCANNERS:
        raise RuntimeError(
            "Unexpected split or scanner."
        )

    region_id = record["region_id"]

    regions[region_id].add(split)

    key = (
        region_id,
        record["tile_id"],
    )

    if scanner in pairs[key]:
        raise RuntimeError(
            f"Duplicate pair record: {key}"
        )

    pairs[key][scanner] = record

    records_by_split_scanner[
        (split, scanner)
    ].append(record)

if len(pairs) != EXPECTED_PAIRS:
    raise RuntimeError(
        "Expected 2160 matched pairs."
    )

if len(regions) != EXPECTED_REGIONS:
    raise RuntimeError(
        "Expected 60 regions."
    )

if any(len(splits) != 1 for splits in regions.values()):
    raise RuntimeError(
        "Region crosses spatial partitions."
    )

if any(set(pair) != set(SCANNERS) for pair in pairs.values()):
    raise RuntimeError(
        "Incomplete scanner pair."
    )

print("Original images:", len(records))
print("Paired positions:", len(pairs))
print("Spatial regions:", len(regions))


print("\n[2] Synthetic-shift recovery experiment...", flush=True)

rng = random.Random(SEED)

synthetic_sample = []

for split in SPLITS:
    for scanner in SCANNERS:
        available = sorted(
            records_by_split_scanner[(split, scanner)],
            key=lambda row: (
                row["region_id"],
                row["tile_id"],
            ),
        )

        count = SYNTHETIC_SAMPLE_COUNTS[split]

        if len(available) < count:
            raise RuntimeError(
                "Insufficient synthetic test samples."
            )

        synthetic_sample.extend(
            rng.sample(
                available,
                count,
            )
        )

synthetic_results = []

for index, record in enumerate(
    synthetic_sample,
    1,
):
    while True:
        dy = rng.randint(
            -SYNTHETIC_MAX_SHIFT,
            SYNTHETIC_MAX_SHIFT,
        )

        dx = rng.randint(
            -SYNTHETIC_MAX_SHIFT,
            SYNTHETIC_MAX_SHIFT,
        )

        if dy != 0 or dx != 0:
            break

    result = inspect_synthetic(
        record,
        dy,
        dx,
    )

    synthetic_results.append(result)

    if index % 30 == 0:
        print(
            f"  Synthetic cases: "
            f"{index}/{len(synthetic_sample)}",
            flush=True,
        )

synthetic_success = sum(
    r["recovered_within_one_pixel"]
    for r in synthetic_results
)

synthetic_rate = (
    synthetic_success
    / len(synthetic_results)
)

print("Synthetic tests:", len(synthetic_results))
print("Recovered within 1 px per axis:", synthetic_success)
print(
    "Recovery rate:",
    f"{100 * synthetic_rate:.2f}%",
)


print("\n[3] Independent AT2/GT450 alignment validation...", flush=True)

real_results = []

for index, key in enumerate(
    sorted(pairs),
    1,
):
    pair = pairs[key]

    if (
        pair["AT2"]["spatial_split"]
        != pair["GT450"]["spatial_split"]
    ):
        raise RuntimeError(
            "Scanner pair split mismatch."
        )

    real_results.append(
        inspect_real_pair(
            pair["AT2"],
            pair["GT450"],
        )
    )

    if index % 250 == 0 or index == EXPECTED_PAIRS:
        print(
            f"  Validated {index}/{EXPECTED_PAIRS} pairs",
            flush=True,
        )


print("\n[4] Creating region-level summaries...", flush=True)

by_region = defaultdict(list)

for result in real_results:
    by_region[
        (
            result["region_id"],
            result["spatial_split"],
        )
    ].append(result)

region_results = []

for (region_id, split), group in sorted(
    by_region.items()
):
    if len(group) != 36:
        raise RuntimeError(
            f"Unexpected region size: {region_id}"
        )

    region_results.append({
        "region_id": region_id,
        "spatial_split": split,
        "pair_count": len(group),
        "bounded_shift_count": sum(
            r["shift_within_limit"]
            for r in group
        ),
        "median_shift_magnitude": median(
            [
                r["estimated_shift_magnitude"]
                for r in group
            ]
        ),
        "median_heldout_original_correlation": median(
            [
                r["heldout_original_correlation"]
                for r in group
            ]
        ),
        "median_heldout_shifted_correlation": median(
            [
                r["heldout_shifted_correlation"]
                for r in group
            ]
        ),
        "median_heldout_improvement": median(
            [
                r["heldout_improvement"]
                for r in group
            ]
        ),
        "heldout_improvement_count": sum(
            r["heldout_improvement_flag"]
            for r in group
        ),
    })

bounded = [
    r for r in real_results
    if r["shift_within_limit"]
]

improved = [
    r for r in bounded
    if r["heldout_improvement_flag"]
]

print("Region summaries:", len(region_results))
print("Bounded shift proposals:", len(bounded))
print("Independent improvements:", len(improved))

if bounded:
    print(
        "Independent improvement proportion:",
        f"{100 * len(improved) / len(bounded):.2f}%",
    )


print("\n[5] Saving reproducible research outputs...", flush=True)

REAL_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

save_csv(
    REAL_OUTPUT,
    real_results,
    REAL_FIELDS,
)

save_csv(
    SYNTHETIC_OUTPUT,
    synthetic_results,
    SYNTHETIC_FIELDS,
)

save_csv(
    REGION_OUTPUT,
    region_results,
    REGION_FIELDS,
)

split_summary = {}

for split in SPLITS:
    subset = [
        r for r in real_results
        if r["spatial_split"] == split
    ]

    split_bounded = [
        r for r in subset
        if r["shift_within_limit"]
    ]

    split_summary[split] = {
        "pairs": len(subset),
        "bounded_shift_proposals": len(split_bounded),
        "heldout_improvements": sum(
            r["heldout_improvement_flag"]
            for r in split_bounded
        ),
        "median_heldout_improvement": median(
            [
                r["heldout_improvement"]
                for r in split_bounded
            ]
        ),
    }

report = {
    "phase": "4",
    "step": "4.1.13",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "dataset": "owkin/plism-dataset-tiles",
    "source_audit_sha256": sha256_file(SOURCE),
    "seed": SEED,
    "images": EXPECTED_IMAGES,
    "paired_positions": EXPECTED_PAIRS,
    "regions": EXPECTED_REGIONS,
    "synthetic_experiment": {
        "cases": len(synthetic_results),
        "max_applied_shift_per_axis": SYNTHETIC_MAX_SHIFT,
        "recovered_within_one_pixel": synthetic_success,
        "recovery_rate": synthetic_rate,
        "error_distribution": distribution(
            [
                r["euclidean_error"]
                for r in synthetic_results
            ]
        ),
    },
    "real_pair_experiment": {
        "left_half_used_for_shift_estimation": True,
        "right_half_used_for_independent_evaluation": True,
        "maximum_accepted_shift_per_axis": MAX_REAL_SHIFT,
        "minimum_improvement": MIN_HELDOUT_IMPROVEMENT,
        "bounded_shift_proposals": len(bounded),
        "independent_improvements": len(improved),
        "improvement_distribution": distribution(
            [
                r["heldout_improvement"]
                for r in bounded
            ]
        ),
    },
    "split_summary": split_summary,
    "real_metrics_sha256": sha256_file(REAL_OUTPUT),
    "synthetic_metrics_sha256": sha256_file(SYNTHETIC_OUTPUT),
    "region_metrics_sha256": sha256_file(REGION_OUTPUT),
    "original_pngs_modified": False,
    "excluded_image_pairs": 0,
    "dataset_approved_for_training": False,
    "biological_independence_verified": False,
    "manual_registration_review_completed": False,
    "limitations": [
        "Synthetic recovery validates only tested transformations",
        "Estimated real-image shifts are not ground truth",
        "Left and right image halves are spatially related",
        "Independent evaluation does not imply specimen independence",
        "Scanner appearance affects gradient magnitude",
        "Phase correlation may fail under nonlinear distortion",
        "Bounded shifts exclude potentially larger displacements",
        "No automatic pair correction or exclusion is justified",
    ],
}

save_json(
    REPORT_OUTPUT,
    report,
)

print("\n" + "=" * 70)
print("PHASE 4.1.13 - RELIABILITY AUDIT COMPLETED")
print("=" * 70)

print("Synthetic cases:", len(synthetic_results))
print(
    "Synthetic recovery rate:",
    f"{100 * synthetic_rate:.2f}%",
)

print("Real scanner pairs:", len(real_results))
print("Bounded shift proposals:", len(bounded))
print("Independent improvements:", len(improved))

if bounded:
    print(
        "Median independent edge-correlation improvement:",
        median([
            r["heldout_improvement"]
            for r in bounded
        ]),
    )

print("\nSaved:")
print(SYNTHETIC_OUTPUT)
print(REAL_OUTPUT)
print(REGION_OUTPUT)
print(REPORT_OUTPUT)

print("\nOriginal PNGs: UNCHANGED")
print("Pairs excluded: 0")
print("Dataset approval: PENDING")
print("Biological independence: UNVERIFIED")
print("\nSTEP 4.1.13 COMPLETED")
