
"""
PHASE 4.1.9 - ORIGINAL PNG DOWNLOAD PILOT

Download one complete 6x6 Phase 4 training region
for GIVH AT2 and GT450.

Expected:
    36 AT2 PNG images
    36 GT450 PNG images
    72 total original PNG images

Read only the exact Parquet row groups needed.

Safety:
- Verify pinned repository revision
- Verify slide_id and tile_id at each row position
- Preserve original PNG bytes
- Validate PNG signature and dimensions
- Record SHA256 hashes
- Never overwrite existing image files
- Support resuming a partial download
- Leave Phase 3B unchanged
"""

from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
from urllib.parse import quote
from io import BytesIO

import csv
import hashlib
import json
import os
import tempfile

import fsspec
import pyarrow.parquet as pq
from PIL import Image


ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "phase4_main_research"

MANIFEST = (
    BASE / "datasets/manifests"
    / "08_stage_a_original_png_download_manifest.csv"
)

PREVIOUS = (
    BASE / "outputs/reports"
    / "08_stage_a_download_manifest_audit.json"
)

IMAGE_ROOT = (
    BASE / "datasets/expanded_plism/stage_a"
)

AUDIT_PATH = (
    BASE / "outputs/tables"
    / "09_stage_a_pilot_download_audit.csv"
)

REPORT_PATH = (
    BASE / "outputs/reports"
    / "09_stage_a_pilot_download_report.json"
)

DATASET_ID = "owkin/plism-dataset-tiles"

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

EXPECTED_PER_SCANNER = 36
EXPECTED_TOTAL = 72

OUTPUTS = (AUDIT_PATH, REPORT_PATH)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


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


def write_csv_new(path, rows, columns):
    with path.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=columns,
        )
        writer.writeheader()
        writer.writerows(rows)


def extract_png_bytes(value):
    """
    A Hugging Face image field may be stored as:
        struct<bytes: binary, path: string>

    We require embedded original PNG bytes.
    External image paths are not accepted.
    """

    if isinstance(value, (bytes, bytearray)):
        return bytes(value)

    if isinstance(value, dict):
        raw = value.get("bytes")

        if isinstance(raw, (bytes, bytearray)):
            return bytes(raw)

    raise RuntimeError(
        "Unexpected PNG storage format. "
        "Expected embedded binary image bytes."
    )


def inspect_png(data):
    if not data.startswith(PNG_SIGNATURE):
        raise RuntimeError(
            "Image does not have a PNG signature."
        )

    with Image.open(BytesIO(data)) as image:
        if image.format != "PNG":
            raise RuntimeError(
                "Image format is not PNG."
            )

        image.verify()

    with Image.open(BytesIO(data)) as image:
        if image.size != (224, 224):
            raise RuntimeError(
                f"Unexpected image dimensions: {image.size}"
            )

        image.load()
        mode = image.mode

    return {
        "width": 224,
        "height": 224,
        "mode": mode,
    }


def safe_destination(relative_path):
    relative = Path(relative_path)

    if relative.is_absolute():
        raise RuntimeError(
            "Absolute image destination not allowed."
        )

    destination = (
        BASE / relative
    ).resolve()

    root = IMAGE_ROOT.resolve()

    if not destination.is_relative_to(root):
        raise RuntimeError(
            "Destination outside Stage A image directory."
        )

    return destination


def atomic_write_new(path, data):
    """
    Write a temporary file, then hard-link it to the
    destination using create-only semantics.

    Existing destination files are never overwritten.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = None

    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=".partial_",
            suffix=".png",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

        os.link(temporary, path)

    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


print("=" * 70, flush=True)
print("PHASE 4.1.9 - ORIGINAL PNG DOWNLOAD PILOT", flush=True)
print("=" * 70, flush=True)

print("\n[1] Validating manifest and research inputs...", flush=True)

for path in (MANIFEST, PREVIOUS):
    if not path.is_file():
        raise RuntimeError(
            f"Missing input: {path}"
        )

for path in OUTPUTS:
    if path.exists():
        raise RuntimeError(
            f"Refusing to overwrite completed result: {path}"
        )

with PREVIOUS.open(
    encoding="utf-8"
) as stream:
    previous = json.load(stream)

if previous.get("planned_original_pngs") != 4320:
    raise RuntimeError(
        "Previous download manifest audit mismatch."
    )

if (
    sha256_file(MANIFEST)
    != previous.get("download_manifest_sha256")
):
    raise RuntimeError(
        "Download manifest SHA256 mismatch."
    )

records = read_csv(MANIFEST)

if len(records) != 4320:
    raise RuntimeError(
        "Expected 4320 manifest records."
    )

training_regions = sorted({
    row["region_id"]
    for row in records
    if row["spatial_split"] == "train"
})

if len(training_regions) != 40:
    raise RuntimeError(
        "Expected 40 training regions."
    )

pilot_region = training_regions[0]

pilot = [
    row for row in records
    if row["region_id"] == pilot_region
]

if len(pilot) != EXPECTED_TOTAL:
    raise RuntimeError(
        "Pilot must contain 72 image records."
    )

counts = Counter(
    row["scanner"] for row in pilot
)

if counts != {
    "AT2": EXPECTED_PER_SCANNER,
    "GT450": EXPECTED_PER_SCANNER,
}:
    raise RuntimeError(
        "Pilot scanner counts are incorrect."
    )

positions = defaultdict(set)

for row in pilot:
    if row["spatial_split"] != "train":
        raise RuntimeError(
            "Pilot selected outside training split."
        )

    if row["source_condition"] != "GIVH":
        raise RuntimeError(
            "Unexpected staining condition."
        )

    positions[row["scanner"]].add(
        row["tile_id"]
    )

if (
    positions["AT2"] != positions["GT450"]
    or len(positions["AT2"]) != 36
):
    raise RuntimeError(
        "Scanner tile pairing mismatch."
    )

revisions = {
    row["dataset_revision"]
    for row in pilot
}

if revisions != {
    previous["dataset_revision"]
}:
    raise RuntimeError(
        "Dataset revision mismatch."
    )

revision = next(iter(revisions))

print("Pilot region:", pilot_region)
print("Scanner pair: AT2 / GT450")
print("Paired positions:", len(positions["AT2"]))
print("Images requested:", len(pilot))
print("Pinned revision:", revision)


# ============================================================
# GROUP EXACT PARQUET READS
# ============================================================

requests = defaultdict(list)

for row in pilot:
    key = (
        row["repository_file"],
        int(row["row_group"]),
    )

    requests[key].append(row)

print("\n[2] Grouping remote Parquet requests...", flush=True)

print("Unique row groups:", len(requests))
print(
    "Unique remote shards:",
    len({key[0] for key in requests}),
)

by_shard = defaultdict(dict)

for (filename, rowgroup), items in requests.items():
    by_shard[filename][rowgroup] = items

filesystem = fsspec.filesystem(
    "http",
    block_size=256 * 1024,
    timeout=90,
)

audit_records = []
completed = 0
downloaded = 0
reused = 0

print("\n[3] Retrieving and validating original PNGs...", flush=True)

for shard_index, filename in enumerate(
    sorted(by_shard), start=1
):

    url = (
        "https://huggingface.co/datasets/"
        f"{DATASET_ID}/resolve/{revision}/"
        f"{quote(filename, safe='/')}"
    )

    print(
        f"  Shard {shard_index}/{len(by_shard)}: "
        f"{filename}",
        flush=True,
    )

    try:
        with filesystem.open(
            url,
            "rb",
            block_size=256 * 1024,
        ) as stream:

            parquet = pq.ParquetFile(stream)

            required = {"slide_id", "tile_id", "png"}

            if not required.issubset(
                set(parquet.schema_arrow.names)
            ):
                raise RuntimeError(
                    "Missing required Parquet columns."
                )

            for rg, items in sorted(
                by_shard[filename].items()
            ):

                # Reading the PNG column decompresses
                # the required row group's image values.
                # This pilot limits how many groups are read.
                table = parquet.read_row_group(
                    rg,
                    columns=[
                        "slide_id",
                        "tile_id",
                        "png",
                    ],
                )

                cols = table.to_pydict()

                for row in items:
                    position = int(
                        row["row_in_group"]
                    )

                    if (
                        position < 0
                        or position >= table.num_rows
                    ):
                        raise RuntimeError(
                            "Invalid Parquet row position."
                        )

                    actual_slide = cols[
                        "slide_id"
                    ][position]

                    actual_tile = cols[
                        "tile_id"
                    ][position]

                    if actual_slide != row["slide_id"]:
                        raise RuntimeError(
                            f"Slide ID mismatch: {actual_slide}"
                        )

                    if actual_tile != row["tile_id"]:
                        raise RuntimeError(
                            f"Tile ID mismatch: {actual_tile}"
                        )

                    png_bytes = extract_png_bytes(
                        cols["png"][position]
                    )

                    png_info = inspect_png(
                        png_bytes
                    )

                    digest = sha256_bytes(
                        png_bytes
                    )

                    destination = safe_destination(
                        row["destination_relative_path"]
                    )

                    if destination.exists():
                        if not destination.is_file():
                            raise RuntimeError(
                                "Destination exists but is not a file."
                            )

                        if sha256_file(destination) != digest:
                            raise RuntimeError(
                                "Existing image hash differs from source: "
                                f"{destination}"
                            )

                        status = "existing_verified"
                        reused += 1

                    else:
                        atomic_write_new(
                            destination,
                            png_bytes,
                        )

                        if sha256_file(destination) != digest:
                            raise RuntimeError(
                                "Written PNG hash mismatch."
                            )

                        status = "downloaded"
                        downloaded += 1

                    audit_records.append({
                        "region_id": pilot_region,
                        "spatial_split": "train",
                        "scanner": row["scanner"],
                        "slide_id": row["slide_id"],
                        "tile_id": row["tile_id"],
                        "repository_file": filename,
                        "row_group": rg,
                        "row_in_group": position,
                        "destination_relative_path":
                            row["destination_relative_path"],
                        "sha256": digest,
                        "size_bytes": len(png_bytes),
                        "width": png_info["width"],
                        "height": png_info["height"],
                        "image_mode": png_info["mode"],
                        "status": status,
                    })

                    completed += 1

    except Exception as exc:
        raise RuntimeError(
            f"Pilot download failed at {filename}: "
            f"{type(exc).__name__}: {exc}\n"
            "Already saved PNGs are preserved. "
            "Rerun this script to validate/reuse them."
        ) from exc

    print(
        f"    Images validated: {completed}/{EXPECTED_TOTAL}",
        flush=True,
    )


# ============================================================
# FINAL PAIRING AND IMAGE AUDIT
# ============================================================

print("\n[4] Verifying complete scanner pairing...", flush=True)

if len(audit_records) != EXPECTED_TOTAL:
    raise RuntimeError(
        "Incomplete pilot image audit."
    )

keys = {
    (r["scanner"], r["tile_id"])
    for r in audit_records
}

if len(keys) != EXPECTED_TOTAL:
    raise RuntimeError(
        "Duplicate or missing image pairing."
    )

for tile_id in positions["AT2"]:
    if (
        ("AT2", tile_id) not in keys
        or ("GT450", tile_id) not in keys
    ):
        raise RuntimeError(
            f"Incomplete pair: {tile_id}"
        )

if any(
    record["width"] != 224
    or record["height"] != 224
    for record in audit_records
):
    raise RuntimeError(
        "Unexpected PNG dimensions."
    )

print("Original PNGs validated:", completed)
print("Complete scanner pairs:", len(positions["AT2"]))
print("New image files:", downloaded)
print("Existing verified files:", reused)


# ============================================================
# SAVE AUDIT
# ============================================================

print("\n[5] Saving pilot provenance...", flush=True)

AUDIT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

write_csv_new(
    AUDIT_PATH,
    sorted(
        audit_records,
        key=lambda r: (
            r["tile_id"],
            r["scanner"],
        ),
    ),
    [
        "region_id",
        "spatial_split",
        "scanner",
        "slide_id",
        "tile_id",
        "repository_file",
        "row_group",
        "row_in_group",
        "destination_relative_path",
        "sha256",
        "size_bytes",
        "width",
        "height",
        "image_mode",
        "status",
    ],
)

report = {
    "phase": "4",
    "step": "4.1.9",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "dataset": DATASET_ID,
    "dataset_revision": revision,
    "pilot_region": pilot_region,
    "training_partition_only": True,
    "requested_images": EXPECTED_TOTAL,
    "validated_images": completed,
    "matched_tile_pairs": len(positions["AT2"]),
    "downloaded_images": downloaded,
    "reused_verified_images": reused,
    "unique_parquet_rowgroups": len(requests),
    "unique_parquet_shards": len(by_shard),
    "source_manifest_sha256": sha256_file(MANIFEST),
    "audit_sha256": sha256_file(AUDIT_PATH),
    "original_png_bytes_preserved": True,
    "all_images_dimensions_verified": True,
    "all_images_sha256_verified": True,
    "biological_independence_verified": False,
    "manual_registration_review_completed": False,
    "final_train_test_protocol_frozen": False,
    "phase3b_modified": False,
    "next_step": (
        "Inspect scanner-paired pilot contact sheet, "
        "then perform resumable Stage A image download "
        "with verification and provenance."
    ),
}

with REPORT_PATH.open(
    "x",
    encoding="utf-8",
) as stream:
    json.dump(
        report,
        stream,
        indent=2,
    )

print("\n" + "=" * 70)
print("PHASE 4.1.9 - ORIGINAL PNG PILOT COMPLETED")
print("=" * 70)

print("Region:", pilot_region)
print("Validated original images:", completed)
print("Matched scanner pairs:", len(positions["AT2"]))
print("New files:", downloaded)
print("Reused files:", reused)

print("\nSaved:")
print(AUDIT_PATH)
print(REPORT_PATH)

print("\nPhase 3B data: UNCHANGED")
print("Original PNG bytes: PRESERVED")
print("Independent biological samples: UNVERIFIED")
print("Manual registration review: PENDING")
print("\nSTEP 4.1.9 COMPLETED")
