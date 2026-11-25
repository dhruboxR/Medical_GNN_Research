
"""
PHASE 4.1.10 - COMPLETE STAGE A ORIGINAL PNG DOWNLOAD

Download and validate the planned 4,320 original PNGs.

Cohort:
    60 regions
    2,160 tile positions
    GIVH AT2 and GT450
    4,320 original PNGs

Safety:
    - Pinned Hugging Face dataset revision
    - Exact source slide/tile/row verification
    - Original PNG bytes preserved
    - PNG signature, dimensions, decoding verified
    - SHA256 hashes recorded
    - Existing files never overwritten
    - Resumable per-row-group checkpoints
    - Completed checkpoint files independently verified
    - No Phase 3B modifications

No feature extraction or model training.
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

IMAGE_ROOT = BASE / "datasets/expanded_plism/stage_a"

CHECKPOINT_DIR = (
    BASE / "datasets/metadata"
    / "10_download_checkpoints"
)

AUDIT = (
    BASE / "outputs/tables"
    / "10_stage_a_original_png_audit.csv"
)
SUMMARY = (
    BASE / "outputs/tables"
    / "10_stage_a_download_summary.csv"
)
REPORT = (
    BASE / "outputs/reports"
    / "10_stage_a_download_report.json"
)

DATASET = "owkin/plism-dataset-tiles"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

EXPECTED_TOTAL = 4320
EXPECTED_POSITIONS = 2160
EXPECTED_REGIONS = 60

SCANNERS = {"AT2", "GT450"}
SPLITS = {"train", "validation", "test"}

FINAL_OUTPUTS = (AUDIT, SUMMARY, REPORT)


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv_exclusive(path, rows, columns):
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_json_exclusive(path, data):
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def image_bytes(value):
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)

    if isinstance(value, dict):
        raw = value.get("bytes")
        if isinstance(raw, (bytes, bytearray)):
            return bytes(raw)

    raise RuntimeError("Unexpected embedded PNG representation.")


def validate_png(raw):
    if not raw.startswith(PNG_MAGIC):
        raise RuntimeError("Invalid PNG signature.")

    with Image.open(BytesIO(raw)) as img:
        if img.format != "PNG":
            raise RuntimeError("Unexpected image format.")
        img.verify()

    with Image.open(BytesIO(raw)) as img:
        if img.size != (224, 224):
            raise RuntimeError(f"Unexpected dimensions: {img.size}")
        img.load()
        mode = img.mode

    return mode


def safe_path(relative_name):
    relative = Path(relative_name)

    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("Unsafe destination path.")

    result = (BASE / relative).resolve()
    root = IMAGE_ROOT.resolve()

    if not result.is_relative_to(root):
        raise RuntimeError("Image destination outside Stage A.")

    if result.suffix.lower() != ".png":
        raise RuntimeError("Destination is not a PNG.")

    return result


def save_original_new(destination, raw):
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = None

    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=".partial_",
            suffix=".png",
            delete=False,
        ) as f:
            temporary = Path(f.name)
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())

        # Atomic create-only operation.
        os.link(temporary, destination)

    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def checkpoint_path(filename, row_group):
    key = f"{filename}|{row_group}".encode("utf-8")
    return CHECKPOINT_DIR / (
        hashlib.sha256(key).hexdigest()[:32] + ".json"
    )


def validate_checkpoint(path, filename, row_group,
                        expected_items, revision,
                        manifest_hash):
    with path.open(encoding="utf-8") as f:
        checkpoint = json.load(f)

    if (
        checkpoint.get("filename") != filename
        or checkpoint.get("row_group") != row_group
        or checkpoint.get("revision") != revision
        or checkpoint.get("manifest_sha256") != manifest_hash
    ):
        raise RuntimeError(f"Incompatible checkpoint: {path}")

    records = checkpoint.get("records", [])

    expected = {
        (x["slide_id"], x["tile_id"])
        for x in expected_items
    }
    found = {
        (x["slide_id"], x["tile_id"])
        for x in records
    }

    if found != expected or len(records) != len(expected_items):
        raise RuntimeError(f"Incomplete checkpoint: {path}")

    for record in records:
        destination = safe_path(
            record["destination_relative_path"]
        )

        if not destination.is_file():
            raise RuntimeError(
                f"Checkpoint image missing: {destination}"
            )

        if digest_file(destination) != record["sha256"]:
            raise RuntimeError(
                f"Checkpoint image hash mismatch: {destination}"
            )

    return records


print("=" * 70, flush=True)
print("PHASE 4.1.10 - COMPLETE STAGE A DOWNLOAD", flush=True)
print("=" * 70, flush=True)

print("\n[1] Validating frozen download plan...", flush=True)

for path in (MANIFEST, PREVIOUS):
    if not path.is_file():
        raise RuntimeError(f"Missing research input: {path}")

for path in FINAL_OUTPUTS:
    if path.exists():
        raise RuntimeError(
            f"Refusing to overwrite completed output: {path}"
        )

with PREVIOUS.open(encoding="utf-8") as f:
    previous = json.load(f)

manifest_hash = digest_file(MANIFEST)

if manifest_hash != previous["download_manifest_sha256"]:
    raise RuntimeError("Download manifest SHA256 mismatch.")

items = read_csv(MANIFEST)

if len(items) != EXPECTED_TOTAL:
    raise RuntimeError("Expected 4320 image records.")

revision = previous["dataset_revision"]

regions = set()
positions = defaultdict(set)
destinations = set()

for item in items:
    if item["dataset_revision"] != revision:
        raise RuntimeError("Mixed dataset revisions.")

    if item["scanner"] not in SCANNERS:
        raise RuntimeError("Unexpected scanner.")

    if item["source_condition"] != "GIVH":
        raise RuntimeError("Unexpected condition.")

    if item["spatial_split"] not in SPLITS:
        raise RuntimeError("Unexpected spatial split.")

    regions.add(item["region_id"])
    positions[item["scanner"]].add(item["tile_id"])

    path = safe_path(item["destination_relative_path"])

    if str(path) in destinations:
        raise RuntimeError("Duplicate destination.")

    destinations.add(str(path))

if len(regions) != EXPECTED_REGIONS:
    raise RuntimeError("Expected 60 regions.")

if positions["AT2"] != positions["GT450"]:
    raise RuntimeError("Scanner position mismatch.")

if len(positions["AT2"]) != EXPECTED_POSITIONS:
    raise RuntimeError("Expected 2160 paired positions.")

print("Regions:", len(regions))
print("Paired positions:", EXPECTED_POSITIONS)
print("Images expected:", EXPECTED_TOTAL)
print("Pinned revision:", revision)


print("\n[2] Organizing source Parquet requests...", flush=True)

by_shard = defaultdict(lambda: defaultdict(list))

for item in items:
    filename = item["repository_file"]
    rg = int(item["row_group"])

    if not filename.startswith("data/") or not filename.endswith(".parquet"):
        raise RuntimeError("Unexpected Parquet source path.")

    if rg < 0 or int(item["row_in_group"]) < 0:
        raise RuntimeError("Negative source row index.")

    by_shard[filename][rg].append(item)

total_groups = sum(len(v) for v in by_shard.values())

print("Remote shards:", len(by_shard))
print("Required row groups:", total_groups)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

fs = fsspec.filesystem(
    "http",
    block_size=256 * 1024,
    timeout=90,
)

audit_records = []
groups_done = 0
reused_groups = 0
new_files = 0
verified_existing = 0


print("\n[3] Downloading original images...", flush=True)

for shard_index, filename in enumerate(sorted(by_shard), 1):
    group_map = by_shard[filename]
    pending = []

    # Reuse only checkpointed files whose hashes pass.
    for rg, group_items in sorted(group_map.items()):
        cp = checkpoint_path(filename, rg)

        if cp.exists():
            prior_records = validate_checkpoint(
                cp, filename, rg, group_items,
                revision, manifest_hash,
            )
            audit_records.extend(prior_records)
            groups_done += 1
            reused_groups += 1
        else:
            pending.append((rg, group_items))

    if pending:
        url = (
            "https://huggingface.co/datasets/"
            f"{DATASET}/resolve/{revision}/"
            f"{quote(filename, safe='/')}"
        )

        try:
            with fs.open(
                url, "rb",
                block_size=256 * 1024,
            ) as stream:
                parquet = pq.ParquetFile(stream)

                required_columns = {"slide_id", "tile_id", "png"}

                if not required_columns.issubset(
                    set(parquet.schema_arrow.names)
                ):
                    raise RuntimeError("Unexpected Parquet schema.")

                for rg, group_items in pending:
                    table = parquet.read_row_group(
                        rg,
                        columns=["slide_id", "tile_id", "png"],
                    )
                    columns = table.to_pydict()
                    group_audit = []

                    for item in group_items:
                        row_index = int(item["row_in_group"])

                        if not (0 <= row_index < table.num_rows):
                            raise RuntimeError("Invalid row position.")

                        if columns["slide_id"][row_index] != item["slide_id"]:
                            raise RuntimeError("Source slide ID mismatch.")

                        if columns["tile_id"][row_index] != item["tile_id"]:
                            raise RuntimeError("Source tile ID mismatch.")

                        raw = image_bytes(columns["png"][row_index])
                        mode = validate_png(raw)
                        source_hash = digest_bytes(raw)

                        destination = safe_path(
                            item["destination_relative_path"]
                        )

                        if destination.exists():
                            if not destination.is_file():
                                raise RuntimeError(
                                    "Existing destination is not a file."
                                )

                            if digest_file(destination) != source_hash:
                                raise RuntimeError(
                                    "Existing PNG differs from source: "
                                    f"{destination}"
                                )

                            status = "existing_verified"
                            verified_existing += 1
                        else:
                            try:
                                save_original_new(destination, raw)
                                status = "downloaded"
                                new_files += 1

                            except FileExistsError:
                                # Another process created the file.
                                status = "existing_verified"
                                verified_existing += 1

                            if digest_file(destination) != source_hash:
                                raise RuntimeError(
                                    "Saved PNG hash mismatch."
                                )

                        group_audit.append({
                            "region_id": item["region_id"],
                            "spatial_split": item["spatial_split"],
                            "scanner": item["scanner"],
                            "slide_id": item["slide_id"],
                            "tile_id": item["tile_id"],
                            "repository_file": filename,
                            "row_group": rg,
                            "row_in_group": row_index,
                            "destination_relative_path":
                                item["destination_relative_path"],
                            "sha256": source_hash,
                            "size_bytes": len(raw),
                            "image_mode": mode,
                            "width": 224,
                            "height": 224,
                            "status": status,
                        })

                    cp = checkpoint_path(filename, rg)

                    save_json_exclusive(cp, {
                        "filename": filename,
                        "row_group": rg,
                        "revision": revision,
                        "manifest_sha256": manifest_hash,
                        "records": group_audit,
                    })

                    audit_records.extend(group_audit)
                    groups_done += 1

        except Exception as exc:
            raise RuntimeError(
                f"Download failed at {filename}: "
                f"{type(exc).__name__}: {exc}\n"
                "Completed checkpoints and original PNGs "
                "were preserved. Rerun to resume."
            ) from exc

    print(
        f"  Shards: {shard_index}/{len(by_shard)} | "
        f"Row groups: {groups_done}/{total_groups} | "
        f"Validated images: {len(audit_records)}/{EXPECTED_TOTAL}",
        flush=True,
    )


print("\n[4] Running final pairing and integrity checks...", flush=True)

if len(audit_records) != EXPECTED_TOTAL:
    raise RuntimeError("Incomplete image download.")

record_keys = {
    (r["scanner"], r["tile_id"])
    for r in audit_records
}

if len(record_keys) != EXPECTED_TOTAL:
    raise RuntimeError("Duplicate or missing scanner/tile records.")

for tile_id in positions["AT2"]:
    if (
        ("AT2", tile_id) not in record_keys
        or ("GT450", tile_id) not in record_keys
    ):
        raise RuntimeError(f"Missing scanner pair: {tile_id}")

counts_by_split = Counter(
    r["spatial_split"]
    for r in audit_records
)

expected_split_counts = {
    "train": 2880,
    "validation": 720,
    "test": 720,
}

if dict(counts_by_split) != expected_split_counts:
    raise RuntimeError(
        f"Split counts mismatch: {counts_by_split}"
    )

print("Validated original images:", len(audit_records))
print("Complete scanner pairs:", EXPECTED_POSITIONS)
print("New image files:", new_files)
print("Existing source-verified files:", verified_existing)
print("Checkpointed row groups reused:", reused_groups)


print("\n[5] Saving final Stage A provenance...", flush=True)

AUDIT.parent.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

audit_records.sort(
    key=lambda r: (
        r["region_id"],
        r["tile_id"],
        r["scanner"],
    )
)

save_csv_exclusive(
    AUDIT,
    audit_records,
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
        "image_mode",
        "width",
        "height",
        "status",
    ],
)

summary_rows = []

for split in ("train", "validation", "test"):
    subset = [
        r for r in audit_records
        if r["spatial_split"] == split
    ]

    summary_rows.append({
        "spatial_split": split,
        "images": len(subset),
        "at2_images": sum(
            r["scanner"] == "AT2" for r in subset
        ),
        "gt450_images": sum(
            r["scanner"] == "GT450" for r in subset
        ),
        "total_png_bytes": sum(
            r["size_bytes"] for r in subset
        ),
    })

save_csv_exclusive(
    SUMMARY,
    summary_rows,
    [
        "spatial_split",
        "images",
        "at2_images",
        "gt450_images",
        "total_png_bytes",
    ],
)

total_bytes = sum(
    r["size_bytes"] for r in audit_records
)

report = {
    "phase": "4",
    "step": "4.1.10",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "dataset": DATASET,
    "revision": revision,
    "regions": len(regions),
    "paired_positions": EXPECTED_POSITIONS,
    "verified_original_pngs": len(audit_records),
    "new_image_files": new_files,
    "existing_source_verified_files": verified_existing,
    "reused_checkpoint_groups": reused_groups,
    "total_png_bytes": total_bytes,
    "total_png_gib": total_bytes / (1024 ** 3),
    "source_manifest_sha256": manifest_hash,
    "final_audit_sha256": digest_file(AUDIT),
    "original_png_bytes_preserved": True,
    "all_pairs_verified": True,
    "biological_independence_verified": False,
    "manual_registration_review_completed": False,
    "evaluation_protocol_frozen": False,
    "phase3b_modified": False,
    "next_step": (
        "Generate image-quality and paired contact-sheet "
        "audits before feature extraction and model training."
    ),
}

save_json_exclusive(REPORT, report)

print("\n" + "=" * 70)
print("PHASE 4.1.10 - STAGE A DOWNLOAD COMPLETED")
print("=" * 70)

print("Validated original PNGs:", len(audit_records))
print("Matched tile pairs:", EXPECTED_POSITIONS)
print("Actual PNG storage (GiB):", f"{report['total_png_gib']:.3f}")

print("\nSaved:")
print(AUDIT)
print(SUMMARY)
print(REPORT)

print("\nPhase 3B: UNCHANGED")
print("Manual registration review: PENDING")
print("Biological independence: UNVERIFIED")
print("\nSTEP 4.1.10 COMPLETED")
