
"""
PHASE 3B - STEP 3B.17
Resumable Original PLISM PNG Retrieval

Default:
    Process ONE verified Parquet row group.

Full retrieval:
    python script.py --all

Features:
    - Frozen manifest SHA256 validation
    - Exact source-location validation
    - HTTP byte-range access
    - One PNG column read per row group
    - Original PNG-byte preservation
    - PNG decoding and dimension checks
    - SHA256 logging
    - Atomic file writes
    - Safe resumability without overwriting
    - CSV and JSON progress reports

The default run is a small feasibility batch.
"""

from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
from io import BytesIO
import argparse
import hashlib
import io
import json
import os
import tempfile

import pandas as pd
import pyarrow.parquet as pq
import requests
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
BASE = ROOT / "multi_region_research"

TABLES = BASE / "outputs/tables"
REPORTS = BASE / "outputs/reports"

IMAGE_ROOT = (
    ROOT / "datasets/PLISM/phase3b_original_png"
)

MANIFEST_PATH = (
    BASE / "outputs/validated/"
    "phase3b_frozen_tile_manifest.csv"
)

LOCATIONS_PATH = (
    TABLES / "plism_exact_parquet_tile_locations.csv"
)

IMAGE_AUDIT_PATH = (
    TABLES / "plism_original_png_download_audit.csv"
)

REPORT_PATH = (
    REPORTS / "17_original_png_retrieval_progress.json"
)

EXPECTED_MANIFEST_HASH = (
    "69d11e1a0dd1107f2e6534632727e2c2291ff83d6b17"
    "cf953ed5056bea9020f9"
)

BASE_URL = (
    "https://huggingface.co/datasets/"
    "owkin/plism-dataset-tiles/resolve/main/data/"
)

EXPECTED_SLIDES = {
    "AT2": "GIVH_AT2_to_GMH_S60",
    "GT450": "GIVH_GT450_to_GMH_S60",
}

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

TIMEOUT = (10, 40)

MAX_SINGLE_READ = 20 * 1024 * 1024
MAX_SHARD_TRANSFER = 120 * 1024 * 1024


# ============================================================
# HELPERS
# ============================================================

def sha256(data):
    return hashlib.sha256(data).hexdigest()


def validate_png(raw):
    if not raw.startswith(PNG_SIGNATURE):
        raise RuntimeError("Missing PNG signature.")

    with Image.open(BytesIO(raw)) as image:
        if image.format != "PNG":
            raise RuntimeError("Invalid image encoding.")
        image.verify()

    with Image.open(BytesIO(raw)) as image:
        image.load()

        if image.size != (224, 224):
            raise RuntimeError(
                f"Unexpected dimensions: {image.size}"
            )

        if image.mode not in ("RGB", "RGBA"):
            raise RuntimeError(
                f"Unexpected image mode: {image.mode}"
            )

        mode = image.mode

    return mode


def save_original_png(destination, raw):
    """
    Write original image bytes without re-encoding.

    Existing images must match byte-for-byte.
    """

    digest = sha256(raw)

    if destination.exists():
        existing_digest = sha256(
            destination.read_bytes()
        )

        if existing_digest != digest:
            raise RuntimeError(
                f"Existing file hash mismatch: {destination}"
            )

        return digest, "EXISTING_VERIFIED"

    destination.parent.mkdir(
        parents=True, exist_ok=True
    )

    # Create a temporary file in the same directory.
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=".partial_",
        suffix=".png",
        delete=False,
    ) as handle:

        temporary_path = Path(handle.name)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())

    try:
        # Exclusive creation avoids overwriting a file
        # that appeared while the temporary file was written.
        try:
            with destination.open("xb") as out:
                out.write(temporary_path.read_bytes())

        except FileExistsError:
            if sha256(destination.read_bytes()) != digest:
                raise RuntimeError(
                    "Conflicting existing image."
                )

            return digest, "EXISTING_VERIFIED"

    finally:
        temporary_path.unlink(missing_ok=True)

    return digest, "DOWNLOADED"


# ============================================================
# HTTP RANGE FILE
# ============================================================

class HTTPRangeFile(io.RawIOBase):

    def __init__(self, url, session):
        super().__init__()

        self.url = url
        self.session = session

        self.position = 0
        self.requests_made = 0
        self.bytes_received = 0

        response = session.head(
            url,
            allow_redirects=True,
            timeout=TIMEOUT,
        )
        response.raise_for_status()

        length = response.headers.get(
            "Content-Length"
        )

        if not length or not length.isdigit():
            raise RuntimeError(
                "Remote file length unavailable."
            )

        self.file_size = int(length)

    def readable(self):
        return True

    def seekable(self):
        return True

    def writable(self):
        return False

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):

        if whence == io.SEEK_SET:
            position = offset

        elif whence == io.SEEK_CUR:
            position = self.position + offset

        elif whence == io.SEEK_END:
            position = self.file_size + offset

        else:
            raise ValueError("Invalid seek mode.")

        if not 0 <= position <= self.file_size:
            raise ValueError("Invalid remote seek.")

        self.position = int(position)
        return self.position

    def read(self, size=-1):

        if size is None or size < 0:
            raise RuntimeError(
                "Unbounded remote reads disabled."
            )

        if size == 0 or self.position >= self.file_size:
            return b""

        size = min(
            size,
            self.file_size - self.position
        )

        if size > MAX_SINGLE_READ:
            raise RuntimeError(
                f"Single read exceeds limit: {size}"
            )

        if (
            self.bytes_received + size
            > MAX_SHARD_TRANSFER
        ):
            raise RuntimeError(
                "Shard transfer budget exceeded. "
                "Process smaller batches."
            )

        start = self.position
        end = start + size - 1

        with self.session.get(
            self.url,
            headers={
                "Range": f"bytes={start}-{end}",
                "Accept-Encoding": "identity",
            },
            stream=True,
            timeout=TIMEOUT,
            allow_redirects=True,
        ) as response:

            if response.status_code != 206:
                raise RuntimeError(
                    f"Expected HTTP 206; "
                    f"received {response.status_code}"
                )

            expected = f"bytes {start}-{end}/"
            actual = response.headers.get(
                "Content-Range", ""
            )

            if not actual.startswith(expected):
                raise RuntimeError(
                    f"Unexpected Content-Range: {actual}"
                )

            data = bytearray()

            for chunk in response.iter_content(65536):
                data.extend(chunk)

                if len(data) > size:
                    raise RuntimeError(
                        "Unexpected response size."
                    )

        if len(data) != size:
            raise RuntimeError(
                f"Incomplete response: {len(data)}/{size}"
            )

        self.position += size
        self.requests_made += 1
        self.bytes_received += size

        return bytes(data)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--all",
    action="store_true",
    help="Process all row groups (after pilot validation)."
)

args = parser.parse_args()

MAX_GROUPS = None if args.all else 1


# ============================================================
# STEP 1: VALIDATE INPUTS
# ============================================================

print("=" * 70, flush=True)
print("PHASE 3B - ORIGINAL PNG RETRIEVAL", flush=True)
print("=" * 70, flush=True)

print("\n[1] Validating frozen research cohort...", flush=True)

if not MANIFEST_PATH.is_file():
    raise RuntimeError("Frozen manifest missing.")

if not LOCATIONS_PATH.is_file():
    raise RuntimeError("Exact source locations missing.")

manifest_hash = sha256(
    MANIFEST_PATH.read_bytes()
)

if manifest_hash != EXPECTED_MANIFEST_HASH:
    raise RuntimeError(
        "Frozen manifest SHA256 mismatch."
    )

manifest = pd.read_csv(
    MANIFEST_PATH,
    dtype={"tile_id": str, "region_id": str},
)

locations = pd.read_csv(
    LOCATIONS_PATH,
    dtype={
        "tile_id": str,
        "scanner": str,
        "shard": str,
        "region_id": str,
    },
)

if len(manifest) != 360:
    raise RuntimeError("Unexpected manifest size.")

if len(locations) != 720:
    raise RuntimeError(
        "Expected exactly 720 source locations."
    )

expected_targets = {
    (str(row.tile_id), scanner): {
        "region_id": str(row.region_id),
        "slide_id": EXPECTED_SLIDES[scanner],
    }
    for row in manifest.itertuples(index=False)
    for scanner in EXPECTED_SLIDES
}

found_targets = set()

for row in locations.itertuples(index=False):

    key = (
        str(row.tile_id),
        str(row.scanner)
    )

    if key not in expected_targets:
        raise RuntimeError(
            f"Unexpected location target: {key}"
        )

    if key in found_targets:
        raise RuntimeError(
            f"Duplicate exact location: {key}"
        )

    found_targets.add(key)

    expected = expected_targets[key]

    if (
        str(row.region_id) != expected["region_id"]
        or str(row.slide_id) != expected["slide_id"]
        or str(row.stainer) != "GIVH"
        or not bool(row.exact_metadata_verified)
    ):
        raise RuntimeError(
            f"Invalid provenance for {key}"
        )

if found_targets != set(expected_targets):
    raise RuntimeError(
        "Source location coverage mismatch."
    )

print("Frozen manifest: VERIFIED")
print("Exact locations: 720/720 VERIFIED")


# ============================================================
# STEP 2: ORGANIZE ROW GROUPS
# ============================================================

print("\n[2] Preparing image row groups...", flush=True)

grouped = defaultdict(list)

for row in locations.itertuples(index=False):
    key = (
        str(row.shard),
        int(row.row_group)
    )

    grouped[key].append(row)

all_groups = sorted(grouped)

print("Total required row groups:", len(all_groups))
print(
    "Mode:",
    "FULL" if args.all else "ONE-GROUP PILOT"
)

# Resume by skipping groups whose images already exist
# and have valid PNG encoding and dimensions.
def group_is_complete(group_key):
    for row in grouped[group_key]:

        destination = (
            IMAGE_ROOT
            / str(row.scanner)
            / f"{row.tile_id}.png"
        )

        if not destination.is_file():
            return False

        validate_png(destination.read_bytes())

    return True


pending_groups = [
    group
    for group in all_groups
    if not group_is_complete(group)
]

if MAX_GROUPS is not None:
    pending_groups = pending_groups[:MAX_GROUPS]

print("Groups selected for this run:", len(pending_groups))


# ============================================================
# STEP 3: DOWNLOAD SELECTED GROUPS
# ============================================================

print("\n[3] Retrieving original PNG chunks...", flush=True)

audit_records = []
total_requests = 0
total_bytes = 0

with requests.Session() as session:

    for shard, rg_index in pending_groups:

        selected_rows = grouped[
            (shard, rg_index)
        ]

        print(
            f"\nShard: {shard}",
            flush=True
        )

        print(
            f"Row group: {rg_index}",
            flush=True
        )

        print(
            f"Required images: {len(selected_rows)}",
            flush=True
        )

        remote = HTTPRangeFile(
            BASE_URL + shard,
            session
        )

        parquet = pq.ParquetFile(
            remote,
            pre_buffer=False,
        )

        # Read PNG column once for the entire group.
        image_table = parquet.read_row_group(
            rg_index,
            columns=["png"],
            use_threads=False,
        )

        image_column = image_table["png"]

        for row in selected_rows:

            row_index = int(
                row.row_index_in_group
            )

            if not 0 <= row_index < image_table.num_rows:
                raise RuntimeError(
                    "Invalid exact image row index."
                )

            asset = image_column[row_index].as_py()

            if not isinstance(asset, dict):
                raise RuntimeError(
                    "Invalid Parquet PNG structure."
                )

            raw = asset.get("bytes")

            if not isinstance(raw, bytes):
                raise RuntimeError(
                    "Missing original PNG bytes."
                )

            mode = validate_png(raw)

            destination = (
                IMAGE_ROOT
                / str(row.scanner)
                / f"{row.tile_id}.png"
            )

            digest, status = save_original_png(
                destination,
                raw
            )

            audit_records.append({
                "region_id": str(row.region_id),
                "tile_id": str(row.tile_id),
                "scanner": str(row.scanner),
                "shard": shard,
                "row_group": rg_index,
                "row_index_in_group": row_index,
                "image_format": "PNG",
                "image_width": 224,
                "image_height": 224,
                "image_mode": mode,
                "image_bytes": len(raw),
                "sha256": digest,
                "status": status,
                "relative_path": str(
                    destination.relative_to(ROOT)
                ),
            })

        total_requests += remote.requests_made
        total_bytes += remote.bytes_received

        print(
            "Images validated:",
            len(selected_rows),
            flush=True
        )

        print(
            "Network transferred:",
            f"{remote.bytes_received / 1_000_000:.3f} MB",
            flush=True
        )

        # Save audit incrementally so a subsequent
        # network failure does not lose prior work.
        if IMAGE_AUDIT_PATH.exists():
            old_audit = pd.read_csv(IMAGE_AUDIT_PATH)
        else:
            old_audit = pd.DataFrame()

        merged = pd.concat(
            [
                old_audit,
                pd.DataFrame(audit_records),
            ],
            ignore_index=True
        )

        merged = merged.drop_duplicates(
            subset=["tile_id", "scanner"],
            keep="last"
        )

        merged.to_csv(
            IMAGE_AUDIT_PATH,
            index=False
        )

        audit_records.clear()

        print(
            "Audit checkpoint: SAVED",
            flush=True
        )


# ============================================================
# STEP 4: VALIDATE OVERALL PROGRESS
# ============================================================

print("\n[4] Auditing collection progress...", flush=True)

verified_images = 0
missing_images = 0
image_hashes = {}

for row in locations.itertuples(index=False):

    destination = (
        IMAGE_ROOT
        / str(row.scanner)
        / f"{row.tile_id}.png"
    )

    if not destination.is_file():
        missing_images += 1
        continue

    raw = destination.read_bytes()

    validate_png(raw)

    image_hashes[
        (str(row.tile_id), str(row.scanner))
    ] = sha256(raw)

    verified_images += 1

if verified_images + missing_images != 720:
    raise RuntimeError(
        "Unexpected collection accounting."
    )

print("Verified local images:", verified_images)
print("Remaining images:", missing_images)


# ============================================================
# STEP 5: SAVE PROGRESS REPORT
# ============================================================

print("\n[5] Saving retrieval report...", flush=True)

REPORTS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

report = {
    "phase": "3B",
    "step": "3B.17",
    "updated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "manifest_sha256": manifest_hash,
    "target_records": 720,
    "verified_local_images": verified_images,
    "remaining_images": missing_images,
    "completed": missing_images == 0,
    "processed_groups_this_run":
        len(pending_groups),
    "http_requests_this_run":
        total_requests,
    "bytes_transferred_this_run":
        total_bytes,
    "original_png_bytes_preserved": True,
    "frozen_manifest_modified": False,
    "dataset_viewer_jpegs_used": False,
    "biological_independence_verified": False,
}

REPORT_PATH.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3B.17 - ORIGINAL IMAGE RETRIEVAL")
print("=" * 70)

print("Verified original PNGs:", verified_images)
print("Remaining original PNGs:", missing_images)

print(
    "Transferred this run:",
    f"{total_bytes / 1_000_000:.3f} MB"
)

print(
    "HTTP requests this run:",
    total_requests
)

print("\nImage directory:")
print(IMAGE_ROOT)

print("\nAudit CSV:")
print(IMAGE_AUDIT_PATH)

print("\nProgress report:")
print(REPORT_PATH)

print("\nFrozen manifest: UNCHANGED")
print("JPEG previews used: NO")

if missing_images == 0:
    print(
        "\nRESULT: All 720 original PNG images "
        "have been collected and validated."
    )
else:
    print(
        "\nRESULT: Partial collection validated. "
        "The downloader can be resumed."
    )
