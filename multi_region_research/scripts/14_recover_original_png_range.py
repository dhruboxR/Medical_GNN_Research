
"""
PHASE 3B - STEP 3B.14

Recover one original PNG from a remote Parquet row group
using bounded HTTP byte-range requests.

Target:
    tile_16_100_127
    Scanner: AT2
    Shard: train-00000-of-00293.parquet
    Row group: 0

Validation:
    Compare recovered PNG SHA256 against the existing
    local Phase 1 PNG.

This is a single-image feasibility experiment.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import io
import json
import re

import pyarrow.parquet as pq
import requests
from PIL import Image
from io import BytesIO


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"

BASE = ROOT / "multi_region_research"

REPORT_DIR = BASE / "outputs/reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = (
    BASE / "outputs/validated/"
    "phase3b_frozen_tile_manifest.csv"
)

EXPECTED_MANIFEST_HASH = (
    "69d11e1a0dd1107f2e6534632727e2c2291ff83d6b17"
    "cf953ed5056bea9020f9"
)

TARGET_TILE = "tile_16_100_127"
TARGET_SCANNER = "AT2"
TARGET_STAINER = "GIVH"

TARGET_SLIDE = "GIVH_AT2_to_GMH_S60"

SHARD_NAME = "train-00000-of-00293.parquet"
ROW_GROUP = 0

URL = (
    "https://huggingface.co/datasets/"
    "owkin/plism-dataset-tiles/resolve/main/data/"
    + SHARD_NAME
)

ORIGINAL_PATH = (
    ROOT
    / "datasets/PLISM/GIVH_AT2_GT450/AT2"
    / f"{TARGET_TILE}.png"
)

REPORT_PATH = (
    REPORT_DIR /
    "14_original_png_range_retrieval.json"
)

# These are intentionally conservative limits.
MAX_SINGLE_REQUEST = 16 * 1024 * 1024
MAX_TOTAL_TRANSFER = 35 * 1024 * 1024

TIMEOUT = (10, 40)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


# ============================================================
# BOUNDED REMOTE FILE READER
# ============================================================

class HTTPRangeFile(io.RawIOBase):
    """
    Seekable, read-only remote file.

    Every read must use HTTP 206 Partial Content.
    Full-file responses are rejected.

    The class tracks aggregate response bytes and
    stops once the configured transfer budget is hit.
    """

    def __init__(self, url, session):
        super().__init__()

        self.url = url
        self.session = session
        self.position = 0

        self.bytes_received = 0
        self.requests_made = 0

        response = session.head(
            url,
            allow_redirects=True,
            timeout=TIMEOUT,
        )

        response.raise_for_status()

        size = response.headers.get(
            "Content-Length"
        )

        if size is None or not size.isdigit():
            raise RuntimeError(
                "Remote file size unavailable"
            )

        self.file_size = int(size)

        print(
            "Remote shard size:",
            self.file_size,
            "bytes",
            flush=True
        )

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
            next_position = offset

        elif whence == io.SEEK_CUR:
            next_position = self.position + offset

        elif whence == io.SEEK_END:
            next_position = self.file_size + offset

        else:
            raise ValueError("Invalid seek mode")

        if not 0 <= next_position <= self.file_size:
            raise ValueError(
                "Seek position outside remote file"
            )

        self.position = int(next_position)

        return self.position

    def read(self, size=-1):
        if self.position >= self.file_size:
            return b""

        if size is None or size < 0:
            raise RuntimeError(
                "Unbounded remote reads are disabled"
            )

        if size == 0:
            return b""

        size = min(
            size,
            self.file_size - self.position
        )

        if size > MAX_SINGLE_REQUEST:
            raise RuntimeError(
                f"Requested {size} bytes; "
                "exceeds single-request limit"
            )

        if (
            self.bytes_received + size
            > MAX_TOTAL_TRANSFER
        ):
            raise RuntimeError(
                "Aggregate transfer budget exceeded"
            )

        start = self.position
        end = start + size - 1

        expected_range = f"bytes {start}-{end}/"

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
                    "Range request rejected: "
                    f"HTTP {response.status_code}"
                )

            returned_range = response.headers.get(
                "Content-Range", ""
            )

            if not returned_range.startswith(
                expected_range
            ):
                raise RuntimeError(
                    "Unexpected Content-Range: "
                    f"{returned_range}"
                )

            data = bytearray()

            for chunk in response.iter_content(65536):
                data.extend(chunk)

                if len(data) > size:
                    raise RuntimeError(
                        "Received more bytes than requested"
                    )

        if len(data) != size:
            raise RuntimeError(
                "Incomplete range response: "
                f"{len(data)} != {size}"
            )

        self.position += size
        self.bytes_received += size
        self.requests_made += 1

        return bytes(data)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


# ============================================================
# STEP 1: VERIFY LOCAL REFERENCE
# ============================================================

print("=" * 70, flush=True)
print("PHASE 3B - ORIGINAL PNG RANGE RETRIEVAL", flush=True)
print("=" * 70, flush=True)

print("\n[1] Checking local reference image...", flush=True)

if not MANIFEST.is_file():
    raise RuntimeError("Frozen manifest missing")

manifest_hash = sha256(
    MANIFEST.read_bytes()
)

if manifest_hash != EXPECTED_MANIFEST_HASH:
    raise RuntimeError(
        "Frozen manifest fingerprint mismatch"
    )

if not ORIGINAL_PATH.is_file():
    raise RuntimeError(
        f"Local reference PNG missing: {ORIGINAL_PATH}"
    )

local_bytes = ORIGINAL_PATH.read_bytes()
local_hash = sha256(local_bytes)

with Image.open(BytesIO(local_bytes)) as image:
    image.verify()

print("Frozen manifest: VERIFIED")
print("Reference PNG:", ORIGINAL_PATH)
print("Reference SHA256:", local_hash)


# ============================================================
# STEP 2: OPEN REMOTE PARQUET USING HTTP RANGES
# ============================================================

print(
    "\n[2] Opening remote Parquet metadata...",
    flush=True
)

with requests.Session() as session:

    remote = HTTPRangeFile(URL, session)

    # PyArrow reads footer metadata through the
    # bounded, seekable HTTPRangeFile.
    #
    # pre_buffer=False avoids proactively fetching
    # larger ranges beyond the requested columns.
    parquet = pq.ParquetFile(
        remote,
        pre_buffer=False,
    )

    if ROW_GROUP >= parquet.metadata.num_row_groups:
        raise RuntimeError(
            "Requested row group does not exist"
        )

    print(
        "Row groups:",
        parquet.metadata.num_row_groups,
        flush=True
    )

    print(
        "Selected row group:",
        ROW_GROUP,
        flush=True
    )

    # ========================================================
    # STEP 3: READ SMALL METADATA COLUMNS FIRST
    # ========================================================

    print(
        "\n[3] Locating target tile in row group...",
        flush=True
    )

    metadata_table = parquet.read_row_group(
        ROW_GROUP,
        columns=[
            "tile_id",
            "scanner",
            "stainer",
            "slide_id",
        ],
        use_threads=False,
    )

    small_rows = metadata_table.to_pylist()

    matches = [
        i
        for i, row in enumerate(small_rows)
        if (
            row.get("tile_id") == TARGET_TILE
            and row.get("scanner") == TARGET_SCANNER
            and row.get("stainer") == TARGET_STAINER
            and row.get("slide_id") == TARGET_SLIDE
        )
    ]

    print(
        "Rows in group:",
        len(small_rows),
        flush=True
    )

    print(
        "Exact matching records:",
        len(matches),
        flush=True
    )

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one original tile "
            "in this row group. "
            "No image column was downloaded."
        )

    target_index = matches[0]

    # ========================================================
    # STEP 4: DOWNLOAD IMAGE COLUMN CHUNK
    # ========================================================

    print(
        "\n[4] Reading selected PNG column chunk...",
        flush=True
    )

    png_table = parquet.read_row_group(
        ROW_GROUP,
        columns=["png"],
        use_threads=False,
    )

    if png_table.num_rows != len(small_rows):
        raise RuntimeError(
            "Metadata/image row count mismatch"
        )

    # Selecting a nested parent column returns
    # the associated bytes/path structure.
    png_values = png_table["png"].to_pylist()

    asset = png_values[target_index]

    if not isinstance(asset, dict):
        raise RuntimeError(
            "Unexpected nested PNG value"
        )

    remote_png = asset.get("bytes")

    if not isinstance(remote_png, bytes):
        raise RuntimeError(
            "Original PNG bytes are unavailable"
        )

    if not remote_png.startswith(
        b"\x89PNG\r\n\x1a\n"
    ):
        raise RuntimeError(
            "Retrieved image is not PNG encoded"
        )

    with Image.open(BytesIO(remote_png)) as image:
        image.verify()

    with Image.open(BytesIO(remote_png)) as image:
        image.load()
        remote_size = image.size
        remote_mode = image.mode

    remote_hash = sha256(remote_png)

    # ========================================================
    # STEP 5: VERIFY SOURCE EQUIVALENCE
    # ========================================================

    print(
        "\n[5] Comparing remote and local PNG...",
        flush=True
    )

    exact_byte_match = (
        remote_png == local_bytes
    )

    print("Remote PNG bytes:", len(remote_png))
    print("Remote PNG dimensions:", remote_size)
    print("Remote image mode:", remote_mode)
    print("Remote SHA256:", remote_hash)
    print("Local SHA256:", local_hash)
    print("Exact byte match:", exact_byte_match)

    print(
        "HTTP range requests:",
        remote.requests_made
    )

    print(
        "Total transferred:",
        remote.bytes_received,
        "bytes"
    )

    # We intentionally do not overwrite or duplicate
    # the existing local reference image.

    report = {
        "phase": "3B",
        "step": "3B.14",
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "shard": SHARD_NAME,
        "row_group": ROW_GROUP,
        "tile_id": TARGET_TILE,
        "scanner": TARGET_SCANNER,
        "stainer": TARGET_STAINER,
        "slide_id": TARGET_SLIDE,
        "remote_png_size_bytes": len(remote_png),
        "remote_png_dimensions": list(remote_size),
        "remote_png_mode": remote_mode,
        "remote_png_sha256": remote_hash,
        "local_reference_sha256": local_hash,
        "exact_byte_match": exact_byte_match,
        "http_range_requests": remote.requests_made,
        "total_http_bytes_received":
            remote.bytes_received,
        "full_shard_downloaded": False,
        "frozen_manifest_modified": False,
        "dataset_viewer_jpeg_used": False,
        "limitations": [
            "Only one tile and one row group were tested",
            "Other selected tiles may be in other shards",
            "Row-group reads transfer more than one PNG",
            "Dataset-wide original image retrieval "
            "is not yet established",
        ],
    }


# ============================================================
# STEP 6: SAVE REPORT
# ============================================================

REPORT_PATH.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("PHASE 3B.14 - ORIGINAL PNG TEST COMPLETED")
print("=" * 70)

print("Recovered image encoding: PNG")
print("Exact source-byte match:", exact_byte_match)
print("Complete shard downloaded: NO")
print("Frozen manifest modified: NO")
print("Report:", REPORT_PATH)

if exact_byte_match:
    print(
        "\nRESULT: Original PNG retrieval verified "
        "for this pilot tile."
    )
else:
    print(
        "\nRESULT: Retrieved PNG differs from the "
        "local reference. Investigate provenance "
        "before bulk retrieval."
    )
