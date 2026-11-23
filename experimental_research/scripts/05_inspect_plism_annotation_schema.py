
"""
Phase 3A - Step 3.5
Inspect original PLISM annotation CSV schemas.

- Uses existing Figshare resource inventory
- Finds official CSV download URLs through the API
- Requests only the first 64 KB of each CSV
- Refuses full-file responses
- Saves field and sample metadata, not image files
"""

from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import csv
import io
import json

import pandas as pd


ROOT = Path.cwd()

OUTPUT = ROOT / "experimental_research" / "outputs"
TABLE_DIR = OUTPUT / "tables"
REPORT_DIR = OUTPUT / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

INVENTORY = (
    TABLE_DIR / "plism_annotation_candidates.csv"
)

TARGET_FILES = {
    "PLISM_sm_en.csv": 23590791,
    "PLISM_wsi_en.csv": 23614422,
}

MAX_BYTES = 65536
TIMEOUT = 45

print("=" * 70, flush=True)
print("PHASE 3A - PLISM ANNOTATION SCHEMA INSPECTION", flush=True)
print("=" * 70, flush=True)


def fetch_json(url):
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "PLISM-Metadata-Audit/1.0",
        },
    )

    with urlopen(request, timeout=TIMEOUT) as response:
        return json.load(response)


def fetch_partial_csv(url):
    request = Request(
        url,
        headers={
            "Range": f"bytes=0-{MAX_BYTES - 1}",
            "Accept-Encoding": "identity",
            "User-Agent": "PLISM-Metadata-Audit/1.0",
        },
    )

    with urlopen(request, timeout=TIMEOUT) as response:
        status = response.status

        if status != 206:
            # Do not read the body of a full-file response.
            return None, f"SKIPPED_HTTP_{status}"

        raw = response.read(MAX_BYTES + 1)

        if len(raw) > MAX_BYTES:
            return None, "SKIPPED_RESPONSE_TOO_LARGE"

        content_range = response.headers.get(
            "Content-Range", ""
        )

        if not content_range.lower().startswith("bytes 0-"):
            return None, "SKIPPED_INVALID_CONTENT_RANGE"

    return raw, "PARTIAL_READ_OK"


def parse_csv_sample(raw):
    text = raw.decode("utf-8-sig", errors="replace")

    # The last line may be incomplete because of
    # the 64 KB byte limit.
    if not text.endswith(("\n", "\r")):
        text = text.rsplit("\n", 1)[0]

    if not text.strip():
        return [], []

    reader = csv.reader(io.StringIO(text))
    records = list(reader)

    if not records:
        return [], []

    headers = records[0]

    sample_rows = [
        record
        for record in records[1:]
        if len(record) == len(headers)
    ][:3]

    return headers, sample_rows


if not INVENTORY.exists():
    raise SystemExit(
        f"Missing inventory: {INVENTORY}"
    )

inventory = pd.read_csv(INVENTORY)

print("\n[1] Loading candidate inventory...")
print(f"Candidate records: {len(inventory)}")

schema_records = []
samples = []
inspection_results = []

for filename, article_id in TARGET_FILES.items():

    print(f"\n[2] Inspecting {filename}...", flush=True)

    try:
        metadata = fetch_json(
            f"https://api.figshare.com/v2/articles/{article_id}"
        )

        matching_files = [
            f for f in metadata.get("files", [])
            if f.get("name") == filename
        ]

        if not matching_files:
            inspection_results.append({
                "filename": filename,
                "status": "FILE_NOT_FOUND",
                "columns_found": 0,
            })
            print("File not found in article metadata.")
            continue

        file_metadata = matching_files[0]
        download_url = file_metadata.get("download_url")

        if not download_url:
            inspection_results.append({
                "filename": filename,
                "status": "NO_DOWNLOAD_URL",
                "columns_found": 0,
            })
            print("No download URL available.")
            continue

        raw, status = fetch_partial_csv(download_url)

        if raw is None:
            inspection_results.append({
                "filename": filename,
                "status": status,
                "columns_found": 0,
            })
            print(f"Schema inspection skipped: {status}")
            continue

        headers, sample_rows = parse_csv_sample(raw)

        print(f"Status: {status}")
        print(f"Columns found: {len(headers)}")

        for index, column in enumerate(headers):
            print(f"  {index + 1}. {column}")

            schema_records.append({
                "filename": filename,
                "article_id": article_id,
                "column_position": index + 1,
                "column_name": column,
            })

        for row_index, values in enumerate(sample_rows):
            sample = {
                "filename": filename,
                "sample_row": row_index + 1,
            }

            sample.update(
                dict(zip(headers, values))
            )

            samples.append(sample)

        inspection_results.append({
            "filename": filename,
            "status": status,
            "columns_found": len(headers),
        })

    except (HTTPError, URLError, TimeoutError) as error:
        print(f"Request failed: {error}")

        inspection_results.append({
            "filename": filename,
            "status": "REQUEST_FAILED",
            "columns_found": 0,
        })


print("\n[3] Saving schema reports...")

schema_df = pd.DataFrame(schema_records)
samples_df = pd.DataFrame(samples)
status_df = pd.DataFrame(inspection_results)

schema_path = TABLE_DIR / "plism_annotation_schema.csv"
sample_path = TABLE_DIR / "plism_annotation_sample_rows.csv"
status_path = REPORT_DIR / "05_annotation_schema_status.json"

schema_df.to_csv(schema_path, index=False)
samples_df.to_csv(sample_path, index=False)

status_path.write_text(
    json.dumps(
        inspection_results,
        indent=2
    ),
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("ANNOTATION SCHEMA INSPECTION FINISHED")
print("=" * 70)

print(status_df.to_string(index=False))
print("\nSaved:")
print(schema_path)
print(sample_path)
print(status_path)

print(
    "\nNo complete annotation CSV files "
    "or image files were saved."
)
