
"""
PHASE 3B - STEP 3B.12
Original Parquet Access Feasibility

Tests:
- Source shard metadata
- HTTP byte-range support
- Parquet footer signature

Safety:
- No full shard downloads
- No image downloads
- No changes to existing datasets or manifests
- Strict request timeouts
"""

from pathlib import Path
from datetime import datetime, timezone
import json

import requests


ROOT = Path.home() / "MEDICAL_GNN_RESEARCH"
REPORT_DIR = (
    ROOT / "multi_region_research/outputs/reports"
)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "owkin/plism-dataset-tiles"

SHARDS = [
    "data/train-00000-of-00293.parquet",
    "data/train-00001-of-00293.parquet",
    "data/train-00002-of-00293.parquet",
]

BASE_URL = (
    "https://huggingface.co/datasets/"
    f"{DATASET}/resolve/main/"
)

TIMEOUT = (10, 20)

results = []

print("=" * 70, flush=True)
print("PHASE 3B - ORIGINAL PARQUET ACCESS TEST", flush=True)
print("=" * 70, flush=True)

with requests.Session() as session:

    for shard in SHARDS:

        print(f"\nTesting: {shard}", flush=True)

        url = BASE_URL + shard

        result = {
            "shard": shard,
            "head_status": None,
            "reported_size": None,
            "range_status": None,
            "content_range": None,
            "received_bytes": 0,
            "footer_magic_valid": False,
            "range_supported": False,
            "error": None,
        }

        try:
            # HEAD requests do not retrieve file content.
            response = session.head(
                url,
                allow_redirects=True,
                timeout=TIMEOUT,
            )

            result["head_status"] = response.status_code

            length = response.headers.get(
                "Content-Length"
            )

            if length and length.isdigit():
                result["reported_size"] = int(length)

            print("  HEAD status:", response.status_code)
            print(
                "  Reported size:",
                result["reported_size"]
            )

            # Request only the final eight bytes.
            # A valid Parquet footer ends with PAR1.
            with session.get(
                url,
                headers={
                    "Range": "bytes=-8",
                    "Accept-Encoding": "identity",
                },
                stream=True,
                timeout=TIMEOUT,
                allow_redirects=True,
            ) as response:

                result["range_status"] = (
                    response.status_code
                )

                result["content_range"] = (
                    response.headers.get(
                        "Content-Range"
                    )
                )

                print(
                    "  Range status:",
                    response.status_code
                )

                print(
                    "  Content-Range:",
                    result["content_range"]
                )

                if response.status_code == 206:

                    # Never read more than a tiny response.
                    data = response.raw.read(
                        16, decode_content=False
                    )

                    result["received_bytes"] = len(data)

                    if len(data) == 8:
                        result["footer_magic_valid"] = (
                            data[-4:] == b"PAR1"
                        )

                        result["range_supported"] = (
                            result["footer_magic_valid"]
                        )

                    print(
                        "  Bytes received:",
                        len(data)
                    )

                    print(
                        "  Parquet footer:",
                        "VALID" if result["footer_magic_valid"]
                        else "NOT VERIFIED"
                    )

                elif response.status_code == 200:

                    print(
                        "  Server did not honor the "
                        "8-byte Range request."
                    )

                    print(
                        "  Full response was not consumed."
                    )

                else:
                    print(
                        "  Range request was unsuccessful."
                    )

        except requests.RequestException as exc:

            result["error"] = str(exc)
            print("  Request error:", exc, flush=True)

        results.append(result)


print("\n" + "=" * 70)
print("PHASE 3B.12 - ACCESS DIAGNOSTIC")
print("=" * 70)

supported = sum(
    r["range_supported"] for r in results
)

print("Shards tested:", len(results))
print("Valid ranged Parquet responses:", supported)

if supported:
    print(
        "Decision: Partial Parquet access appears feasible."
    )
else:
    print(
        "Decision: Partial Parquet access not established."
    )

report = {
    "phase": "3B",
    "step": "3B.12",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "dataset": DATASET,
    "results": results,
    "images_downloaded": 0,
    "full_shards_downloaded": 0,
    "original_image_bytes_verified": False,
}

report_path = (
    REPORT_DIR /
    "12_original_parquet_access_feasibility.json"
)

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print("\nReport:", report_path)
print("No complete shards or images downloaded.")
