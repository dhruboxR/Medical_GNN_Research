
"""
Phase 3A - Step 3.4
Discover Original PLISM Annotation Resources

Queries the public Figshare API to:
1. List original PLISM collection articles
2. Inspect their attached filenames and file sizes
3. Identify candidate metadata/annotation resources
4. Save a provenance inventory

Does NOT download dataset files, images, or archives.
Uses Python standard library + pandas.
"""

from pathlib import Path
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
import json
import time

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path.cwd()

OUTPUT_ROOT = ROOT / "experimental_research" / "outputs"
TABLE_DIR = OUTPUT_ROOT / "tables"
REPORT_DIR = OUTPUT_ROOT / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

API_BASE = "https://api.figshare.com/v2"
COLLECTION_ID = 6773925

MAX_PAGES = 5
PAGE_SIZE = 100
REQUEST_TIMEOUT = 30

CANDIDATE_EXTENSIONS = (
    ".csv", ".tsv", ".json", ".xlsx",
    ".xls", ".geojson", ".xml"
)

CANDIDATE_KEYWORDS = (
    "annotation", "metadata", "label", "tissue",
    "core", "specimen", "coordinate", "sample",
    "mapping", "information"
)


# ============================================================
# API HELPER
# ============================================================

def get_json(url, params=None):
    if params:
        url += "?" + urlencode(params)

    request = Request(
        url,
        headers={
            "User-Agent": "PLISM-research-metadata-audit/1.0",
            "Accept": "application/json"
        }
    )

    with urlopen(
        request,
        timeout=REQUEST_TIMEOUT
    ) as response:
        return json.load(response)


# ============================================================
# STEP 1: COLLECTION METADATA
# ============================================================

print("=" * 70, flush=True)
print("PHASE 3A - PLISM ANNOTATION RESOURCE DISCOVERY", flush=True)
print("=" * 70, flush=True)

print("\n[1] Reading Figshare collection metadata...", flush=True)

collection_url = (
    f"{API_BASE}/collections/{COLLECTION_ID}"
)

try:
    collection = get_json(collection_url)

except (HTTPError, URLError, TimeoutError) as exc:
    raise SystemExit(
        f"Figshare API unavailable: {exc}\n"
        "No dataset files were downloaded."
    )

print("Collection:", collection.get("title", "Unknown"))
print("Collection ID:", COLLECTION_ID)


# ============================================================
# STEP 2: LIST COLLECTION ARTICLES
# ============================================================

print("\n[2] Listing collection articles...", flush=True)

articles = []

for page in range(1, MAX_PAGES + 1):

    url = (
        f"{API_BASE}/collections/"
        f"{COLLECTION_ID}/articles"
    )

    try:
        batch = get_json(
            url,
            {
                "page": page,
                "page_size": PAGE_SIZE
            }
        )

    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"Page {page} error: {exc}", flush=True)
        break

    if not isinstance(batch, list) or not batch:
        break

    articles.extend(batch)

    print(
        f"Page {page}: {len(batch)} articles",
        flush=True
    )

    if len(batch) < PAGE_SIZE:
        break

    time.sleep(1)

print(f"Discovered articles: {len(articles)}")


# ============================================================
# STEP 3: INSPECT ARTICLE ATTACHMENTS
# ============================================================

print("\n[3] Inspecting attached file metadata...", flush=True)

file_records = []
article_records = []
errors = []

for index, article in enumerate(articles, start=1):

    article_id = article.get("id")

    if article_id is None:
        continue

    print(
        f"Inspecting article {index}/{len(articles)} "
        f"(ID {article_id})",
        flush=True
    )

    try:
        detail = get_json(
            f"{API_BASE}/articles/{article_id}"
        )

    except (HTTPError, URLError, TimeoutError) as exc:
        errors.append({
            "article_id": article_id,
            "error": str(exc)
        })
        continue

    title = detail.get("title", "")

    article_records.append({
        "article_id": article_id,
        "title": title,
        "doi": detail.get("doi", ""),
        "url": detail.get("url_public_html", ""),
        "files_count": len(detail.get("files") or [])
    })

    for item in detail.get("files") or []:

        filename = item.get("name") or ""
        lower = filename.lower()

        is_candidate = (
            lower.endswith(CANDIDATE_EXTENSIONS)
            or any(
                keyword in lower
                for keyword in CANDIDATE_KEYWORDS
            )
        )

        size_bytes = int(item.get("size") or 0)

        file_records.append({
            "article_id": article_id,
            "article_title": title,
            "file_id": item.get("id"),
            "filename": filename,
            "size_bytes": size_bytes,
            "size_mb": round(
                size_bytes / (1024 ** 2), 3
            ),
            "candidate_annotation": is_candidate,
            "article_url": detail.get(
                "url_public_html", ""
            )
        })

    time.sleep(1)


# ============================================================
# STEP 4: SAVE INVENTORIES
# ============================================================

print("\n[4] Saving inventories...", flush=True)

article_df = pd.DataFrame(
    article_records,
    columns=[
        "article_id", "title", "doi",
        "url", "files_count"
    ]
)

file_df = pd.DataFrame(
    file_records,
    columns=[
        "article_id",
        "article_title",
        "file_id",
        "filename",
        "size_bytes",
        "size_mb",
        "candidate_annotation",
        "article_url"
    ]
)

article_path = (
    TABLE_DIR / "plism_figshare_articles.csv"
)

file_path = (
    TABLE_DIR / "plism_figshare_file_inventory.csv"
)

candidate_path = (
    TABLE_DIR / "plism_annotation_candidates.csv"
)

article_df.to_csv(article_path, index=False)
file_df.to_csv(file_path, index=False)

candidate_df = file_df[
    file_df["candidate_annotation"]
].copy()

candidate_df.to_csv(candidate_path, index=False)


# ============================================================
# STEP 5: REPORT
# ============================================================

print("\n[5] Generating discovery report...", flush=True)

report = {
    "phase": "3A",
    "step": "3.4",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "collection_id": COLLECTION_ID,
    "collection_title": collection.get("title"),
    "articles_found": len(articles),
    "articles_inspected": len(article_records),
    "attached_files_found": len(file_records),
    "candidate_annotation_files": len(candidate_df),
    "article_errors": errors,
    "complete_collection_inventory": (
        len(articles) < MAX_PAGES * PAGE_SIZE
        and len(errors) == 0
    ),
    "dataset_files_downloaded": 0,
    "tissue_core_mapping_verified": False
}

report_path = (
    REPORT_DIR / "04_plism_annotation_resource_discovery.json"
)

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("ANNOTATION RESOURCE DISCOVERY COMPLETED")
print("=" * 70)

print(f"Collection articles: {len(articles)}")
print(f"Articles inspected: {len(article_records)}")
print(f"Attached files: {len(file_records)}")
print(f"Annotation candidates: {len(candidate_df)}")
print(f"Article inspection errors: {len(errors)}")
print("Dataset files downloaded: 0")

print("\nCandidate annotation resources:")

if candidate_df.empty:
    print("No candidate annotation files identified.")
else:
    print(
        candidate_df[
            ["article_id", "filename", "size_mb"]
        ].to_string(index=False)
    )

print("\nSaved files:")
print(article_path)
print(file_path)
print(candidate_path)
print(report_path)

print("\nSTEP 3.4 COMPLETED")
