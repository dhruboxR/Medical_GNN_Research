
from pathlib import Path
from io import BytesIO
from time import perf_counter
from datasets import load_dataset, Image as HFImage
from PIL import Image as PILImage
import pandas as pd

# ==================================================
# CONFIGURATION
# ==================================================

REPO = "owkin/plism-dataset-tiles"

MANIFEST_PATH = Path(
    "dataset_exploration/outputs/tables/"
    "plism_spatial_region_manifest.csv"
)

IMAGE_ROOT = Path(
    "datasets/PLISM/GIVH_AT2_GT450"
)

TABLE_DIR = Path(
    "dataset_exploration/outputs/tables"
)

REPORT_DIR = Path(
    "dataset_exploration/outputs/reports"
)

SCANNERS = ["AT2", "GT450"]

EXPECTED_STAINER = "GIVH"
EXPECTED_SIZE = (224, 224)

TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

start_time = perf_counter()

print("=" * 70)
print("PLISM SPATIAL REGION IMAGE DOWNLOADER")
print("=" * 70)

# ==================================================
# STEP 1: LOAD VERIFIED SPATIAL MANIFEST
# ==================================================

print("\n[1] Loading verified spatial manifest...")

if not MANIFEST_PATH.exists():
    raise SystemExit(
        f"Manifest not found: {MANIFEST_PATH}"
    )

manifest = pd.read_csv(MANIFEST_PATH)

required_columns = {
    "tile_id",
    "stainer",
    "scanner_a",
    "scanner_b",
    "slide_id_a",
    "slide_id_b",
    "source_shard_a",
    "source_shard_b",
}

missing_columns = required_columns - set(manifest.columns)

if missing_columns:
    raise SystemExit(
        f"Missing manifest columns: {missing_columns}"
    )

if manifest.empty:
    raise SystemExit("Manifest is empty.")

if manifest["tile_id"].duplicated().any():
    raise SystemExit("Duplicate tile IDs in manifest.")

if not manifest["stainer"].eq(EXPECTED_STAINER).all():
    raise SystemExit("Unexpected stainer in manifest.")

if not manifest["scanner_a"].eq("AT2").all():
    raise SystemExit("Unexpected scanner A.")

if not manifest["scanner_b"].eq("GT450").all():
    raise SystemExit("Unexpected scanner B.")

target_ids = set(manifest["tile_id"])

print(f"Verified matched locations: {len(target_ids)}")

# ==================================================
# STEP 2: IMAGE VALIDATION HELPER
# ==================================================

def valid_image(path):
    if not path.exists():
        return False

    try:
        with PILImage.open(path) as image:
            image.verify()

        with PILImage.open(path) as image:
            return image.size == EXPECTED_SIZE

    except Exception:
        return False


# ==================================================
# STEP 3: REUSE EXISTING IMAGES
# ==================================================

print("\n[2] Checking previously downloaded images...")

completed = {}
downloaded_counts = {}
reused_counts = {}

for scanner in SCANNERS:

    scanner_dir = IMAGE_ROOT / scanner
    scanner_dir.mkdir(parents=True, exist_ok=True)

    existing = set()

    for tile_id in target_ids:
        path = scanner_dir / f"{tile_id}.png"

        if valid_image(path):
            existing.add(tile_id)

    completed[scanner] = existing
    reused_counts[scanner] = len(existing)
    downloaded_counts[scanner] = 0

    print(
        f"{scanner}: {len(existing)}/{len(target_ids)} "
        "images already available"
    )

# ==================================================
# STEP 4: DOWNLOAD MISSING IMAGES
# ==================================================

print("\n[3] Downloading missing images...")

for scanner, shard_col, slide_col in [
    ("AT2", "source_shard_a", "slide_id_a"),
    ("GT450", "source_shard_b", "slide_id_b"),
]:

    scanner_dir = IMAGE_ROOT / scanner

    pending = target_ids - completed[scanner]

    print(
        f"\nScanner {scanner}: "
        f"{len(pending)} images needed",
        flush=True
    )

    if not pending:
        continue

    # Each scanner may use one or more shards.
    shard_groups = manifest.groupby(shard_col)

    for shard_index, shard_rows in shard_groups:

        shard_index = int(shard_index)

        shard_targets = (
            set(shard_rows["tile_id"]) & pending
        )

        if not shard_targets:
            continue

        expected_slides = set(
            shard_rows[slide_col].astype(str)
        )

        filename = (
            f"data/train-{shard_index:05d}"
            "-of-00293.parquet"
        )

        url = f"hf://datasets/{REPO}/{filename}"

        print(
            f"  Opening shard {shard_index}...",
            flush=True
        )

        ds = load_dataset(
            "parquet",
            data_files={"train": url},
            split="train",
            streaming=True
        )

        ds = ds.cast_column(
            "png",
            HFImage(decode=False)
        )

        ds = ds.select_columns([
            "stainer",
            "scanner",
            "slide_id",
            "tile_id",
            "png"
        ])

        found_in_shard = set()

        for sample in ds:

            tile_id = sample["tile_id"]

            if tile_id not in shard_targets:
                continue

            if sample["scanner"] != scanner:
                continue

            if sample["stainer"] != EXPECTED_STAINER:
                continue

            if sample["slide_id"] not in expected_slides:
                continue

            image_bytes = sample["png"].get("bytes")

            if image_bytes is None:
                print(
                    f"  No inline bytes for {tile_id}"
                )
                continue

            try:
                with PILImage.open(
                    BytesIO(image_bytes)
                ) as image:
                    image.verify()

                with PILImage.open(
                    BytesIO(image_bytes)
                ) as image:
                    if image.size != EXPECTED_SIZE:
                        print(
                            f"  Unexpected dimensions: "
                            f"{tile_id}"
                        )
                        continue

                output_path = (
                    scanner_dir / f"{tile_id}.png"
                )

                output_path.write_bytes(image_bytes)

                completed[scanner].add(tile_id)
                found_in_shard.add(tile_id)

                downloaded_counts[scanner] += 1

                count = downloaded_counts[scanner]

                if count % 10 == 0:
                    print(
                        f"  Downloaded {count} "
                        f"new {scanner} images",
                        flush=True
                    )

            except Exception as exc:
                print(
                    f"  Failed {tile_id}: {exc}",
                    flush=True
                )

            if found_in_shard == shard_targets:
                break

        missing_in_shard = (
            shard_targets - found_in_shard
        )

        if missing_in_shard:
            print(
                f"  Missing from shard "
                f"{shard_index}: "
                f"{len(missing_in_shard)}"
            )

    print(
        f"{scanner}: "
        f"{len(completed[scanner])}/"
        f"{len(target_ids)} available",
        flush=True
    )

# ==================================================
# STEP 5: VALIDATE COMPLETE PAIRS
# ==================================================

print("\n[4] Validating final image pairs...")

records = []

for _, row in manifest.iterrows():

    tile_id = row["tile_id"]

    path_a = IMAGE_ROOT / "AT2" / f"{tile_id}.png"
    path_b = IMAGE_ROOT / "GT450" / f"{tile_id}.png"

    valid_a = valid_image(path_a)
    valid_b = valid_image(path_b)

    records.append({
        "tile_id": tile_id,
        "grid_row": row.get("grid_row"),
        "grid_col": row.get("grid_col"),
        "stainer": EXPECTED_STAINER,
        "at2_path": str(path_a),
        "gt450_path": str(path_b),
        "at2_valid": valid_a,
        "gt450_valid": valid_b,
        "complete_pair": valid_a and valid_b
    })

result = pd.DataFrame(records)

complete_pairs = int(
    result["complete_pair"].sum()
)

total_valid_images = int(
    result["at2_valid"].sum()
    + result["gt450_valid"].sum()
)

print(f"Complete matched pairs: {complete_pairs}")
print(f"Validated image files: {total_valid_images}")

# ==================================================
# STEP 6: SAVE RESULTS
# ==================================================

print("\n[5] Saving final manifest and report...")

final_manifest = (
    TABLE_DIR / "plism_spatial_region_download_manifest.csv"
)

report_path = (
    REPORT_DIR / "plism_spatial_region_download_report.txt"
)

result.to_csv(final_manifest, index=False)

elapsed = perf_counter() - start_time

with open(report_path, "w", encoding="utf-8") as f:

    f.write("PLISM Spatial Region Download Report\n")
    f.write("=" * 50 + "\n\n")

    f.write(
        f"Target locations: {len(target_ids)}\n"
    )

    f.write(
        f"Complete pairs: {complete_pairs}\n"
    )

    f.write(
        f"Valid image files: {total_valid_images}\n"
    )

    for scanner in SCANNERS:
        f.write(
            f"{scanner} reused: "
            f"{reused_counts[scanner]}\n"
        )
        f.write(
            f"{scanner} newly downloaded: "
            f"{downloaded_counts[scanner]}\n"
        )

    f.write(
        f"Execution time: {elapsed:.2f} seconds\n"
    )

print("\n" + "=" * 70)
print("PLISM SPATIAL REGION DOWNLOAD FINISHED")
print("=" * 70)

for scanner in SCANNERS:
    print(
        f"{scanner}: reused "
        f"{reused_counts[scanner]}, "
        f"downloaded "
        f"{downloaded_counts[scanner]}"
    )

print(f"\nComplete pairs: {complete_pairs}/{len(target_ids)}")
print(f"Valid images: {total_valid_images}")
print(f"Manifest: {final_manifest}")
print(f"Report: {report_path}")
print(f"Execution time: {elapsed:.2f}s")

if complete_pairs != len(target_ids):
    print("\nWARNING: Some matched pairs are incomplete.")
    raise SystemExit(1)

print("\nAll requested matched pairs are available.")
