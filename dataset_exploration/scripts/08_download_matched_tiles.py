
from datasets import load_dataset, Image
from pathlib import Path
from io import BytesIO
import pandas as pd
import time

# --------------------------------------------------
# Configuration
# --------------------------------------------------
REPO = "owkin/plism-dataset-tiles"

MATCHED_CSV = Path(
    "dataset_exploration/outputs/tables/"
    "plism_matched_tile_candidates.csv"
)

OUTPUT_DIR = Path("datasets/PLISM/GIVH_AT2_GT450")
TABLE_DIR = Path("dataset_exploration/outputs/tables")

SCANNERS = {
    "AT2": 0,
    "GT450": 3,
}

MAX_PAIRS = 20

print("=" * 65)
print("PLISM MATCHED IMAGE DOWNLOADER")
print("=" * 65)

start_time = time.perf_counter()

# --------------------------------------------------
# Step 1: Read verified matched identifiers
# --------------------------------------------------
matched = pd.read_csv(MATCHED_CSV)

target_ids = matched["tile_id"].drop_duplicates().head(
    MAX_PAIRS
).tolist()

target_set = set(target_ids)

print(f"\n[1] Requested matching locations: {len(target_ids)}")

# --------------------------------------------------
# Step 2: Download matched tiles
# --------------------------------------------------
saved_records = []

for scanner, shard in SCANNERS.items():

    print(f"\n[2] Processing scanner: {scanner}", flush=True)

    scanner_dir = OUTPUT_DIR / scanner
    scanner_dir.mkdir(parents=True, exist_ok=True)

    filename = f"data/train-{shard:05d}-of-00293.parquet"
    url = f"hf://datasets/{REPO}/{filename}"

    ds = load_dataset(
        "parquet",
        data_files={"train": url},
        split="train",
        streaming=True
    )

    # Keep image bytes without automatic PIL decoding.
    ds = ds.cast_column("png", Image(decode=False))

    ds = ds.select_columns([
        "stainer", "scanner",
        "slide_id", "tile_id", "png"
    ])

    found = set()

    for sample in ds:

        if sample["scanner"] != scanner:
            continue

        if sample["stainer"] != "GIVH":
            continue

        tile_id = sample["tile_id"]

        if tile_id not in target_set:
            continue

        if tile_id in found:
            continue

        image_data = sample["png"]
        image_bytes = image_data.get("bytes")

        if image_bytes is None:
            raise RuntimeError(
                f"No image bytes available: {scanner}/{tile_id}"
            )

        # Validate the PNG before saving.
        from PIL import Image as PILImage

        with PILImage.open(BytesIO(image_bytes)) as img:
            img.verify()

        with PILImage.open(BytesIO(image_bytes)) as img:
            width, height = img.size

        output_path = scanner_dir / f"{tile_id}.png"
        output_path.write_bytes(image_bytes)

        found.add(tile_id)

        saved_records.append({
            "scanner": scanner,
            "stainer": sample["stainer"],
            "slide_id": sample["slide_id"],
            "tile_id": tile_id,
            "width": width,
            "height": height,
            "file_path": str(output_path)
        })

        print(
            f"Saved {scanner}/{tile_id}.png "
            f"({width}x{height})",
            flush=True
        )

        if len(found) == len(target_set):
            break

    missing = target_set - found

    print(f"\n{scanner}: {len(found)}/{len(target_set)} saved")

    if missing:
        print(
            f"Missing tiles: {sorted(missing)[:10]}"
        )

# --------------------------------------------------
# Step 3: Validate paired downloads
# --------------------------------------------------
print("\n[3] Checking downloaded pairs...")

df = pd.DataFrame(saved_records)

if df.empty:
    raise SystemExit("No images were saved.")

pair_counts = df.groupby("tile_id")["scanner"].nunique()

complete_ids = pair_counts[pair_counts == 2].index.tolist()

print(f"Complete matched pairs: {len(complete_ids)}")
print(f"Total image files: {len(df)}")

manifest = TABLE_DIR / "plism_download_manifest.csv"
df.to_csv(manifest, index=False)

print(f"\nManifest: {manifest}")
print(f"Images: {OUTPUT_DIR}")

elapsed = time.perf_counter() - start_time

print("\n" + "=" * 65)
print("DOWNLOAD COMPLETED")
print(f"Execution time: {elapsed:.2f} seconds")
print("=" * 65)
