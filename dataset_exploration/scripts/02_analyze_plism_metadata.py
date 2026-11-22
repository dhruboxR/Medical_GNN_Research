
from datasets import load_dataset
from collections import Counter
from pathlib import Path
import pandas as pd
import re
import time

# --------------------------------------------------
# Configuration
# --------------------------------------------------
DATASET_NAME = "owkin/plism-dataset-tiles"
MAX_RECORDS = 100
PROGRESS_INTERVAL = 10

OUTPUT_DIR = Path("dataset_exploration/outputs")
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

start_time = time.perf_counter()

print("=" * 65)
print("PLISM METADATA ANALYSIS")
print("=" * 65)

# --------------------------------------------------
# Step 1: Load dataset in streaming mode
# --------------------------------------------------
print("\n[1] Connecting to Hugging Face...", flush=True)

try:
    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True
    )

    print("Streaming connection initialized.", flush=True)

except Exception as e:
    print(f"Dataset connection failed: {e}")
    raise SystemExit(1)

# --------------------------------------------------
# Step 2: Select metadata only
# --------------------------------------------------
print("\n[2] Selecting metadata columns...", flush=True)

columns = [
    "__key__",
    "stainer",
    "scanner",
    "slide_id",
    "tile_id"
]

available_columns = [
    col for col in columns
    if col in dataset.features
]

dataset = dataset.select_columns(available_columns)

print(f"Selected columns: {available_columns}")

# --------------------------------------------------
# Step 3: Read 100 records with progress
# --------------------------------------------------
print(
    f"\n[3] Reading up to {MAX_RECORDS} records...",
    flush=True
)

records = []

try:
    for i, sample in enumerate(dataset):

        records.append(sample)

        count = i + 1

        if count % PROGRESS_INTERVAL == 0:
            elapsed = time.perf_counter() - start_time

            print(
                f"Processed {count}/{MAX_RECORDS} records "
                f"| Elapsed: {elapsed:.1f}s",
                flush=True
            )

        if count >= MAX_RECORDS:
            break

except KeyboardInterrupt:
    print("\nInterrupted by user.", flush=True)

except Exception as e:
    print(f"\nError while reading records: {e}")

if not records:
    raise SystemExit("No metadata records collected.")

df = pd.DataFrame(records)

print(f"\nTotal records collected: {len(df)}")

# --------------------------------------------------
# Step 4: Scanner distribution
# --------------------------------------------------
print("\n[4] Scanner distribution:")

print(df["scanner"].value_counts().to_string())

# --------------------------------------------------
# Step 5: Stainer distribution
# --------------------------------------------------
print("\n[5] Stainer distribution:")

print(df["stainer"].value_counts().to_string())

# --------------------------------------------------
# Step 6: Slide groups
# --------------------------------------------------
print("\n[6] Slide groups:")

print(df["slide_id"].value_counts().head(20).to_string())

# --------------------------------------------------
# Step 7: Parse tile identifiers
# --------------------------------------------------
print("\n[7] Parsing tile identifiers...")

pattern = re.compile(r"^tile_(\d+)_(\d+)_(\d+)$")

def parse_tile_id(tile_id):

    match = pattern.match(str(tile_id))

    if match:
        return tuple(map(int, match.groups()))

    return (None, None, None)


parsed = df["tile_id"].apply(parse_tile_id)

df[
    ["tile_part_1", "tile_part_2", "tile_part_3"]
] = pd.DataFrame(
    parsed.tolist(),
    index=df.index
)

print("\nParsed tile examples:")

print(
    df[
        [
            "tile_id",
            "tile_part_1",
            "tile_part_2",
            "tile_part_3"
        ]
    ].head(10).to_string(index=False)
)

# --------------------------------------------------
# Step 8: Check repeated tile IDs
# --------------------------------------------------
print("\n[8] Checking repeated tile IDs...")

tile_counts = Counter(df["tile_id"])

repeated_tiles = {
    tile: count
    for tile, count in tile_counts.items()
    if count > 1
}

print(f"Repeated tile IDs: {len(repeated_tiles)}")

print(
    "Note: Repeated IDs alone do not confirm "
    "cross-scanner alignment."
)

# --------------------------------------------------
# Step 9: Scanner-slide combinations
# --------------------------------------------------
print("\n[9] Scanner and slide combinations:")

grouped = (
    df.groupby(["scanner", "slide_id"])
    .size()
    .reset_index(name="count")
)

print(grouped.to_string(index=False))

# --------------------------------------------------
# Step 10: Save reports
# --------------------------------------------------
print("\n[10] Saving reports...")

csv_path = TABLE_DIR / f"plism_metadata_{len(df)}.csv"

report_path = REPORT_DIR / "plism_metadata_analysis.txt"

df.to_csv(csv_path, index=False)

elapsed = time.perf_counter() - start_time

with open(report_path, "w", encoding="utf-8") as f:

    f.write("PLISM Metadata Analysis\n")
    f.write("=" * 45 + "\n\n")

    f.write(f"Dataset: {DATASET_NAME}\n")
    f.write(f"Records collected: {len(df)}\n")
    f.write(f"Unique scanners: {df['scanner'].nunique()}\n")
    f.write(f"Unique stainers: {df['stainer'].nunique()}\n")
    f.write(f"Unique slide groups: {df['slide_id'].nunique()}\n")
    f.write(f"Repeated tile IDs: {len(repeated_tiles)}\n")
    f.write(f"Execution time: {elapsed:.2f} seconds\n\n")

    f.write("Scanner distribution:\n")
    f.write(df["scanner"].value_counts().to_string())

    f.write("\n\nSlide distribution:\n")
    f.write(df["slide_id"].value_counts().to_string())

print("\nSaved files:")
print(csv_path)
print(report_path)

print("\n" + "=" * 65)
print("PLISM METADATA ANALYSIS COMPLETED")
print(f"Total execution time: {elapsed:.2f} seconds")
print("=" * 65)
