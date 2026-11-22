
from datasets import load_dataset
from pathlib import Path
import pandas as pd
import time

# Configuration
REPO_ID = "owkin/plism-dataset-tiles"
SHARD_INDICES = [0, 75, 150, 225, 292]
TOTAL_SHARDS = 293
SAMPLES_PER_SHARD = 10

OUTPUT_DIR = Path("dataset_exploration/outputs")
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "__key__",
    "stainer",
    "scanner",
    "slide_id",
    "tile_id"
]

print("=" * 65)
print("PLISM SHARD METADATA INSPECTION")
print("=" * 65)

start = time.perf_counter()
all_records = []

for shard_index in SHARD_INDICES:

    filename = (
        f"data/train-{shard_index:05d}"
        f"-of-{TOTAL_SHARDS:05d}.parquet"
    )

    url = (
        f"hf://datasets/{REPO_ID}/{filename}"
    )

    print(f"\nInspecting shard {shard_index}...", flush=True)

    shard_start = time.perf_counter()

    try:
        dataset = load_dataset(
            "parquet",
            data_files={"train": url},
            split="train",
            streaming=True
        )

        # Avoid decoding or retrieving PNG columns.
        dataset = dataset.select_columns(COLUMNS)

        count = 0

        for sample in dataset.take(SAMPLES_PER_SHARD):
            sample["shard_index"] = shard_index
            all_records.append(sample)
            count += 1

        elapsed = time.perf_counter() - shard_start

        print(
            f"  Records: {count}"
            f" | Time: {elapsed:.2f}s",
            flush=True
        )

    except Exception as exc:
        print(
            f"  Error on shard {shard_index}: {exc}",
            flush=True
        )

if not all_records:
    raise SystemExit("No metadata was retrieved.")

df = pd.DataFrame(all_records)

print("\n" + "=" * 65)
print("RESULTS")
print("=" * 65)

print("\n[1] Sampled scanner groups:")
print(
    df.groupby(["shard_index", "scanner", "stainer"])
      .size()
      .reset_index(name="records")
      .to_string(index=False)
)

print("\n[2] Slide groups:")
print(
    df.groupby(
        ["shard_index", "slide_id", "scanner", "stainer"]
    )
    .size()
    .reset_index(name="records")
    .to_string(index=False)
)

print("\n[3] Unique scanners:")
print(sorted(df["scanner"].dropna().unique()))

print("\n[4] Unique stainers:")
print(sorted(df["stainer"].dropna().unique()))

print("\n[5] Example tile identifiers:")
print(
    df[
        ["shard_index", "slide_id", "tile_id"]
    ].head(15).to_string(index=False)
)

# Save results
csv_path = TABLE_DIR / "plism_shard_samples.csv"
report_path = REPORT_DIR / "plism_shard_inspection.txt"

df.to_csv(csv_path, index=False)

elapsed = time.perf_counter() - start

with open(report_path, "w", encoding="utf-8") as f:
    f.write("PLISM Shard Inspection\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Sampled shards: {SHARD_INDICES}\n")
    f.write(f"Records retrieved: {len(df)}\n")
    f.write(f"Scanners found: {df['scanner'].unique().tolist()}\n")
    f.write(f"Stainers found: {df['stainer'].unique().tolist()}\n")
    f.write(f"Execution time: {elapsed:.2f}s\n\n")
    f.write("Slide groups:\n")
    f.write(
        df.groupby(
            ["shard_index", "slide_id", "scanner", "stainer"]
        ).size().to_string()
    )

print("\n[6] Saved reports:")
print(csv_path)
print(report_path)

print(f"\nExecution time: {elapsed:.2f}s")
print("Inspection completed.")
