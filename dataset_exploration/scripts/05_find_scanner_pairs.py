
from datasets import load_dataset
from collections import defaultdict
from pathlib import Path
import pandas as pd
import time

REPO_ID = "owkin/plism-dataset-tiles"
TOTAL_SHARDS = 293

# Inspect nearby shards to look for matching stainers.
SHARD_INDICES = [0, 4, 8, 12, 16, 20, 24]
SAMPLES_PER_SHARD = 2

OUTPUT_DIR = Path("dataset_exploration/outputs")
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "stainer", "scanner", "slide_id", "tile_id"
]

print("=" * 65)
print("PLISM SAME-STAINER SCANNER PAIR DISCOVERY")
print("=" * 65)

start_time = time.perf_counter()
records = []

for index in SHARD_INDICES:

    filename = (
        f"data/train-{index:05d}"
        f"-of-{TOTAL_SHARDS:05d}.parquet"
    )

    url = f"hf://datasets/{REPO_ID}/{filename}"

    print(f"\nChecking shard {index}...", flush=True)

    try:
        ds = load_dataset(
            "parquet",
            data_files={"train": url},
            split="train",
            streaming=True
        )

        ds = ds.select_columns(COLUMNS)

        for sample in ds.take(SAMPLES_PER_SHARD):
            records.append({
                "shard": index,
                **sample
            })

        print("  Metadata retrieved.", flush=True)

    except Exception as exc:
        print(f"  Error: {exc}", flush=True)

if not records:
    raise SystemExit("No records collected.")

df = pd.DataFrame(records)

print("\n[1] Discovered scanner/stainer groups:")

groups = (
    df.groupby(
        ["stainer", "scanner", "slide_id"],
        as_index=False
    )
    .agg(
        sampled_rows=("tile_id", "size"),
        example_shard=("shard", "min")
    )
)

print(groups.to_string(index=False))

print("\n[2] Looking for same-stainer scanner pairs...")

by_stainer = defaultdict(set)

for _, row in groups.iterrows():
    by_stainer[row["stainer"]].add(row["scanner"])

found = False

for stainer, scanners in sorted(by_stainer.items()):
    if len(scanners) >= 2:
        found = True
        print(
            f"Stainer: {stainer} | "
            f"Scanners: {sorted(scanners)}"
        )

if not found:
    print(
        "No same-stainer scanner pair identified "
        "in this sample."
    )

print("\n[3] Saving results...")

csv_path = TABLE_DIR / "plism_scanner_pair_candidates.csv"
report_path = REPORT_DIR / "plism_scanner_pair_report.txt"

groups.to_csv(csv_path, index=False)

with open(report_path, "w", encoding="utf-8") as f:
    f.write("PLISM Same-Stainer Scanner Discovery\n")
    f.write("=" * 45 + "\n\n")
    f.write(groups.to_string(index=False))
    f.write("\n\nSame-stainer candidates:\n")

    for stainer, scanners in sorted(by_stainer.items()):
        if len(scanners) >= 2:
            f.write(
                f"{stainer}: {sorted(scanners)}\n"
            )

elapsed = time.perf_counter() - start_time

print(f"\nSaved: {csv_path}")
print(f"Saved: {report_path}")
print(f"\nExecution time: {elapsed:.2f}s")
print("Discovery completed.")
