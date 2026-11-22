
from datasets import load_dataset
from pathlib import Path
import pandas as pd
import time

REPO = "owkin/plism-dataset-tiles"
SHARD_INDEX = 3
TOTAL_SHARDS = 293
MAX_SCAN_ROWS = 8000

ROOT = Path("dataset_exploration/outputs")
INPUT = ROOT / "tables/plism_at2_gt450_metadata_samples.csv"
OUTPUT = ROOT / "tables/plism_matched_tile_candidates.csv"

print("=" * 65)
print("PLISM MATCHED TILE DISCOVERY")
print("=" * 65)

start = time.perf_counter()

# --------------------------------------------------
# 1. Load previously collected AT2 identifiers
# --------------------------------------------------
if not INPUT.exists():
    raise SystemExit(f"Missing input file: {INPUT}")

df = pd.read_csv(INPUT)

at2 = df[
    (df["scanner"] == "AT2") &
    (df["stainer"] == "GIVH")
].copy()

target_ids = set(at2["tile_id"])

print(f"\n[1] AT2 target tile IDs: {len(target_ids)}")

# --------------------------------------------------
# 2. Inspect shard 3 for matching GT450 records
# --------------------------------------------------
filename = (
    f"data/train-{SHARD_INDEX:05d}"
    f"-of-{TOTAL_SHARDS:05d}.parquet"
)

url = f"hf://datasets/{REPO}/{filename}"

print(f"\n[2] Inspecting shard {SHARD_INDEX}...", flush=True)

ds = load_dataset(
    "parquet",
    data_files={"train": url},
    split="train",
    streaming=True
)

ds = ds.select_columns([
    "stainer", "scanner", "slide_id", "tile_id"
])

matches = []
gt450_seen = 0
first_gt450 = None

for i, sample in enumerate(ds):

    if (
        sample["scanner"] == "GT450"
        and sample["stainer"] == "GIVH"
    ):
        gt450_seen += 1

        if first_gt450 is None:
            first_gt450 = sample["tile_id"]

        if sample["tile_id"] in target_ids:
            matches.append({
                "tile_id": sample["tile_id"],
                "stainer": sample["stainer"],
                "scanner_a": "AT2",
                "scanner_b": "GT450",
                "slide_id_a": at2["slide_id"].iloc[0],
                "slide_id_b": sample["slide_id"],
                "source_shard_a": 0,
                "source_shard_b": SHARD_INDEX,
            })

    if (i + 1) % 1000 == 0:
        print(
            f"Scanned {i + 1} rows | "
            f"GT450 rows: {gt450_seen} | "
            f"Matches: {len(matches)}",
            flush=True
        )

    if len(matches) == len(target_ids):
        break

    if i + 1 >= MAX_SCAN_ROWS:
        break

# --------------------------------------------------
# 3. Report results
# --------------------------------------------------
print("\n[3] Results")
print(f"GT450 records examined: {gt450_seen}")
print(f"First GT450 tile ID: {first_gt450}")
print(f"Matching tile IDs: {len(matches)}")

matched_df = pd.DataFrame(matches)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
matched_df.to_csv(OUTPUT, index=False)

if matches:
    print("\nExample matches:")
    print(matched_df.head(10).to_string(index=False))
else:
    print("\nNo matches found in this shard.")
    print("This does not invalidate cross-scanner alignment.")
    print("Other shard ranges may contain matching IDs.")

elapsed = time.perf_counter() - start

print(f"\nSaved: {OUTPUT}")
print(f"Execution time: {elapsed:.2f}s")
print("Matched tile discovery completed.")
