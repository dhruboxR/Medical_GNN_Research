
from datasets import load_dataset
from pathlib import Path
import pandas as pd
import time

REPO_ID = "owkin/plism-dataset-tiles"
TOTAL_SHARDS = 293

TARGETS = {
    0: "AT2",
    4: "GT450",
}

SAMPLE_SIZE = 100

OUTPUT = Path("dataset_exploration/outputs/tables")
OUTPUT.mkdir(parents=True, exist_ok=True)

records = []
start = time.perf_counter()

print("=" * 60)
print("PLISM CROSS-SCANNER TILE ID VERIFICATION")
print("=" * 60)

for shard_index, expected_scanner in TARGETS.items():

    filename = (
        f"data/train-{shard_index:05d}"
        f"-of-{TOTAL_SHARDS:05d}.parquet"
    )

    url = f"hf://datasets/{REPO_ID}/{filename}"

    print(
        f"\nChecking shard {shard_index} "
        f"({expected_scanner})...",
        flush=True
    )

    ds = load_dataset(
        "parquet",
        data_files={"train": url},
        split="train",
        streaming=True
    )

    ds = ds.select_columns([
        "stainer", "scanner", "slide_id", "tile_id"
    ])

    count = 0

    for sample in ds:
        if (
            sample["scanner"] == expected_scanner
            and sample["stainer"] == "GIVH"
        ):
            records.append({
                "shard": shard_index,
                **sample
            })
            count += 1

        if count >= SAMPLE_SIZE:
            break

    print(f"Collected {count} matching records.", flush=True)

if not records:
    raise SystemExit("No matching metadata retrieved.")

df = pd.DataFrame(records)

print("\n[1] Sampled scanner groups:")
print(
    df.groupby(["scanner", "slide_id"])
      .size()
      .to_string()
)

at2 = set(
    df.loc[df["scanner"] == "AT2", "tile_id"]
)

gt450 = set(
    df.loc[df["scanner"] == "GT450", "tile_id"]
)

shared = sorted(at2.intersection(gt450))

print("\n[2] Tile identifier comparison:")
print(f"AT2 unique IDs: {len(at2)}")
print(f"GT450 unique IDs: {len(gt450)}")
print(f"Shared tile IDs: {len(shared)}")

print("\n[3] Sample identifiers:")
for scanner in ["AT2", "GT450"]:
    print(f"\n{scanner}:")
    values = df.loc[
        df["scanner"] == scanner, "tile_id"
    ].head(5)

    print(values.to_string(index=False))

print("\n[4] Shared IDs:")
print(shared[:20] if shared else "None in sampled rows")

df.to_csv(
    OUTPUT / "plism_at2_gt450_metadata_samples.csv",
    index=False
)

pd.DataFrame({
    "tile_id": shared
}).to_csv(
    OUTPUT / "plism_at2_gt450_shared_ids.csv",
    index=False
)

print("\nImportant:")
print(
    "No shared IDs in these samples does not "
    "mean the full scanner groups lack matches."
)

print(
    f"\nExecution time: "
    f"{time.perf_counter() - start:.2f}s"
)
print("Verification completed.")
