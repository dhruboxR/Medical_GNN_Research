
from datasets import load_dataset
from collections import Counter
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------
DATASET_NAME = "owkin/plism-dataset-tiles"
SAMPLE_SIZE = 30

OUTPUT_DIR = Path("dataset_exploration/outputs")
REPORT_DIR = OUTPUT_DIR / "reports"
TABLE_DIR = OUTPUT_DIR / "tables"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PLISM DATASET INSPECTION")
print("=" * 60)

# --------------------------------------------------
# Step 1: Load dataset in streaming mode
# --------------------------------------------------
print("\n[1] Connecting to Hugging Face...")

dataset = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True
)

print("Streaming connection initialized.")

# --------------------------------------------------
# Step 2: Inspect dataset structure
# --------------------------------------------------
print("\n[2] Dataset features:")

for name, feature in dataset.features.items():
    print(f"  {name}: {feature}")

# Read metadata columns only.
# This avoids decoding images.
metadata_fields = [
    field for field in
    ["__key__", "stainer", "scanner", "slide_id", "tile_id"]
    if field in dataset.features
]

dataset = dataset.select_columns(metadata_fields)

# --------------------------------------------------
# Step 3: Read a small sample
# --------------------------------------------------
print(f"\n[3] Reading {SAMPLE_SIZE} metadata records...")

records = []

for sample in dataset.take(SAMPLE_SIZE):
    records.append(sample)

df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError("No records returned from PLISM.")

print("\nFirst five records:")
print(df.head().to_string(index=False))

# --------------------------------------------------
# Step 4: Inspect scanner and stainer values
# --------------------------------------------------
print("\n[4] Scanner distribution in sample:")

if "scanner" in df.columns:
    print(df["scanner"].value_counts())

print("\nStainer distribution in sample:")

if "stainer" in df.columns:
    print(df["stainer"].value_counts())

print("\nUnique slide IDs in sample:")

if "slide_id" in df.columns:
    print(df["slide_id"].nunique())

# --------------------------------------------------
# Step 5: Inspect possible matching identifiers
# --------------------------------------------------
print("\n[5] Inspecting tile identifiers...")

if "tile_id" in df.columns:
    print(df["tile_id"].head(10).to_string(index=False))

    repeated = Counter(df["tile_id"])
    duplicates = sum(v > 1 for v in repeated.values())

    print(f"\nRepeated tile IDs in sample: {duplicates}")
    print("Note: Repeated IDs do not automatically prove")
    print("that tiles are aligned across scanners.")

# --------------------------------------------------
# Step 6: Save reports
# --------------------------------------------------
csv_path = TABLE_DIR / "plism_sample_metadata.csv"
report_path = REPORT_DIR / "plism_inspection.txt"

df.to_csv(csv_path, index=False)

with open(report_path, "w", encoding="utf-8") as f:
    f.write("PLISM Dataset Initial Inspection\n")
    f.write("=" * 40 + "\n")
    f.write(f"Dataset: {DATASET_NAME}\n")
    f.write(f"Sample records: {len(df)}\n")
    f.write(f"Metadata columns: {list(df.columns)}\n\n")

    for col in ["scanner", "stainer", "slide_id"]:
        if col in df.columns:
            f.write(f"{col} distribution:\n")
            f.write(df[col].value_counts().to_string())
            f.write("\n\n")

print("\n[6] Files saved:")
print(csv_path)
print(report_path)

print("\nPLISM inspection completed successfully.")
