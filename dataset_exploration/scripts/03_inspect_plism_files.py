
from huggingface_hub import HfApi
from collections import Counter
from pathlib import Path
import csv

REPO_ID = "owkin/plism-dataset-tiles"

output_dir = Path("dataset_exploration/outputs/tables")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PLISM REMOTE FILE INVENTORY")
print("=" * 60)

api = HfApi()

print("\n[1] Retrieving file list...", flush=True)

files = api.list_repo_files(
    repo_id=REPO_ID,
    repo_type="dataset"
)

print(f"Total repository files: {len(files)}")

print("\n[2] File extension distribution:")

extensions = Counter(
    Path(file).suffix.lower() or "(no extension)"
    for file in files
)

for extension, count in extensions.most_common():
    print(f"{extension}: {count}")

print("\n[3] First 20 data files:")

data_files = [
    file for file in files
    if file.startswith("data/")
]

for file in data_files[:20]:
    print(file)

print(f"\nTotal data files: {len(data_files)}")

print("\n[4] Saving inventory...")

output_file = output_dir / "plism_remote_file_inventory.csv"

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["file_path"])

    for file in files:
        writer.writerow([file])

print(f"Saved: {output_file}")
print("\nInventory completed.")
