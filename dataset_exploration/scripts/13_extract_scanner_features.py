
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

# ==================================================
# CONFIGURATION
# ==================================================

MODEL_NAME = "facebook/dinov2-small"
BATCH_SIZE = 8

ROOT = Path("datasets/PLISM/GIVH_AT2_GT450")
MANIFEST = Path(
    "dataset_exploration/outputs/tables/"
    "plism_spatial_region_download_manifest.csv"
)
OUTPUT = Path(
    "dataset_exploration/outputs/features"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

SCANNERS = ["AT2", "GT450"]
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 65)
print("PLISM CROSS-SCANNER FEATURE EXTRACTION")
print("=" * 65)
print(f"Device: {device}")
print(f"Model: {MODEL_NAME}")

# ==================================================
# STEP 1: LOAD VERIFIED MANIFEST
# ==================================================

df = pd.read_csv(MANIFEST)

df = df[
    df["complete_pair"].astype(str).str.lower() == "true"
].copy()

df = df.sort_values(
    ["grid_row", "grid_col"]
).reset_index(drop=True)

if df.empty:
    raise RuntimeError("No complete matched pairs.")

tile_ids = df["tile_id"].tolist()

print(f"\nMatched locations: {len(tile_ids)}")

# ==================================================
# STEP 2: LOAD PRETRAINED MODEL
# ==================================================

print("\nLoading DINOv2-Small...", flush=True)

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
).to(device)

model.eval()

# ==================================================
# STEP 3: FEATURE EXTRACTION
# ==================================================

features = {}

for scanner in SCANNERS:

    print(f"\nExtracting {scanner} features...")

    scanner_features = []

    for start in tqdm(
        range(0, len(tile_ids), BATCH_SIZE),
        desc=scanner
    ):

        batch_ids = tile_ids[
            start:start + BATCH_SIZE
        ]

        images = []

        for tile_id in batch_ids:

            path = ROOT / scanner / f"{tile_id}.png"

            with Image.open(path) as img:
                images.append(img.convert("RGB"))

        inputs = processor(
            images=images,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = model(**inputs)

            # CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]

        scanner_features.append(
            embeddings.cpu().numpy()
        )

    features[scanner] = np.concatenate(
        scanner_features,
        axis=0
    )

    print(
        f"{scanner} feature shape: "
        f"{features[scanner].shape}"
    )

# ==================================================
# STEP 4: SAVE FEATURE VECTORS
# ==================================================

print("\nSaving extracted features...")

feature_path = OUTPUT / "plism_dinov2_features.npz"

np.savez_compressed(
    feature_path,
    at2=features["AT2"],
    gt450=features["GT450"],
    tile_ids=np.array(tile_ids),
    grid_rows=df["grid_row"].to_numpy(),
    grid_cols=df["grid_col"].to_numpy()
)

# ==================================================
# STEP 5: PAIRED FEATURE SIMILARITY
# ==================================================

print("\nCalculating paired feature similarity...")

a = features["AT2"]
b = features["GT450"]

a_norm = a / np.maximum(
    np.linalg.norm(a, axis=1, keepdims=True),
    1e-12
)

b_norm = b / np.maximum(
    np.linalg.norm(b, axis=1, keepdims=True),
    1e-12
)

cosine_similarity = np.sum(
    a_norm * b_norm,
    axis=1
)

euclidean_distance = np.linalg.norm(
    a - b,
    axis=1
)

results = pd.DataFrame({
    "tile_id": tile_ids,
    "grid_row": df["grid_row"],
    "grid_col": df["grid_col"],
    "cosine_similarity": cosine_similarity,
    "euclidean_distance": euclidean_distance
})

csv_path = OUTPUT / "plism_dinov2_pair_similarity.csv"

results.to_csv(csv_path, index=False)

# ==================================================
# STEP 6: REPORT
# ==================================================

print("\n" + "=" * 65)
print("FEATURE EXTRACTION COMPLETED")
print("=" * 65)

print(f"Matched pairs: {len(tile_ids)}")
print(f"Feature dimension: {a.shape[1]}")

print(
    f"Mean paired cosine similarity: "
    f"{cosine_similarity.mean():.6f}"
)

print(
    f"Minimum paired cosine similarity: "
    f"{cosine_similarity.min():.6f}"
)

print(
    f"Maximum paired cosine similarity: "
    f"{cosine_similarity.max():.6f}"
)

print(
    f"Mean Euclidean distance: "
    f"{euclidean_distance.mean():.6f}"
)

print(f"\nFeatures saved: {feature_path}")
print(f"Metrics saved: {csv_path}")
