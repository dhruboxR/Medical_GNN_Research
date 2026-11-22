
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np

# --------------------------------------------------
# Configuration
# --------------------------------------------------
ROOT = Path("datasets/PLISM/GIVH_AT2_GT450")
MANIFEST = Path(
    "dataset_exploration/outputs/tables/"
    "plism_download_manifest.csv"
)
OUTPUT_DIR = Path("dataset_exploration/outputs/figures")
TABLE_DIR = Path("dataset_exploration/outputs/tables")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

MAX_PAIRS = 12
TILE_SIZE = 224
PADDING = 16
HEADER_HEIGHT = 58
ROW_HEIGHT = TILE_SIZE + 45

print("=" * 65)
print("PLISM CROSS-SCANNER IMAGE VISUALIZATION")
print("=" * 65)

# --------------------------------------------------
# Step 1: Validate manifest
# --------------------------------------------------
df = pd.read_csv(MANIFEST)

required = {"tile_id", "scanner"}
if not required.issubset(df.columns):
    raise ValueError("Manifest is missing required columns.")

counts = df.groupby("tile_id")["scanner"].nunique()
paired_ids = sorted(counts[counts == 2].index)

selected_ids = paired_ids[:MAX_PAIRS]

print(f"\nComplete pairs available: {len(paired_ids)}")
print(f"Pairs selected for visualization: {len(selected_ids)}")

if not selected_ids:
    raise SystemExit("No complete pairs found.")

# --------------------------------------------------
# Step 2: Create comparison canvas
# --------------------------------------------------
width = TILE_SIZE * 2 + PADDING * 3
height = HEADER_HEIGHT + ROW_HEIGHT * len(selected_ids)

canvas = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(canvas)

draw.text(
    (PADDING, 12),
    "Scanner AT2",
    fill="black"
)

draw.text(
    (TILE_SIZE + PADDING * 2, 12),
    "Scanner GT450",
    fill="black"
)

metrics = []

# --------------------------------------------------
# Step 3: Compare matched pairs
# --------------------------------------------------
for index, tile_id in enumerate(selected_ids):

    path_a = ROOT / "AT2" / f"{tile_id}.png"
    path_b = ROOT / "GT450" / f"{tile_id}.png"

    if not path_a.exists() or not path_b.exists():
        raise FileNotFoundError(tile_id)

    with Image.open(path_a) as img:
        image_a = img.convert("RGB")

    with Image.open(path_b) as img:
        image_b = img.convert("RGB")

    if image_a.size != image_b.size:
        raise ValueError(f"Size mismatch: {tile_id}")

    if image_a.size != (TILE_SIZE, TILE_SIZE):
        raise ValueError(f"Unexpected size: {tile_id}")

    y = HEADER_HEIGHT + index * ROW_HEIGHT

    canvas.paste(image_a, (PADDING, y))
    canvas.paste(
        image_b,
        (TILE_SIZE + PADDING * 2, y)
    )

    draw.text(
        (PADDING, y + TILE_SIZE + 6),
        tile_id,
        fill="black"
    )

    # Descriptive pixel difference only.
    # Not a registration-quality measurement.
    a = np.asarray(image_a, dtype=np.float32)
    b = np.asarray(image_b, dtype=np.float32)

    mean_absolute_difference = float(
        np.abs(a - b).mean()
    )

    metrics.append({
        "tile_id": tile_id,
        "mean_abs_pixel_difference": (
            mean_absolute_difference
        ),
        "width": image_a.width,
        "height": image_a.height
    })

    print(
        f"Compared {index + 1}/{len(selected_ids)}: "
        f"{tile_id}"
    )

# --------------------------------------------------
# Step 4: Save comparison image
# --------------------------------------------------
image_path = (
    OUTPUT_DIR / "plism_at2_gt450_comparison.png"
)

canvas.save(image_path)

# --------------------------------------------------
# Step 5: Save descriptive metrics
# --------------------------------------------------
metrics_df = pd.DataFrame(metrics)

csv_path = (
    TABLE_DIR / "plism_pair_visual_metrics.csv"
)

metrics_df.to_csv(csv_path, index=False)

print("\n" + "=" * 65)
print("VISUALIZATION COMPLETED")
print("=" * 65)

print(f"Comparison image: {image_path}")
print(f"Metrics CSV: {csv_path}")

print(
    "Mean absolute pixel difference: "
    f"{metrics_df['mean_abs_pixel_difference'].mean():.3f}"
)

print("\nNote:")
print("Pixel differences reflect scanner appearance,")
print("color variation, and possible spatial offsets.")
print("They are not evidence of graph topology drift.")
