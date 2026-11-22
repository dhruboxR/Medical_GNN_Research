
from pathlib import Path
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np

# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path("datasets/PLISM/GIVH_AT2_GT450")

MANIFEST = Path(
    "dataset_exploration/outputs/tables/"
    "plism_spatial_region_download_manifest.csv"
)

FIGURE_DIR = Path(
    "dataset_exploration/outputs/figures"
)

TABLE_DIR = Path(
    "dataset_exploration/outputs/tables"
)

REPORT_DIR = Path(
    "dataset_exploration/outputs/reports"
)

for folder in [FIGURE_DIR, TABLE_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

SCANNERS = ["AT2", "GT450"]
TILE_SIZE = 224
GAP = 20

print("=" * 65)
print("PLISM SPATIAL MOSAIC VALIDATION")
print("=" * 65)

# ==================================================
# STEP 1: LOAD MANIFEST
# ==================================================

df = pd.read_csv(MANIFEST)

required = {
    "tile_id", "grid_row", "grid_col",
    "at2_valid", "gt450_valid", "complete_pair"
}

missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing manifest columns: {missing}")

df = df[df["complete_pair"] == True].copy()

if df.empty:
    raise RuntimeError("No complete matched pairs found.")

if df[["grid_row", "grid_col"]].duplicated().any():
    raise ValueError("Duplicate grid coordinates found.")

min_row = int(df["grid_row"].min())
max_row = int(df["grid_row"].max())
min_col = int(df["grid_col"].min())
max_col = int(df["grid_col"].max())

n_rows = max_row - min_row + 1
n_cols = max_col - min_col + 1

print(f"\n[1] Matched pairs: {len(df)}")
print(f"Grid dimensions: {n_rows} x {n_cols}")
print(f"Rows: {min_row} to {max_row}")
print(f"Columns: {min_col} to {max_col}")

# ==================================================
# STEP 2: RECONSTRUCT MOSAICS
# ==================================================

mosaics = {}

mosaic_width = n_cols * TILE_SIZE
mosaic_height = n_rows * TILE_SIZE

for scanner in SCANNERS:

    print(f"\n[2] Building {scanner} mosaic...")

    mosaic = Image.new(
        "RGB",
        (mosaic_width, mosaic_height),
        (210, 210, 210)
    )

    for _, row in df.iterrows():

        tile_id = row["tile_id"]

        image_path = (
            ROOT / scanner / f"{tile_id}.png"
        )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if image.size != (TILE_SIZE, TILE_SIZE):
                raise ValueError(
                    f"Unexpected image size: {image_path}"
                )

            x = (
                int(row["grid_col"]) - min_col
            ) * TILE_SIZE

            y = (
                int(row["grid_row"]) - min_row
            ) * TILE_SIZE

            mosaic.paste(image, (x, y))

    mosaics[scanner] = mosaic

    output_path = (
        FIGURE_DIR / f"plism_{scanner}_mosaic.png"
    )

    mosaic.save(output_path)

    print(f"Saved: {output_path}")

# ==================================================
# STEP 3: SIDE-BY-SIDE MOSAIC
# ==================================================

print("\n[3] Creating comparison mosaic...")

comparison = Image.new(
    "RGB",
    (
        mosaic_width * 2 + GAP,
        mosaic_height + 40
    ),
    "white"
)

comparison.paste(mosaics["AT2"], (0, 40))

comparison.paste(
    mosaics["GT450"],
    (mosaic_width + GAP, 40)
)

draw = ImageDraw.Draw(comparison)

draw.text((10, 10), "AT2", fill="black")
draw.text(
    (mosaic_width + GAP + 10, 10),
    "GT450",
    fill="black"
)

comparison_path = (
    FIGURE_DIR / "plism_spatial_mosaic_comparison.png"
)

comparison.save(comparison_path)

print(f"Saved: {comparison_path}")

# ==================================================
# STEP 4: IMAGE QUALITY METRICS
# ==================================================

print("\n[4] Calculating image quality metrics...")

metrics = []

for _, row in df.iterrows():

    tile_id = row["tile_id"]

    images = {}

    for scanner in SCANNERS:

        path = ROOT / scanner / f"{tile_id}.png"

        with Image.open(path) as img:
            images[scanner] = np.asarray(
                img.convert("RGB"),
                dtype=np.float32
            )

    a = images["AT2"]
    b = images["GT450"]

    # Descriptive pixel difference
    mae = float(np.abs(a - b).mean())

    # Approximate background exclusion.
    # This is only a heuristic for H&E imagery.
    def tissue_fraction(image):
        white_background = np.all(
            image > 220,
            axis=2
        )

        return float(
            (~white_background).mean()
        )

    metrics.append({
        "tile_id": tile_id,
        "grid_row": int(row["grid_row"]),
        "grid_col": int(row["grid_col"]),
        "at2_mean_brightness": float(a.mean()),
        "gt450_mean_brightness": float(b.mean()),
        "pixel_mae": mae,
        "at2_tissue_fraction": tissue_fraction(a),
        "gt450_tissue_fraction": tissue_fraction(b)
    })

metrics_df = pd.DataFrame(metrics)

metrics_path = (
    TABLE_DIR / "plism_spatial_mosaic_metrics.csv"
)

metrics_df.to_csv(metrics_path, index=False)

# ==================================================
# STEP 5: REPORT
# ==================================================

print("\n[5] Summary statistics:")

summary = {
    "Matched pairs": len(df),
    "Grid rows": n_rows,
    "Grid columns": n_cols,
    "Grid coverage": len(df) / (n_rows * n_cols),
    "Mean pixel MAE": metrics_df["pixel_mae"].mean(),
    "AT2 mean brightness":
        metrics_df["at2_mean_brightness"].mean(),
    "GT450 mean brightness":
        metrics_df["gt450_mean_brightness"].mean(),
    "AT2 mean tissue fraction":
        metrics_df["at2_tissue_fraction"].mean(),
    "GT450 mean tissue fraction":
        metrics_df["gt450_tissue_fraction"].mean()
}

for key, value in summary.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

report_path = (
    REPORT_DIR / "plism_spatial_mosaic_report.txt"
)

with open(report_path, "w", encoding="utf-8") as f:
    f.write("PLISM Spatial Mosaic Validation\n")
    f.write("=" * 45 + "\n\n")

    for key, value in summary.items():
        f.write(f"{key}: {value}\n")

    f.write(
        "\nNotes:\n"
        "- Tile coordinates are inferred from tile IDs.\n"
        "- Gray positions indicate missing tiles.\n"
        "- Tissue fraction uses a simple brightness threshold.\n"
        "- Pixel MAE is not a registration metric.\n"
        "- Graph stability has not yet been evaluated.\n"
    )

print("\n[6] Output files:")
print(comparison_path)
print(metrics_path)
print(report_path)

print("\n" + "=" * 65)
print("SPATIAL MOSAIC VALIDATION COMPLETED")
print("=" * 65)
