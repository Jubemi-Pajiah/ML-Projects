import os
import glob
import numpy as np
import pandas as pd
import rasterio

# ================= CONFIG =================
base_dir = "Flood_Hotspots_Project"
final_dir = os.path.join(base_dir, "final_40m")
output_path = os.path.join(base_dir, "outputs")
os.makedirs(output_path, exist_ok=True)

# Label and features
label_raster = os.path.join(final_dir, "flood_labels_40m.tif")
feature_rasters = sorted([
    os.path.join(final_dir, f) for f in os.listdir(final_dir)
    if f.endswith(".tif") and not f.startswith("flood_labels")
])

# ================= FUNCTIONS =================
def load_raster(path):
    """Load raster and flatten to 1D array."""
    with rasterio.open(path) as src:
        arr = src.read(1)
        arr = arr.astype("float32")
    return arr.flatten()

def get_raster_meta(path):
    """Return raster shape, CRS, and resolution info."""
    with rasterio.open(path) as src:
        return {
            "name": os.path.basename(path),
            "shape": src.shape,
            "crs": src.crs,
            "res": src.res,
            "bounds": src.bounds
        }

# ================= STEP 1: LOAD LABEL =================
print("🚀 Loading rasters for stacking...\n")

print(f"📦 Label raster: {os.path.basename(label_raster)}")
y = load_raster(label_raster)
print("✅ Loaded flood label raster.\n")

# ================= STEP 2: LOAD FEATURES =================
feature_arrays = []
feature_names = []

for fpath in feature_rasters:
    name = os.path.basename(fpath)
    # Skip duplicates (e.g., intermediate files)
    if not any(key in name.lower() for key in [
        "dem_40m_norm", "lagos_pop_40m_norm", "worldcover_40m"
    ]) and "_40m_norm" not in name:
        continue
    if "flood" in name.lower():
        continue

    print(f"✅ Loaded feature: {name}")
    arr = load_raster(fpath)
    feature_arrays.append(arr)
    feature_names.append(name.replace(".tif", ""))

# ================= STEP 3: STACK & CLEAN =================
print("\n🔄 Stacking and flattening rasters...")
X = np.stack(feature_arrays, axis=1)

# Combine label + features
data = np.column_stack((y, X))
columns = ["flood_label"] + feature_names
df = pd.DataFrame(data, columns=columns)

# Remove NaNs and invalids
before = len(df)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
after = len(df)
removed = before - after
print(f"🧹 Removed {removed:,} rows with NaN or invalid values.")

# ================= STEP 4: CATEGORICAL HANDLING =================
# Encode worldcover if present
if "worldcover_40m" in df.columns:
    df["worldcover_40m"] = df["worldcover_40m"].astype(int)
    print("✅ Encoded WorldCover as integer categories.")

# ================= STEP 5: SAVE =================
final_csv = os.path.join(output_path, "lagos_flood_dataset.csv")
df.to_csv(final_csv, index=False)
print(f"\n💾 Dataset saved at: {final_csv}")

print(f"✅ Final dataset size: {df.shape[0]:,} rows × {df.shape[1]} columns")
print("\n🎯 Step complete — dataset ready for ML training!")
