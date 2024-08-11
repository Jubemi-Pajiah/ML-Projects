import os
import numpy as np
import rasterio

# Input and output directories
in_dir = "Flood_Hotspots_Project/align_all_rasters"
out_dir = "Flood_Hotspots_Project/normalized_40m"
os.makedirs(out_dir, exist_ok=True)

# List all TIF files in the input directory
files = [f for f in os.listdir(in_dir) if f.endswith(".tif")]

def normalize_array(arr):
    """Normalize array values to 0-1 range, ignoring NaN."""
    arr = arr.astype("float32")
    mask = np.isnan(arr)
    valid = arr[~mask]
    if valid.size == 0:
        return arr
    arr_min, arr_max = valid.min(), valid.max()
    if arr_max == arr_min:
        return np.zeros_like(arr)
    norm = (arr - arr_min) / (arr_max - arr_min)
    norm[mask] = np.nan
    return norm

print("Starting normalization of continuous rasters...\n")

for file_name in files:
    lower = file_name.lower()

    # Skip categorical rasters
    if "world-cover" in lower:
        print(f"⏭Skipping categorical raster: {file_name}")
        continue

    # Define input/output paths
    in_path = os.path.join(in_dir, file_name)
    out_path = os.path.join(out_dir, file_name.replace(".tif", "_norm.tif"))

    print(f"Normalizing: {file_name} ...")

    # Open raster
    with rasterio.open(in_path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile.copy()

    # Replace NoData values with NaN
    if profile.get("nodata") is not None:
        arr[arr == profile["nodata"]] = np.nan

    # Normalize
    norm_arr = normalize_array(arr)

    # Update profile for output
    profile.update(dtype="float32", nodata=np.nan)

    # Save normalized raster
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(norm_arr, 1)

    print(f"Saved normalized raster: {os.path.basename(out_path)}\n")

print("All continuous rasters normalized successfully!")
