import os
import rasterio

norm_dir = "Flood_Hotspots_Project/normalized_40m"
resamp_dir = "Flood_Hotspots_Project/resampled_40m"

# Collect all normalized rasters (DEM, slope, rainfall, population, flood labels)
raster_paths = [
    os.path.join(norm_dir, f)
    for f in os.listdir(norm_dir)
    if f.endswith(".tif")
]

# Add world cover from resampled folder
raster_paths.append(os.path.join(resamp_dir, "world-cover_40m.tif"))

print("📏 Checking raster dimensions...\n")

for path in raster_paths:
    with rasterio.open(path) as src:
        print(f"{os.path.basename(path):35s} -> shape: {src.height}×{src.width} | res: {src.res}")

print("\n✅ Check complete — compare shapes above.")
