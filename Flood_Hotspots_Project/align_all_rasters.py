import os
import shutil
import rasterio
from rasterio.warp import reproject, Resampling

# --- Paths ---
base_dir = "Flood_Hotspots_Project"
norm_dir = os.path.join(base_dir, "normalized_40m")
resamp_dir = os.path.join(base_dir, "resampled_40m")
out_dir = os.path.join(base_dir, "aligned_40m")
os.makedirs(out_dir, exist_ok=True)

# --- Reference raster (DEM) ---
ref_path = os.path.join(norm_dir, "dem_40m_norm.tif")

with rasterio.open(ref_path) as ref:
    ref_meta = ref.meta.copy()
    ref_crs = ref.crs
    ref_transform = ref.transform
    ref_shape = (ref.height, ref.width)

print("🧭 Using DEM as reference grid:")
print(f"   CRS: {ref_crs}")
print(f"   Shape: {ref_shape}")
print(f"   Resolution: {ref.res}\n")

# --- Identify rasters to align ---
all_files = os.listdir(norm_dir)
months = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
]
rainfall_files = [
    f for f in all_files
    if any(m in f.lower() for m in months) and f.endswith(".tif")
]

pop_files = [f for f in all_files if "pop" in f.lower() and f.endswith(".tif")]

# Core rasters already aligned
aligned_core = [
    "dem_40m_norm.tif",
    "slope_40m_norm.tif",
    "flood_labels_40m.tif"
]

# Add land cover (from resampled folder)
world_cover_path = os.path.join(resamp_dir, "world-cover_40m.tif")

# --- Function to reproject/resample ---
def align_raster(src_path, dst_path):
    with rasterio.open(src_path) as src:
        dst_meta = ref_meta.copy()
        dst_meta.update(dtype="float32", nodata=src.nodata)
        with rasterio.open(dst_path, "w", **dst_meta) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear
            )

# --- Align rainfall rasters ---
print("🌧 Aligning rainfall rasters...")
for f in rainfall_files:
    src = os.path.join(norm_dir, f)
    dst = os.path.join(out_dir, f.replace(".tif", "_aligned.tif"))
    print(f"   → {f}")
    align_raster(src, dst)
print("✅ Rainfall rasters aligned.\n")

# --- Align population rasters ---
print("👥 Aligning population rasters...")
for f in pop_files:
    src = os.path.join(norm_dir, f)
    dst = os.path.join(out_dir, f.replace(".tif", "_aligned.tif"))
    print(f"   → {f}")
    align_raster(src, dst)
print("✅ Population rasters aligned.\n")

# --- Copy already-aligned rasters ---
print("📦 Copying already aligned rasters...")
for f in aligned_core:
    src = os.path.join(norm_dir, f)
    dst = os.path.join(out_dir, f)
    shutil.copy2(src, dst)

shutil.copy2(world_cover_path, os.path.join(out_dir, os.path.basename(world_cover_path)))
print("✅ Core rasters copied.\n")

print("🎯 All rasters are now aligned and saved in:")
print(f"   {out_dir}")
