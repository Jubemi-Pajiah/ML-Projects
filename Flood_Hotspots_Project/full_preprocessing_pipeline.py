import os
import glob
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
import geopandas as gpd
from config import mask_layer, mask_path, raw_dir, final_dir, target_crs

os.makedirs(final_dir, exist_ok=True)


# Helpers
def normalize_array(arr):
    arr = arr.astype("float32")
    valid = np.isfinite(arr)
    if not valid.any():
        return arr
    mn, mx = np.nanmin(arr[valid]), np.nanmax(arr[valid])
    if mx == mn:
        out = np.zeros_like(arr, dtype="float32")
        out[~valid] = np.nan
        return out
    out = (arr - mn) / (mx - mn)
    out[~valid] = np.nan
    return out

def clip_to_boundary(src_path, gdf, out_path):
    with rasterio.open(src_path) as src:
        gdf_src = gdf.to_crs(src.crs)
        geom = [gdf_src.union_all()]
        data, transform = mask(src, geom, crop=True)
        meta = src.meta.copy()
        meta.update({"height": data.shape[1], "width": data.shape[2], "transform": transform})
        tmp = out_path + ".tmp"
        with rasterio.open(tmp, "w", **meta) as dst:
            dst.write(data)
    os.replace(tmp, out_path)
    return out_path

def align_to_ref(src_path, ref_path, out_path, resampling):
    """Resample src onto the grid of ref_path (shape/transform/crs)."""
    with rasterio.open(src_path) as src, rasterio.open(ref_path) as ref:
        dst_profile = ref.profile.copy()
        dst_profile.update({"dtype": "float32", "count": 1})
        dst = np.empty((ref.height, ref.width), dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            resampling=resampling
        )
        tmp = out_path + ".tmp"
        with rasterio.open(tmp, "w", **dst_profile) as out:
            out.write(dst, 1)
    os.replace(tmp, out_path)
    return out_path

def align_categorical_to_ref(src_path, ref_path, out_path):
    """Nearest-neighbour alignment for categorical (e.g., WorldCover)."""
    with rasterio.open(src_path) as src, rasterio.open(ref_path) as ref:
        dst_profile = ref.profile.copy()
        dst_profile.update({"dtype": "uint16", "count": 1, "nodata": 0})
        dst = np.zeros((ref.height, ref.width), dtype="uint16")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            resampling=Resampling.nearest
        )
        tmp = out_path + ".tmp"
        with rasterio.open(tmp, "w", **dst_profile) as out:
            out.write(dst, 1)
    os.replace(tmp, out_path)
    return out_path

def remove_everything_except(keepers):
    for f in os.listdir(final_dir):
        fp = os.path.join(final_dir, f)
        if not os.path.isfile(fp): 
            continue
        if f not in keepers:
            os.remove(fp)

# 0) Load boundary, set reference grid
print("📍 Loading boundary & reference grid...")
gdf = gpd.read_file(mask_path, layer=mask_layer).to_crs(target_crs)

# Use the normalized DEM as the reference grid
ref_candidates = [
    os.path.join(final_dir, "dem_40m_norm.tif"),
    os.path.join(final_dir, "dem_40m.tif")
]
ref_path = None
for c in ref_candidates:
    if os.path.exists(c):
        ref_path = c
        break
if ref_path is None:
    raise FileNotFoundError("No DEM reference found (dem_40m_norm.tif or dem_40m.tif).")

print(f"Reference grid: {os.path.basename(ref_path)}")

# 1) Rebuild WorldCover (categorical) aligned to DEM grid
print("Rebuilding WorldCover (categorical)...")
wc_raw = glob.glob(os.path.join(raw_dir, "world-cover", "*.tif"))
if len(wc_raw) == 0:
    print("No world-cover rasters found in raw/world-cover/. Skipping.")
else:
    # If multiple worldcover tiles, clip+merge in QGIS beforehand; here we assume one tile already clipped to Lagos or a single tile covering it.
    # Clip to boundary first to reduce size
    wc_clip = os.path.join(final_dir, "worldcover_clip.tif")
    wc_final = os.path.join(final_dir, "worldcover_40m.tif")
    clip_to_boundary(wc_raw[0], gdf, wc_clip)
    align_categorical_to_ref(wc_clip, ref_path, wc_final)
    # Clean temp
    if os.path.exists(wc_clip): os.remove(wc_clip)
    print("worldcover_40m.tif saved.")

# 2) Rebuild Population aligned to DEM grid and normalize
print("👥 Rebuilding Population (align + normalize)...")
pop_raw = glob.glob(os.path.join(raw_dir, "pop", "*.tif"))
if len(pop_raw) == 0:
    print("No population raster found in raw/pop/. Skipping.")
else:
    pop_clip = os.path.join(final_dir, "pop_clip_deg.tif")
    pop_aligned = os.path.join(final_dir, "lagos_pop_40m.tif")
    pop_norm = os.path.join(final_dir, "lagos_pop_40m_norm.tif")

    # Clip to boundary in native CRS, then align to ref grid (bilinear)
    clip_to_boundary(pop_raw[0], gdf, pop_clip)
    align_to_ref(pop_clip, ref_path, pop_aligned, Resampling.bilinear)

    # Normalize
    with rasterio.open(pop_aligned) as src:
        arr = src.read(1)
        norm = normalize_array(arr)
        prof = src.profile.copy()
    with rasterio.open(pop_norm + ".tmp", "w", **prof) as dst:
        dst.write(norm, 1)
    os.replace(pop_norm + ".tmp", pop_norm)

    # remove intermediate
    for p in [pop_clip, pop_aligned]:
        if os.path.exists(p): os.remove(p)

    print("lagos_pop_40m_norm.tif saved.")

# 3) Rebuild ALL rainfall rasters aligned & normalized
print("🌧 Rebuilding CHIRPS rainfall rasters (align + normalize)...")
rain_raw = sorted(glob.glob(os.path.join(raw_dir, "chirps", "*.tif")))
if len(rain_raw) == 0:
    print("No CHIRPS rasters in raw/chirps/. Skipping.")
else:
    for rp in rain_raw:
        base = os.path.splitext(os.path.basename(rp))[0]
        clip_p = os.path.join(final_dir, f"{base}_clip.tif")
        aligned = os.path.join(final_dir, f"{base}_40m.tif")
        out_norm = os.path.join(final_dir, f"{base}_40m_norm.tif")

        # clip then align (bilinear)
        clip_to_boundary(rp, gdf, clip_p)
        align_to_ref(clip_p, ref_path, aligned, Resampling.bilinear)

        # normalize
        with rasterio.open(aligned) as src:
            arr = src.read(1)
            norm = normalize_array(arr)
            prof = src.profile.copy()
        with rasterio.open(out_norm + ".tmp", "w", **prof) as dst:
            dst.write(norm, 1)
        os.replace(out_norm + ".tmp", out_norm)

        # remove intermediates
        for p in [clip_p, aligned]:
            if os.path.exists(p): os.remove(p)

        print(f"Saved: {os.path.basename(out_norm)}")

print("Rainfall rebuild complete.")

# 4) Keep ONLY ML-ready assets
print("🧹 Cleaning folder (keep only *_40m_norm.tif + flood_labels_40m.tif + worldcover_40m.tif)...")
keepers = set()

# Always keep label
if os.path.exists(os.path.join(final_dir, "flood_labels_40m.tif")):
    keepers.add("flood_labels_40m.tif")

# Keep DEM norm (reference + feature)
if os.path.exists(os.path.join(final_dir, "dem_40m_norm.tif")):
    keepers.add("dem_40m_norm.tif")

# Keep population norm
if os.path.exists(os.path.join(final_dir, "lagos_pop_40m_norm.tif")):
    keepers.add("lagos_pop_40m_norm.tif")

# Keep all rainfall *_40m_norm.tif
for f in os.listdir(final_dir):
    if f.endswith("_40m_norm.tif"):
        keepers.add(f)

# Keep categorical worldcover (not normalized)
if os.path.exists(os.path.join(final_dir, "worldcover_40m.tif")):
    keepers.add("worldcover_40m.tif")

remove_everything_except(keepers)
print("Folder cleaned.")

# 5) Final verification
print("\nFinal verification...")
ref = os.path.join(final_dir, "dem_40m_norm.tif")
with rasterio.open(ref) as r:
    ref_shape, ref_res = r.shape, r.res

for f in sorted(os.listdir(final_dir)):
    if not f.endswith(".tif"): 
        continue
    with rasterio.open(os.path.join(final_dir, f)) as src:
        print(f"{f:35} | Shape OK: {src.shape == ref_shape} | Res OK: {src.res == ref_res}")

print("\nDone — rainfall saved, population fixed, only ML-ready files remain.")
