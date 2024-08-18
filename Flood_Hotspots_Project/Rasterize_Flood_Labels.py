import os
import rasterio
import geopandas as gpd
import numpy as np
from rasterio import features

#CONFIG
base_dir = "Flood_Hotspots_Project"
raw_dir = os.path.join(base_dir, "raw")
final_dir = os.path.join(base_dir, "final_40m")
os.makedirs(final_dir, exist_ok=True)

mask_path = os.path.join(raw_dir, "lagos-ward-floods.gpkg")
mask_layer = "lagoswards__nigeria_ward_shapefiles__nigeria__ward_boundaries"

label_field = "wards_with_floods_flooded"

ref_raster = os.path.join(final_dir, "dem_40m_norm.tif")

# Output label file
label_raster_path = os.path.join(final_dir, "flood_labels_40m.tif")

target_crs = "EPSG:32631"

#STEP 1: LOAD AND REPROJECT
print("📍 Loading Lagos flood polygons...")
gdf = gpd.read_file(mask_path, layer=mask_layer)

if label_field not in gdf.columns:
    print(f"❌ '{label_field}' not found in shapefile! Available columns:")
    print(gdf.columns.tolist())
    raise SystemExit

# Convert classification values to numeric (0 or 1)
gdf[label_field] = gdf[label_field].astype(str).str.lower().map(
    {"yes": 1, "true": 1, "1": 1, "flooded": 1}
).fillna(0).astype("uint8")

# Reproject shapefile to match DEM CRS
gdf = gdf.to_crs(target_crs)
print("Flood polygons reprojected to EPSG:32631")

#STEP 2: RASTERIZE
print("Rasterizing flood polygons to align with DEM grid...")

# Use DEM grid for alignment
with rasterio.open(ref_raster) as ref:
    meta = ref.meta.copy()
    transform = ref.transform
    out_shape = (ref.height, ref.width)

# Rasterize polygons
shapes = [(geom, val) for geom, val in zip(gdf.geometry, gdf[label_field])]
label_data = features.rasterize(
    shapes=shapes,
    out_shape=out_shape,
    transform=transform,
    fill=0,
    all_touched=True,
    dtype="uint8"
)

meta.update({
    "driver": "GTiff",
    "count": 1,
    "dtype": "uint8",
    "nodata": 0
})

#STEP 3: SAVE
tmp_out = label_raster_path + ".tmp"
with rasterio.open(tmp_out, "w", **meta) as dst:
    dst.write(label_data, 1)
os.replace(tmp_out, label_raster_path)

print(f"Flood labels rasterized and saved at: {label_raster_path}")

#STEP 4: VERIFY
with rasterio.open(ref_raster) as ref, rasterio.open(label_raster_path) as lbl:
    print("\nVerification:")
    print(f"DEM shape: {ref.shape}, resolution: {ref.res}")
    print(f"Label shape: {lbl.shape}, resolution: {lbl.res}")
    print(f"Shape OK: {lbl.shape == ref.shape}, Res OK: {lbl.res == ref.res}")

print("\nFlood label raster successfully reprojected, classified, and aligned.")
