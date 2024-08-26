import os
import joblib
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import matplotlib.colors as mcolors
from Flood_Hotspots_Project.config import base_dir, final_dir, output_dir, model_path

# SETUP
os.makedirs(output_dir, exist_ok=True)
boundary_path = os.path.join(final_dir, "lagos_boundary_reproj.gpkg")

# LOAD MODEL
print("Loading trained Random Forest model...")
model = joblib.load(model_path)

# LOAD RASTERS

norm_rasters = sorted([f for f in os.listdir(final_dir) if f.endswith("_norm.tif")])
cat_raster = os.path.join(final_dir, "worldcover_40m.tif")
if not os.path.exists(cat_raster):
    raise FileNotFoundError("Missing categorical raster: worldcover_40m.tif")

raster_files = norm_rasters + ["worldcover_40m.tif"]

stack_data = []
meta_ref = None
for i, fname in enumerate(raster_files, start=1):
    path = os.path.join(final_dir, fname)
    with rasterio.open(path) as src:
        data = src.read(1)
        stack_data.append(data)
        if meta_ref is None:
            meta_ref = src.meta.copy()
    print(f"  {i:02d}/{len(raster_files)} Loaded: {fname}")

stack = np.stack(stack_data)
rows, cols = stack.shape[1:]

# PREDICT
X = stack.reshape(len(raster_files), -1).T
y_prob = model.predict_proba(X)[:, 1].reshape(rows, cols)
print("Prediction complete.\n")

# SAVE TEMP FLOAT MAP
temp_path = os.path.join(output_dir, "flood_risk_prob_temp.tif")
meta = meta_ref.copy()
meta.update({"count": 1, "dtype": "float32", "nodata": np.nan})
with rasterio.open(temp_path, "w", **meta) as dst:
    dst.write(y_prob.astype("float32"), 1)

# MASK & COLORIZE
print("Clipping and colorizing final output...")
gdf = gpd.read_file(boundary_path)
geom = [gdf.union_all()]

with rasterio.open(temp_path) as src:
    clipped, transform = mask(src, geom, crop=True)
    meta_clip = src.meta.copy()
    meta_clip.update({
        "height": clipped.shape[1],
        "width": clipped.shape[2],
        "transform": transform
    })

# color map: Green → Yellow → Red
norm = mcolors.Normalize(vmin=0, vmax=1)
colors = ["#006837", "#ffff00", "#bd0026"]  # green → yellow → red
cmap = mcolors.LinearSegmentedColormap.from_list("flood_risk", colors, N=256)

rgba = cmap(norm(clipped.squeeze()))
rgb = np.moveaxis((rgba[:, :, :3] * 255).astype(np.uint8), 2, 0)

# Save only colored map
final_path = os.path.join(output_dir, "flood_risk_prob_colored.tif")
meta_color = meta_clip.copy()
meta_color.update({"count": 3, "dtype": "uint8", "nodata": 0})

with rasterio.open(final_path, "w", **meta_color) as dst:
    dst.write(rgb)

# Clean up temporary file
os.remove(temp_path)

print(f"Final colorized flood probability map saved: {final_path}")
print("Done.")
