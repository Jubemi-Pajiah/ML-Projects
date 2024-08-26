import os
import geopandas as gpd
from Flood_Hotspots_Project.config import raw_dir, final_dir, target_crs, mask_path, mask_layer

# Ensure output directory exists
os.makedirs(final_dir, exist_ok=True)

print("Loading original Lagos boundary...")
gdf = gpd.read_file(mask_path, layer=mask_layer)
print(f"Original CRS: {gdf.crs}")

print(f"Reprojecting to target CRS: {target_crs} ...")
gdf_reproj = gdf.to_crs(target_crs)

out_path = os.path.join(final_dir, "lagos_boundary_reproj.gpkg")
gdf_reproj.to_file(out_path, driver="GPKG")
print(f"Saved reprojected boundary: {out_path}")
print(f"Reprojected CRS: {gdf_reproj.crs}")
