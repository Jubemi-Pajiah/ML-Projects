import os

base_dir = "Flood_Hotspots_Project"
raw_dir  = os.path.join(base_dir, "raw")
final_dir= os.path.join(base_dir, "final_40m")

output_path = os.path.join(base_dir, "outputs")

out_dir = os.path.join(base_dir, "resampled_40m")

mask_path  = os.path.join(raw_dir, "lagos-ward-floods.gpkg")
mask_layer = "lagoswards__nigeria_ward_shapefiles__nigeria__ward_boundaries"

norm_dir = "Flood_Hotspots_Project/normalized_40m"
resamp_dir = "Flood_Hotspots_Project/resampled_40m"

target_crs = "EPSG:32631"

output_dir = os.path.join(base_dir, "predictions/output")
model_path = os.path.join(base_dir, "outputs", "flood_rf_model.pkl")
