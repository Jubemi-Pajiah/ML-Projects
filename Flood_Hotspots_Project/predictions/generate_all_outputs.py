# Flood_Hotspots_Project/predictions/generate_all_outputs.py

import os
import re
import joblib
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from rasterio.mask import mask
from Flood_Hotspots_Project.config import base_dir, final_dir, output_dir, model_path

# =========================
# CONFIG & SETUP
# =========================
os.makedirs(output_dir, exist_ok=True)
boundary_path = os.path.join(final_dir, "lagos_boundary_reproj.gpkg")

# Color ramps
RAMP_GYR = ["#006837", "#ffff00", "#bd0026"]  # green -> yellow -> red
GYR_CMAP = mcolors.LinearSegmentedColormap.from_list("gyr", RAMP_GYR, N=256)

# Risk class thresholds
LOW_THR = 0.33
HIGH_THR = 0.66

# Regex to detect monthly rainfall rasters from your final_40m directory
# Matches names like "march-2020_40m_norm.tif" or "september-2021_40m_norm.tif"
MONTH_RE = re.compile(r"(january|february|march|april|may|june|july|august|september|october|november|december)-\d{4}_40m_norm\.tif$", re.I)


# =========================
# UTILITIES
# =========================
def load_boundary_geom(boundary_gpkg):
    gdf = gpd.read_file(boundary_gpkg)
    geom = [gdf.union_all()]
    return geom

def stack_features(feature_files):
    """Load rasters into a 3D stack [bands, rows, cols]; return stack & reference meta."""
    stack = []
    meta_ref = None
    for i, path in enumerate(feature_files, start=1):
        with rasterio.open(path) as src:
            arr = src.read(1)
            stack.append(arr)
            if meta_ref is None:
                meta_ref = src.meta.copy()
        print(f"  Loaded: {os.path.basename(path)} ({i}/{len(feature_files)})")
    stack = np.stack(stack)  # [B, R, C]
    return stack, meta_ref

def predict_map(model, stack):
    """Predict probability map for a given feature stack."""
    B, R, C = stack.shape
    X = stack.reshape(B, -1).T
    probs = model.predict_proba(X)[:, 1].reshape(R, C)
    return probs

def write_clipped_rgb(output_path, rgb_data, meta_like, boundary_geom):
    """Clip an RGB array [3, R, C] to Lagos boundary and save one colored TIFF."""
    # Write a temporary full-extent RGB to clip from
    temp_path = output_path.replace(".tif", "_temp.tif")
    meta = meta_like.copy()
    meta.update({"count": 3, "dtype": "uint8", "nodata": 0})
    with rasterio.open(temp_path, "w", **meta) as dst:
        dst.write(rgb_data)

    with rasterio.open(temp_path) as src:
        clipped, transform = mask(src, boundary_geom, crop=True)
        meta_clip = src.meta.copy()
        meta_clip.update({
            "height": clipped.shape[1],
            "width": clipped.shape[2],
            "transform": transform
        })
        with rasterio.open(output_path, "w", **meta_clip) as dst:
            dst.write(clipped)

    os.remove(temp_path)

def probs_to_rgb(probs, cmap):
    """Map float32 probs [R, C] to RGB uint8 [3, R, C]."""
    norm = mcolors.Normalize(vmin=0, vmax=1)
    rgba = cmap(norm(probs))
    rgb = np.moveaxis((rgba[:, :, :3] * 255).astype(np.uint8), 2, 0)
    return rgb

def classify_probs(probs, low_thr=LOW_THR, high_thr=HIGH_THR):
    """Return class map [R, C]: 0=Low, 1=Moderate, 2=High."""
    cls = np.zeros_like(probs, dtype=np.uint8)
    cls[(probs >= low_thr) & (probs < high_thr)] = 1
    cls[probs >= high_thr] = 2
    return cls

def classes_to_rgb(classes):
    """Map 0/1/2 to green/yellow/red RGB uint8 [3, R, C]."""
    color_map = {
        0: (0, 104, 55),     # green
        1: (255, 255, 0),    # yellow
        2: (189, 0, 38)      # red
    }
    R, C = classes.shape
    rgb = np.zeros((3, R, C), dtype=np.uint8)
    for k, (r, g, b) in color_map.items():
        mask_k = (classes == k)
        rgb[0][mask_k] = r
        rgb[1][mask_k] = g
        rgb[2][mask_k] = b
    return rgb

def find_feature_files():
    """Return lists of paths for continuous features and the categorical raster."""
    files = os.listdir(final_dir)
    norm_files = sorted([f for f in files if f.endswith("_norm.tif")])

    # Partition into rainfall vs other continuous features
    rainfall_files = [os.path.join(final_dir, f) for f in norm_files if re.search(MONTH_RE, f)]
    other_continuous = [os.path.join(final_dir, f) for f in norm_files if not re.search(MONTH_RE, f)]

    # Categorical landcover
    landcover_path = os.path.join(final_dir, "worldcover_40m.tif")
    if not os.path.exists(landcover_path):
        raise FileNotFoundError("Missing categorical raster: worldcover_40m.tif")

    # Full feature order = [all continuous] + [categorical at the end]
    # Keep a consistent order for prediction
    continuous = other_continuous + rainfall_files
    feature_files = continuous + [landcover_path]

    return feature_files, rainfall_files, other_continuous, landcover_path

def index_groups(continuous, rainfall_files):
    """Return indices for DEM, POP and rainfall bands within the continuous stack."""
    # We expect dem/pop files present in 'continuous'
    dem_idx = None
    pop_idx = None
    rain_idxs = []

    for i, path in enumerate(continuous):
        name = os.path.basename(path).lower()
        if name.startswith("dem_") and name.endswith("_norm.tif"):
            dem_idx = i
        if name.startswith("lagos_pop_") and name.endswith("_norm.tif"):
            pop_idx = i
        if path in rainfall_files:
            rain_idxs.append(i)

    return dem_idx, pop_idx, rain_idxs


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("Loading trained model...")
    model = joblib.load(model_path)
    print("Model loaded.")

    print("Loading Lagos boundary...")
    geom = load_boundary_geom(boundary_path)
    print("Boundary loaded.\n")

    # Discover features
    print("Discovering feature rasters...")
    feature_files, rainfall_files, other_continuous, landcover_path = find_feature_files()
    print(f"  Found {len(feature_files)-1} continuous features and landcover categorical.")
    print(f"  Rainfall rasters detected: {len(rainfall_files)}\n")

    # Build full stack for BASE prediction
    print("Stacking feature rasters for base prediction...")
    stack, meta_ref = stack_features(feature_files)
    B, R, C = stack.shape
    print(f"Stack shape: {stack.shape} [bands, rows, cols]\n")

    # Predict BASE flood probabilities
    print("Predicting base flood probabilities...")
    base_probs = predict_map(model, stack)
    print("Base prediction complete.")

    # Save colorized, clipped probability map
    prob_colored_path = os.path.join(output_dir, "flood_risk_prob_colored.tif")
    print("Saving colorized, clipped base probability map...")
    rgb = probs_to_rgb(base_probs, GYR_CMAP)
    write_clipped_rgb(prob_colored_path, rgb, meta_ref, geom)
    print(f"  -> {prob_colored_path}\n")

    # Save colorized, clipped CLASS map (Low/Moderate/High)
    print("Creating and saving classified risk map...")
    classes = classify_probs(base_probs, LOW_THR, HIGH_THR)
    class_rgb = classes_to_rgb(classes)
    class_path = os.path.join(output_dir, "flood_risk_class_colored.tif")
    write_clipped_rgb(class_path, class_rgb, meta_ref, geom)
    print(f"  -> {class_path}\n")

    # =========================
    # MONTHLY MAPS + TABLE + CHARTS
    # =========================
    # Strategy:
    #  - For each monthly rainfall raster, create a new feature stack where:
    #      * DEM & POP & other continuous features stay as-is from final_40m
    #      * All rainfall features are set to their global median EXCEPT the target month,
    #        which keeps its actual raster values.
    #  - Predict, colorize, clip, and save that month's map

    print("Preparing monthly rainfall-only prediction series...")
    continuous = other_continuous + rainfall_files
    landcover = landcover_path
    dem_idx, pop_idx, rain_idxs = index_groups(continuous, rainfall_files)

    # Precompute medians for each rainfall band (global, over all pixels)
    rainfall_medians = []
    print("Computing rainfall band medians for non-target months...")
    for i, rf in enumerate(rainfall_files):
        with rasterio.open(rf) as src:
            arr = src.read(1).astype("float32")
            med = float(np.nanmedian(arr))
            rainfall_medians.append(med)
    print("Rainfall medians computed.\n")

    monthly_summary = []  # for CSV & chart

    print("Generating monthly rainfall prediction maps...")
    for j, rf in enumerate(rainfall_files):
        month_name = os.path.basename(rf).replace("_40m_norm.tif", "")  # e.g., "march-2021"

        # Build modified continuous stack: copy base continuous
        cont_stack, _ = stack_features(continuous)
        # Set all rainfall bands to their medians
        for k, ridx in enumerate(rain_idxs):
            cont_stack[ridx, :, :] = rainfall_medians[k]
        # Put the target month back to its real raster values
        with rasterio.open(rf) as src:
            target_arr = src.read(1).astype("float32")
        cont_stack[rain_idxs[j], :, :] = target_arr

        # Full stack = modified continuous + landcover (categorical)
        landcover_arr, _ = stack_features([landcover])
        full_stack = np.concatenate([cont_stack, landcover_arr], axis=0)

        # Predict
        probs = predict_map(model, full_stack)

        # Save colorized, clipped TIFF
        out_month_path = os.path.join(output_dir, f"flood_pred_{month_name}.tif")
        rgb_month = probs_to_rgb(probs, GYR_CMAP)
        write_clipped_rgb(out_month_path, rgb_month, meta_ref, geom)
        print(f"  -> {out_month_path}")

        # Collect summary statistics
        mean_prob = float(np.nanmean(probs))
        monthly_summary.append({"month": month_name, "mean_probability": mean_prob})

    print("\nMonthly prediction series complete.\n")

    # Save CSV
    csv_path = os.path.join(output_dir, "monthly_flood_probabilities.csv")
    df_month = pd.DataFrame(monthly_summary)
    # Sort months in chronological order if possible using a custom key:
    def month_sort_key(mstr):
        m = mstr.split("-")[0].lower()
        y = int(mstr.split("-")[1])
        order = ["january","february","march","april","may","june",
                 "july","august","september","october","november","december"]
        return (y, order.index(m) if m in order else 99)
    df_month = df_month.sort_values(by="month", key=lambda s: s.map(month_sort_key))
    df_month.to_csv(csv_path, index=False)
    print(f"Saved monthly summary table: {csv_path}")

    # Monthly chart
    png_month = os.path.join(output_dir, "monthly_flood_probabilities.png")
    plt.figure(figsize=(12, 5))
    plt.plot(df_month["month"], df_month["mean_probability"], marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Flood Probability")
    plt.title("Monthly Flood Probability over Lagos")
    plt.tight_layout()
    plt.savefig(png_month, dpi=200)
    plt.close()
    print(f"Saved monthly chart: {png_month}")

    # Rainfall sensitivity: pick month with highest baseline mean
    top_row = df_month.loc[df_month["mean_probability"].idxmax()]
    top_month = top_row["month"]
    top_idx = [i for i, rf in enumerate(rainfall_files)
               if os.path.basename(rf).startswith(top_month)][0]
    top_rain_file = rainfall_files[top_idx]
    print(f"\nRunning rainfall sensitivity on {top_month} ...")

    # Build baseline stack with rainfall medians (as above)
    base_cont_stack, _ = stack_features(continuous)
    for k, ridx in enumerate(rain_idxs):
        base_cont_stack[ridx, :, :] = rainfall_medians[k]

    with rasterio.open(top_rain_file) as src:
        base_top_arr = src.read(1).astype("float32")

    scales = [-0.5, -0.25, 0.0, 0.25, 0.5]  # -50% to +50%
    sens_results = []
    for sc in scales:
        cont_mod = base_cont_stack.copy()
        mod_arr = base_top_arr * (1.0 + sc)
        # clamp to [0,1] since rasters are normalized
        mod_arr = np.clip(mod_arr, 0.0, 1.0)
        cont_mod[rain_idxs[top_idx], :, :] = mod_arr

        landcover_arr, _ = stack_features([landcover])
        full_stack = np.concatenate([cont_mod, landcover_arr], axis=0)
        probs = predict_map(model, full_stack)
        sens_results.append({"scale": sc, "mean_probability": float(np.nanmean(probs))})

    # Sensitivity chart
    df_sens = pd.DataFrame(sens_results)
    png_sens = os.path.join(output_dir, "rainfall_sensitivity.png")
    plt.figure(figsize=(6, 4))
    plt.plot((df_sens["scale"] * 100.0), df_sens["mean_probability"], marker="o")
    plt.xlabel(f"Rainfall change for {top_month} (%)")
    plt.ylabel("Mean Flood Probability")
    plt.title("Rainfall Sensitivity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_sens, dpi=200)
    plt.close()
    print(f"Saved rainfall sensitivity chart: {png_sens}\n")

    print("All outputs generated successfully.")
