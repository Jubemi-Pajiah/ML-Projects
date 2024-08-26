# Flood_Hotspots_Project/predictions/influence_maps.py
import os, re, joblib, numpy as np, rasterio, geopandas as gpd
import matplotlib.colors as mcolors
from rasterio.mask import mask
from Flood_Hotspots_Project.config import final_dir, output_dir, model_path

MONTH_RE = re.compile(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
                      r"january|february|march|april|may|june|july|august|"
                      r"september|october|november|december)-\d{4}_40m_norm\.tif$", re.I)
RAMP_BR = ["#08306b","#4292c6","#ffffb2","#bd0026"]  # blue->red
BR_CMAP = mcolors.LinearSegmentedColormap.from_list("br", RAMP_BR, N=256)

def load_boundary():
    path = os.path.join(final_dir, "lagos_boundary_reproj.gpkg")
    gdf = gpd.read_file(path)
    return [gdf.union_all()]

def list_features():
    files = os.listdir(final_dir)
    norm = sorted([os.path.join(final_dir,f) for f in files if f.endswith("_norm.tif")])
    rainfall = [f for f in norm if re.search(MONTH_RE, os.path.basename(f))]
    other = [f for f in norm if f not in rainfall]
    lc = os.path.join(final_dir,"worldcover_40m.tif")
    if not os.path.exists(lc): raise FileNotFoundError("worldcover_40m.tif missing")
    return other, rainfall, lc

def stack(files):
    arrs=[]; meta=None
    for p in files:
        with rasterio.open(p) as src:
            a = src.read(1)
            arrs.append(a)
            if meta is None: meta=src.meta.copy()
    return np.stack(arrs), meta

def predict(model, st):
    B,R,C = st.shape
    X = st.reshape(B,-1).T
    return model.predict_proba(X)[:,1].reshape(R,C)

def write_clipped_rgb(out_path, arr, meta_like, geom, cmap):
    # map arr (float) to RGB
    norm = mcolors.Normalize(vmin=np.nanmin(arr), vmax=np.nanmax(arr))
    rgba = cmap(norm(arr))
    rgb  = np.moveaxis((rgba[:,:,:3]*255).astype(np.uint8),2,0)
    tmp = out_path.replace(".tif","_tmp.tif")
    meta = meta_like.copy(); meta.update({"count":3,"dtype":"uint8","nodata":0})
    with rasterio.open(tmp,"w",**meta) as dst: dst.write(rgb)
    with rasterio.open(tmp) as src:
        clipped, transform = mask(src, geom, crop=True)
        m = src.meta.copy()
        m.update({"height":clipped.shape[1], "width":clipped.shape[2], "transform":transform})
        with rasterio.open(out_path,"w",**m) as dst: dst.write(clipped)
    os.remove(tmp)

if __name__=="__main__":
    os.makedirs(output_dir, exist_ok=True)
    print("Loading model...")
    model = joblib.load(model_path)
    geom = load_boundary()

    other, rainfall, landcover = list_features()
    continuous = other + rainfall  # all continuous
    full_list  = continuous + [landcover]
    print(f"Continuous features: {len(continuous)}, rainfall bands: {len(rainfall)}")

    # Base stack
    print("Building base stack and predicting...")
    base_stack, meta = stack(full_list)
    base_probs = predict(model, base_stack)

    # Prepare typical values
    print("Computing typical (median/mode) values...")
    # DEM
    dem_idx = next(i for i,p in enumerate(continuous) if os.path.basename(p).startswith("dem_"))
    with rasterio.open(continuous[dem_idx]) as s: dem_med = float(np.nanmedian(s.read(1)))
    # POP
    pop_idx = next(i for i,p in enumerate(continuous) if os.path.basename(p).startswith("lagos_pop_"))
    with rasterio.open(continuous[pop_idx]) as s: pop_med = float(np.nanmedian(s.read(1)))
    # # RAINFALL medians per band
    # rain_indices = [i for i,p in enumerate(continuous) if p in rainfall]
    # rain_meds = []
    # for rp in rainfall:
    #     with rasterio.open(rp) as s: rain_meds.append(float(np.nanmedian(s.read(1))))
    # LANDCOVER mode
    with rasterio.open(landcover) as s:
        lc_arr = s.read(1).astype(np.int32)
    vals, counts = np.unique(lc_arr[lc_arr!=0], return_counts=True)
    lc_mode = int(vals[np.argmax(counts)]) if len(vals)>0 else 0

    # Influence DEM
    print("DEM influence...")
    dem_stack = base_stack.copy()
    dem_stack[dem_idx,:,:] = dem_med
    dem_probs = predict(model, dem_stack)
    infl_dem = base_probs - dem_probs
    write_clipped_rgb(os.path.join(output_dir,"influence_dem.tif"), infl_dem, meta, geom, BR_CMAP)

    # Influence Population
    print("Population influence...")
    pop_stack = base_stack.copy()
    pop_stack[pop_idx,:,:] = pop_med
    pop_probs = predict(model, pop_stack)
    infl_pop = base_probs - pop_probs
    write_clipped_rgb(os.path.join(output_dir,"influence_population.tif"), infl_pop, meta, geom, BR_CMAP)

    # # Influence Rainfall (replace all rainfall bands with their medians)
    # print("Rainfall influence...")
    # rain_stack = base_stack.copy()
    # for k,idx in enumerate(rain_indices):
    #     rain_stack[idx,:,:] = rain_meds[k]
    # rain_probs = predict(model, rain_stack)
    # infl_rain = base_probs - rain_probs
    # write_clipped_rgb(os.path.join(output_dir,"influence_rainfall.tif"), infl_rain, meta, geom, BR_CMAP)

    # Influence Landcover (set all pixels to modal class)
    print("Landcover influence...")
    lc_stack = base_stack.copy()
    lc_stack[-1,:,:] = lc_mode
    lc_probs = predict(model, lc_stack)
    infl_lc = base_probs - lc_probs
    write_clipped_rgb(os.path.join(output_dir,"influence_landcover.tif"), infl_lc, meta, geom, BR_CMAP)

    print("Influence maps saved.")
