# Flood_Hotspots_Project/predictions/monthly_flood_predictions.py
import os, re, joblib, numpy as np, rasterio, geopandas as gpd
import matplotlib.colors as mcolors
from rasterio.mask import mask
from Flood_Hotspots_Project.config import final_dir, output_dir, model_path

RAMP_GYR = ["#006837", "#ffff00", "#bd0026"]
GYR_CMAP = mcolors.LinearSegmentedColormap.from_list("gyr", RAMP_GYR, N=256)
MONTH_RE = re.compile(r"(january|february|march|april|may|june|july|august|"
                      r"september|october|november|december)-\d{4}_40m_norm\.tif$", re.I)

def load_boundary():
    bpath = os.path.join(final_dir, "lagos_boundary_reproj.gpkg")
    gdf = gpd.read_file(bpath)
    return [gdf.union_all()]

def list_features():
    files = os.listdir(final_dir)
    norm = sorted([os.path.join(final_dir,f) for f in files if f.endswith("_norm.tif")])
    rains = [f for f in norm if re.search(MONTH_RE, os.path.basename(f))]
    other = [f for f in norm if f not in rains]
    lc = os.path.join(final_dir, "worldcover_40m.tif")
    if not os.path.exists(lc): raise FileNotFoundError("worldcover_40m.tif missing")
    return other, rains, lc

def stack(paths):
    arrs=[]; meta=None
    for p in paths:
        with rasterio.open(p) as src:
            a = src.read(1)
            arrs.append(a)
            if meta is None: meta = src.meta.copy()
    return np.stack(arrs), meta

def predict(model, st):
    B,R,C = st.shape
    X = st.reshape(B,-1).T
    return model.predict_proba(X)[:,1].reshape(R,C)

def to_rgb(probs):
    norm = mcolors.Normalize(vmin=0, vmax=1)
    rgba = GYR_CMAP(norm(probs))
    return np.moveaxis((rgba[:,:,:3]*255).astype(np.uint8),2,0)

def write_clipped_rgb(out_path, rgb, meta_like, geom):
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

    other, rains, lc = list_features()
    print(f"Found {len(rains)} rainfall rasters.")

    # Pre-median rainfall bands for non-target months
    medians=[]
    for rp in rains:
        with rasterio.open(rp) as s:
            medians.append(float(np.nanmedian(s.read(1))))

    for j, rp in enumerate(rains):
        month_name = os.path.basename(rp).replace("_40m_norm.tif","")
        print(f"Rendering {month_name}...")

        cont, meta = stack(other + rains)
        # set all rainfall to median
        for k in range(len(rains)):
            cont[len(other)+k,:,:] = medians[k]
        # restore target month
        with rasterio.open(rp) as s: cont[len(other)+j,:,:] = s.read(1).astype("float32")

        lc_arr, _ = stack([lc])
        full = np.concatenate([cont, lc_arr], axis=0)
        probs = predict(model, full)
        rgb = to_rgb(probs)
        outp = os.path.join(output_dir, f"flood_pred_{month_name}.tif")
        write_clipped_rgb(outp, rgb, meta, geom)
        print(f"Saved -> {outp}")

    print("Monthly predictions complete.")
