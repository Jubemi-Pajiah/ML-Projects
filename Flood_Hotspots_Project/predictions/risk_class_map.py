# Flood_Hotspots_Project/predictions/risk_class_map.py
import os, re, joblib, numpy as np, rasterio, geopandas as gpd
import matplotlib.colors as mcolors
from rasterio.mask import mask
from Flood_Hotspots_Project.config import final_dir, output_dir, model_path

LOW_THR, HIGH_THR = 0.33, 0.66
RAMP_GYR = ["#006837", "#ffff00", "#bd0026"]  # green->yellow->red
GYR_CMAP = mcolors.LinearSegmentedColormap.from_list("gyr", RAMP_GYR, N=256)
MONTH_RE = re.compile(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
                      r"january|february|march|april|may|june|july|august|"
                      r"september|october|november|december)-\d{4}_40m_norm\.tif$", re.I)

def load_boundary():
    bpath = os.path.join(final_dir, "lagos_boundary_reproj.gpkg")
    gdf = gpd.read_file(bpath)
    return [gdf.union_all()]

def list_features():
    files = os.listdir(final_dir)
    cont = sorted([os.path.join(final_dir,f) for f in files if f.endswith("_norm.tif")])
    rainfall = [f for f in cont if re.search(MONTH_RE, os.path.basename(f))]
    other = [f for f in cont if f not in rainfall]
    landcover = os.path.join(final_dir, "worldcover_40m.tif")
    if not os.path.exists(landcover):
        raise FileNotFoundError("worldcover_40m.tif not found")
    feats = other + rainfall + [landcover]
    return feats, landcover

def stack(files):
    arrs = []
    meta = None
    for p in files:
        with rasterio.open(p) as src:
            a = src.read(1)
            arrs.append(a)
            if meta is None: meta = src.meta.copy()
    return np.stack(arrs), meta

def predict(model, stack):
    B,R,C = stack.shape
    X = stack.reshape(B,-1).T
    prob = model.predict_proba(X)[:,1].reshape(R,C)
    return prob

def classes_to_rgb(classes):
    cmap = {0:(0,104,55), 1:(255,255,0), 2:(189,0,38)}
    R,C = classes.shape
    rgb = np.zeros((3,R,C), dtype=np.uint8)
    for k,(r,g,b) in cmap.items():
        m = (classes==k)
        rgb[0][m]=r; rgb[1][m]=g; rgb[2][m]=b
    return rgb

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
    print("Model loaded.")
    geom = load_boundary()
    feats, _ = list_features()
    print("Stacking features...")
    st, meta = stack(feats)
    print("Predicting base probabilities...")
    probs = predict(model, st)
    print("Classifying...")
    cls = np.zeros_like(probs, dtype=np.uint8)
    cls[(probs>=LOW_THR)&(probs<HIGH_THR)] = 1
    cls[probs>=HIGH_THR] = 2
    rgb = classes_to_rgb(cls)
    out = os.path.join(output_dir, "flood_risk_class_colored.tif")
    print("Saving classified risk map...")
    write_clipped_rgb(out, rgb, meta, geom)
    print(f"Done -> {out}")
