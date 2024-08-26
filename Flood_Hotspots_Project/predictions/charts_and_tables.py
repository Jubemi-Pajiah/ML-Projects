# Flood_Hotspots_Project/predictions/charts_and_tables.py
import os, re, joblib, numpy as np, pandas as pd, rasterio, geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.mask import mask
from Flood_Hotspots_Project.config import final_dir, output_dir, model_path

MONTH_RE = re.compile(r"(january|february|march|april|may|june|july|august|"
                      r"september|october|november|december)-\d{4}_40m_norm\.tif$", re.I)

def load_boundary():
    gdf = gpd.read_file(os.path.join(final_dir, "lagos_boundary_reproj.gpkg"))
    return [gdf.union_all()]

def stack(paths):
    arrs=[]; meta=None
    for p in paths:
        with rasterio.open(p) as src:
            a = src.read(1); arrs.append(a)
            if meta is None: meta=src.meta.copy()
    return np.stack(arrs), meta

def predict(model, st):
    B,R,C = st.shape
    X = st.reshape(B,-1).T
    return model.predict_proba(X)[:,1].reshape(R,C)

def list_features():
    files = os.listdir(final_dir)
    norm = sorted([os.path.join(final_dir,f) for f in files if f.endswith("_norm.tif")])
    rains = [f for f in norm if re.search(MONTH_RE, os.path.basename(f))]
    other = [f for f in norm if f not in rains]
    lc = os.path.join(final_dir,"worldcover_40m.tif")
    return other, rains, lc

if __name__=="__main__":
    os.makedirs(output_dir, exist_ok=True)
    print("Loading model...")
    model = joblib.load(model_path)

    other, rains, lc = list_features()
    geom = load_boundary()

    # Build baseline monthly series identical to monthly_flood_predictions.py
    medians=[]
    for rp in rains:
        with rasterio.open(rp) as s: medians.append(float(np.nanmedian(s.read(1))))

    rows=[]
    for j,rp in enumerate(rains):
        month = os.path.basename(rp).replace("_40m_norm.tif","")
        cont, meta = stack(other + rains)
        for k in range(len(rains)): cont[len(other)+k,:,:] = medians[k]
        with rasterio.open(rp) as s: cont[len(other)+j,:,:] = s.read(1).astype("float32")
        lc_arr,_ = stack([lc])
        full = np.concatenate([cont, lc_arr], axis=0)
        probs = predict(model, full)
        rows.append({"month": month, "mean_probability": float(np.nanmean(probs))})

    df = pd.DataFrame(rows)

    # Sort months chronologically
    def month_sort_key(mstr):
        m = mstr.split("-")[0].lower(); y = int(mstr.split("-")[1])
        order = ["january","february","march","april","may","june",
                 "july","august","september","october","november","december"]
        return (y, order.index(m) if m in order else 99)
    df = df.sort_values(by="month", key=lambda s: s.map(month_sort_key))

    csv_path = os.path.join(output_dir, "monthly_flood_probabilities.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved table -> {csv_path}")

    # Chart 1: monthly means
    png1 = os.path.join(output_dir, "monthly_flood_probabilities.png")
    plt.figure(figsize=(12,5))
    plt.plot(df["month"], df["mean_probability"], marker="o")
    plt.xticks(rotation=45, ha="right"); plt.ylabel("Mean Flood Probability")
    plt.title("Monthly Flood Probability over Lagos"); plt.tight_layout()
    plt.savefig(png1, dpi=200); plt.close()
    print(f"Saved chart -> {png1}")

    # Chart 2: rainfall sensitivity around the peak month
    top = df.loc[df["mean_probability"].idxmax(),"month"]
    top_idx = [i for i,rp in enumerate(rains) if os.path.basename(rp).startswith(top)][0]
    with rasterio.open(rains[top_idx]) as s:
        base_arr = s.read(1).astype("float32")
    cont, meta = stack(other + rains)
    for k in range(len(rains)): cont[len(other)+k,:,:] = medians[k]

    scales = [-0.5, -0.25, 0.0, 0.25, 0.5]
    sens=[]
    for sc in scales:
        cont_mod = cont.copy()
        cont_mod[len(other)+top_idx,:,:] = np.clip(base_arr*(1+sc), 0.0, 1.0)
        lc_arr,_ = stack([lc])
        full = np.concatenate([cont_mod, lc_arr], axis=0)
        probs = predict(model, full)
        sens.append({"scale_pct": int(sc*100), "mean_probability": float(np.nanmean(probs))})

    df_sens = pd.DataFrame(sens)
    png2 = os.path.join(output_dir, "rainfall_sensitivity.png")
    plt.figure(figsize=(6,4))
    plt.plot(df_sens["scale_pct"], df_sens["mean_probability"], marker="o")
    plt.xlabel(f"Rainfall change for {top} (%)")
    plt.ylabel("Mean Flood Probability")
    plt.title("Rainfall Sensitivity")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(png2, dpi=200); plt.close()
    print(f"Saved chart -> {png2}")

    print("Charts and table complete.")
