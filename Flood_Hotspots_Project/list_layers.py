import pandas as pd

# Load your dataset
df = pd.read_csv("Flood_Hotspots_Project/outputs/lagos_flood_dataset.csv")

# Define feature columns (exclude label)
feature_cols = [c for c in df.columns if c != "flood_label"]

# Identify empty vs non-empty rows
empty_mask = (df[feature_cols] == 0).all(axis=1)
num_empty = empty_mask.sum()
num_non_empty = len(df) - num_empty

# Print results
print(f"🧩 Total rows: {len(df):,}")
print(f"⚪ Empty rows (all-zero features): {num_empty:,}")
print(f"🟢 Non-empty rows (with valid data): {num_non_empty:,}")
print(f"📉 Percentage empty: {num_empty / len(df) * 100:.2f}%")
