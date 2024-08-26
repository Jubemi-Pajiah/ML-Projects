import os
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

# =============================
# CONFIGURATION
# =============================
BASE_DIR = "Flood_Hotspots_Project"
DATA_PATH = os.path.join(BASE_DIR, "outputs", "lagos_flood_dataset.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(OUT_DIR, "flood_rf_model.pkl")
LOG_PATH = os.path.join(OUT_DIR, "training_log.txt")

os.makedirs(OUT_DIR, exist_ok=True)

# =============================
# LOGGING SETUP
# =============================
class Logger(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(LOG_PATH)

# =============================
# LOAD DATA
# =============================
print("Loading Lagos flood dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset — {len(df):,} rows × {len(df.columns)} columns")

if "flood_label" not in df.columns:
    raise ValueError("Missing 'flood_label' column in dataset!")

# =============================
# REMOVE EMPTY (ALL-ZERO) ROWS
# =============================
print("Checking for empty rows (all-zero features)...")
non_label_cols = [c for c in df.columns if c != "flood_label"]
before = len(df)
df = df.loc[~(df[non_label_cols].sum(axis=1) == 0)]
after = len(df)
removed = before - after
print(f"Removed {removed:,} empty rows — {after:,} rows remaining.\n")

# =============================
# TRAIN / TEST SPLIT
# =============================
print("Splitting dataset into train/test sets...")
X = df.drop(columns=["flood_label"])
y = df["flood_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}\n")

# =============================
# TRAIN RANDOM FOREST (per-tree progress)
# =============================
n_trees = 200
print(f"Training Random Forest model with {n_trees} trees...")

rf = RandomForestClassifier(
    n_estimators=1,           # start with 1 tree
    max_depth=15,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    warm_start=True           # allows adding trees incrementally
)

for i in tqdm(range(1, n_trees + 1), desc="Training Progress", ncols=80):
    rf.n_estimators = i
    rf.fit(X_train, y_train)

print("Model training complete.\n")

# =============================
# EVALUATE PERFORMANCE
# =============================
print("Evaluating model on test set...")
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {round(acc * 100, 2)}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()

# =============================
# SAVE MODEL
# =============================
print(f"Saving trained model → {MODEL_PATH}")
joblib.dump(rf, MODEL_PATH)
print("Model saved successfully.\n")

# =============================
# FEATURE IMPORTANCES
# =============================
print("Top 10 Most Important Features:")
importances = pd.Series(rf.feature_importances_, index=X.columns)
top10 = importances.sort_values(ascending=False).head(10)
print(top10)

print("\nTraining complete — model ready for flood map generation.")
