import os, json, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks

from config import (
    DATASET_DIR, IMG_SIZE, BATCH_SIZE, MIN_IMAGES_PER_CLASS,
    ALPHA, BETA, EPOCHS, UNFREEZE_LAST,
    V2_MODELS, V2_LOGS
)
from Step3_DataLoader import load_datasets

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

print("V2: starting training")

# ===== Load datasets =====
train_ds, val_ds, test_ds, valid_classes = load_datasets(
    dataset_dir=DATASET_DIR,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    min_images=MIN_IMAGES_PER_CLASS
)
print(f"Datasets ready | classes: {len(valid_classes)}")

# ===== Rock → Type mapping =====
labels_df = pd.read_csv("Rock_Identification_Poject/Input_Resouces/Rock_Labels.csv")
labels_df = labels_df[labels_df["Rock"].isin(valid_classes)]
type_names = sorted(labels_df["Type"].unique())
type_to_idx = {t: i for i, t in enumerate(type_names)}

rock_to_type_idx = [type_to_idx[labels_df[labels_df["Rock"] == rc]["Type"].iloc[0]] for rc in valid_classes]
rock_to_type_idx_tf = tf.constant(rock_to_type_idx, dtype=tf.int32)
print(f"Rock families: {type_names}")

def add_type_labels(x, y_rock):
    rock_idx = tf.argmax(y_rock, axis=1, output_type=tf.int32)
    type_idx = tf.gather(rock_to_type_idx_tf, rock_idx)
    y_type = tf.one_hot(type_idx, depth=len(type_names), dtype=tf.float32)
    return x, {"type_output": y_type, "rock_output": y_rock}

train_ds = train_ds.map(add_type_labels, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.map(add_type_labels,   num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.map(add_type_labels,  num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

# ===== Build model =====
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.3)(x)

type_output = layers.Dense(len(type_names), activation="softmax", name="type_output")(x)
rock_output = layers.Dense(len(valid_classes), activation="softmax", name="rock_output")(x)
model = Model(inputs=base_model.input, outputs=[type_output, rock_output])

metrics = {
    "type_output": ["accuracy"],
    "rock_output": ["accuracy", tf.keras.metrics.TopKCategororicalAccuracy(k=3, name="top3_acc") if hasattr(tf.keras.metrics, "TopKCategororicalAccuracy") else tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")]
}
# Handle potential TF metric name differences gracefully
metrics["rock_output"] = ["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")]

losses = {"type_output": "categorical_crossentropy", "rock_output": "categorical_crossentropy"}
loss_weights = {"type_output": ALPHA, "rock_output": BETA}

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=losses, loss_weights=loss_weights, metrics=metrics)
print("Model compiled")

# ===== Callbacks =====
best_ckpt = V2_MODELS / "best_multitask_v2.keras"
cbs = [
    callbacks.EarlyStopping(monitor="val_rock_output_top3_acc", patience=5, mode="max", restore_best_weights=True),
    callbacks.ModelCheckpoint(best_ckpt.as_posix(), monitor="val_rock_output_top3_acc", mode="max", save_best_only=True),
    callbacks.ReduceLROnPlateau(monitor="val_rock_output_top3_acc", mode="max", factor=0.5, patience=2, verbose=1),
]

# ===== Train (frozen) =====
print("Phase 1: training with frozen backbone")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbs)

# ===== Fine-tune =====
for layer in base_model.layers[-UNFREEZE_LAST:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=losses, loss_weights=loss_weights, metrics=metrics)
print(f"Phase 2: fine-tuning last {UNFREEZE_LAST} layers")
history_ft = model.fit(train_ds, validation_data=val_ds, epochs=max(10, EPOCHS // 2), callbacks=cbs)

# ===== Evaluate =====
print("Evaluating on test set")
test_results = model.evaluate(test_ds, return_dict=True)
print("Test metrics:")
for k, v in test_results.items():
    print(f" - {k}: {v:.4f}")

# ===== Save artifacts to V2 =====
final_keras = V2_MODELS / "rock_classifier_multitask_v2.keras"
final_sm    = V2_MODELS / "SavedModel_RockClassifier_V2"

model.save(final_keras.as_posix())
model.export(final_sm.as_posix())
print(f"Saved: {final_keras.name} and {final_sm.name}/")

# Label maps
label_maps_path = V2_MODELS / "label_maps_v2.json"
with open(label_maps_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "valid_classes": list(map(str, valid_classes)),
            "type_names": list(map[str, type_names]) if False else list(map(str, type_names)),
            "best_checkpoint": best_ckpt.name
        },
        f, indent=2
    )
print(f"Saved label maps → {label_maps_path.name}")

# Histories & test metrics
hist_out = V2_LOGS / "history_v2.json"
hist = {"phase1": history.history, "phase2": history_ft.history}
with open(hist_out, "w", encoding="utf-8") as f:
    json.dump(hist, f, indent=2)

test_out = V2_LOGS / "test_metrics_v2.json"
with open(test_out, "w", encoding="utf-8") as f:
    json.dump({k: float(v) for k, v in test_results.items()}, f, indent=2)

print(f"Logs saved  {hist_out.name}, {test_out.name}")
print("V2 complete")
