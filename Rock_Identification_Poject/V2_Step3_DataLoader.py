import tensorflow as tf
import pandas as pd
from config import DATASET_DIR, IMG_SIZE, BATCH_SIZE, MIN_IMAGES_PER_CLASS

def load_datasets(dataset_dir=DATASET_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE, min_images=MIN_IMAGES_PER_CLASS):
    dist_file = dataset_dir / "class_distribution.csv"
    df = pd.read_csv(dist_file)

    valid_classes   = df[df["Total"] >= min_images]["Class"].tolist()
    skipped_classes = df[df["Total"]  < min_images]["Class"].tolist()

    print(f"Keeping {len(valid_classes)} classes")
    if skipped_classes:
        print(f"Skipping {len(skipped_classes)} classes (<{min_images} imgs)")

    # Stronger V2 augmentation
    train_augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2),
    ], name="data_augmentation")

    # Load splits
    def _make_ds(split, augment=False):
        ds = tf.keras.utils.image_dataset_from_directory(
            dataset_dir / split,
            labels="inferred",
            label_mode="categorical",
            class_names=valid_classes,
            image_size=img_size,
            batch_size=batch_size,
            shuffle=True if split == "train" else False
        )
        if augment:
            ds = ds.map(lambda x, y: (train_augment(x) / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(lambda x, y: (x / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)

    train_ds = _make_ds("train", augment=True)
    val_ds   = _make_ds("val",   augment=False)
    test_ds  = _make_ds("test",  augment=False)

    print(f"Datasets → train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")
    return train_ds, val_ds, test_ds, valid_classes

if __name__ == "__main__":
    load_datasets()
