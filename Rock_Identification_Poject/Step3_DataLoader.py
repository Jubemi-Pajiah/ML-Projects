import tensorflow as tf
import pandas as pd
from config import DATASET_DIR, IMG_SIZE, BATCH_SIZE, MIN_IMAGES_PER_CLASS

def load_datasets(dataset_dir=DATASET_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE, min_images=MIN_IMAGES_PER_CLASS):
    dist_file = dataset_dir / "class_distribution.csv"
    df = pd.read_csv(dist_file)

    # Filtering out the classes that are too small
    valid_classes = df[df["Total"] >= min_images]["Class"].tolist()
    skipped_classes = df[df["Total"] < min_images]["Class"].tolist()

    print(f"Keeping {len(valid_classes)} classes")
    if skipped_classes:
        print(f"Skipping {len(skipped_classes)} classes (too few images): {', '.join(skipped_classes)}")

    train_augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.1)
    ])

#Loading the datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / "train",
        labels="inferred",
        label_mode="categorical",
        class_names=valid_classes,
        image_size=img_size,
        batch_size=batch_size
    ).map(lambda x, y: (train_augment(x) / 255.0, y))

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / "val",
        labels="inferred",
        label_mode="categorical",
        class_names=valid_classes,
        image_size=img_size,
        batch_size=batch_size
    ).map(lambda x, y: (x / 255.0, y))

    test_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / "test",
        labels="inferred",
        label_mode="categorical",
        class_names=valid_classes,
        image_size=img_size,
        batch_size=batch_size
    ).map(lambda x, y: (x / 255.0, y))

    return train_ds, val_ds, test_ds, valid_classes

if __name__ == "__main__":
    train_ds, val_ds, test_ds, valid_classes = load_datasets()
    print(f"Dataset loaded: {len(valid_classes)} valid classes")
