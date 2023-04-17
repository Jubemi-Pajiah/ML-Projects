from pathlib import Path

# Dataset settings
DATASET_DIR = Path("Rock_Identification_Poject/Outputs/rock_dataset_split")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MIN_IMAGES_PER_CLASS = 30

# Training settings
ALPHA = 0.3  # loss weight for rock type
BETA = 0.7   # loss weight for rock name
EPOCHS = 10
