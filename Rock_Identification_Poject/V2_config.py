from pathlib import Path

# Dataset settings
DATASET_DIR = Path("Rock_Identification_Poject/Outputs/rock_dataset_split")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MIN_IMAGES_PER_CLASS = 30

# Training settings
ALPHA = 0.3 
BETA  = 0.7  
EPOCHS = 25  
UNFREEZE_LAST = 80  

# Outputs
V2_ROOT = Path("Rock_Identification_Poject/Outputs/V2")
V2_MODELS = V2_ROOT / "models"
V2_LOGS   = V2_ROOT / "logs"

# Ensure folders exist
V2_MODELS.mkdir(parents=True, exist_ok=True)
V2_LOGS.mkdir(parents=True, exist_ok=True)
