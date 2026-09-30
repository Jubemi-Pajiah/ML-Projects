# Rock Identification Project

This project trains a deep learning model to **classify rocks and minerals** into specific types (for example basalt, granite, coal) and into their broader geological families (**igneous, metamorphic, sedimentary**). It uses more than 13,000 images. The idea came from my first field work exercise in Igarra, Edo State, an igneous terrain, and the goal was a model that could help identify rocks in the field.

It includes:
- Dataset normalization and organization.
- Class filtering and stratified splitting (train/val/test).
- DataLoader with augmentation and normalization.
- MobileNetV2 backbone with **multi-task heads** (rock and rock type classification).
- Class imbalance handling (weights for rare classes).
- Model training (frozen and fine-tuned phases).
- Saving in both `.keras` and TensorFlow **SavedModel** format.
- Export of label maps for inference.

---

## Project Structure

```
Rock_Identification_Poject/
|-- Input_Resouces/            # Rock labels (CSV); raw images are ignored in git
|-- Outputs/
|   |-- rock_dataset_clean/
|   |-- rock_dataset_split/
|   `-- V2/                    # All Version 2 artifacts (models, logs, metrics)
|-- config.py                  # Centralized config (paths, hyperparameters)
|-- Step1_NormalizeImages.py   # Normalize images (resize, RGB, clean folders)
|-- Step2_ImageClassification.py  # Generate metadata and class distributions
|-- Step3_DataLoader.py        # Load datasets with augmentation
|-- V2_Step3_DataLoader.py     # Load datasets with augmentation (version 2)
|-- TrainModel.py              # Training pipeline
|-- V2_TrainModel.py           # Training pipeline (version 2)
|-- label_maps.json            # Class/type label mappings
`-- README.md                  # Project documentation
```

Dependencies are listed in `requirements.txt` at the repository root.

---

## Setup

### 1. Clone repository
```bash
git clone https://github.com/Jubemi-Pajiah/ML-Projects.git
cd ML-Projects
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows (PowerShell)
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Data Preparation (Step 1-2)

- Normalize images (resize to `224x224`, RGB).
- Classes with fewer than **30 images** are skipped.
- The dataset is split into **train / val / test** and stored in `Outputs/rock_dataset_split/`.
- A `class_distribution.csv` file is created for transparency.

---

## Data Loading (Step 3)

Implemented in `Step3_DataLoader.py`:
- Training data is augmented with random flips, rotations, zooms, and brightness changes.
- Validation and test datasets are normalized only.
- Outputs TensorFlow `tf.data.Dataset` pipelines.

---

## Model Training

- Backbone: **MobileNetV2** (pretrained on ImageNet).
- Two heads:
  - `rock_output`: fine-grained rock classification.
  - `type_output`: broader rock type classification.
- Class imbalance is handled with a weighted loss.
- Training runs in two phases:
  1. **Frozen backbone** (train the classifier only).
  2. **Fine-tuning** the last 40 layers for better accuracy.

### Metrics
- Rock accuracy (top-1)
- Rock top-3 accuracy
- Rock type accuracy

---

## Results

### Version 1 (V1)

- **Rock accuracy (top-1)**: ~41%
- **Rock accuracy (top-3)**: ~59%
- **Rock type accuracy (igneous, metamorphic, sedimentary)**: ~67%

### Version 2 (V2) enhancements
- **Augmentation:** stronger (rotation, contrast, brightness).
- **Epochs:** increased to **25 (frozen) + 8 (fine-tune)**.
- **Unfreezing:** last **80 layers** (vs. 40 in V1).

**Outcome:**
- Rock accuracy (top-1): **~39%** (slight dip)
- Rock accuracy (top-3): **~58%**
- Rock type accuracy: **~65%**

Heavier augmentation and longer training did not improve results here. V2 did establish a more robust training pipeline, with outputs stored in structured folders for reproducibility. Telling many visually similar rocks apart from photos alone is hard, so the next steps would be more data for rare classes and richer views of each sample.

---

## Saved Outputs

After training you will find:

- `rock_classifier_multitask.keras`: lightweight modern Keras model
- `SavedModel_RockClassifier/`: full TF SavedModel (for Serving / TFLite)
- `label_maps.json`: maps classes and types for inference

V2 artifacts:
- `Outputs/V2/models/rock_classifier_multitask.keras`
- `Outputs/V2/models/SavedModel_RockClassifier_V2/`
- `Outputs/V2/label_maps.json`

---

## Credits

- Built with **TensorFlow / Keras**
- Dataset: Rocks and Minerals (custom cleaned dataset)
- Author: Jubemi Pajiah
- Contact: info@jubemi.com
