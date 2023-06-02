# 🪨 Rock Identification Project

This project trains a deep learning model to **classify rocks and minerals** into their specific types (e.g., Basalt, Granite, Coal) and also their broader geological families (**Igneous, Metamorphic, Sedimentary**).  I used over 13,000 images for this model. I was inspired to do this after my first field work exercise in Igarra, Edo State. It's an igneous terrain. The idea is to build a rock detection model to prepare me for my next field work 😂😁🤫.

It includes:
- Dataset normalization and organization.  
- Class filtering and stratified splitting (train/val/test).  
- DataLoader with augmentation and normalization.  
- MobileNetV2 backbone with **multi-task heads** (rock + rock type classification).  
- Class imbalance handling (weights for rare classes).  
- Model training (frozen + fine-tuned phases).  
- Saving in both `.keras` and TensorFlow **SavedModel** format.  
- Export of label maps for inference.  

---

## Project Structure

```
Rock_Identification_Poject/
│
├── Input_Resouces/         # Rock labels (CSV), raw images (ignored in git)
├── Outputs/                
│   ├── rock_dataset_clean/
│   ├── rock_dataset_split/
│   └── V2/                 # All Version 2 artifacts (models, logs, metrics)
│
├── config.py                       # Centralized config (paths, hyperparams)
├── Step1_NormalizeImages.py        # Normalize images (resize, RGB, clean folders)
├── Step2_ImageClassification.py    # Generate metadata & class distributions
├── Step3_DataLoader.py             # Load datasets w/ augmentation
├── V2_Step3_DataLoader.py          # Load datasets w/ augmentation version 2
├── TrainModel.py                   # Training pipeline 
├── V2_TrainModel.py                # Training pipeline version 2
├── label_maps.json                 # Class/type label mappings
│
├── requirements.txt                # Dependencies
└── README.md                       # Project documentation
```

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
source venv/bin/activate    # Linux / macOS
venv\Scripts\activate     # Windows (PowerShell)
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Data Preparation (Step 1–2)

- Normalize images (resize to `224x224`, RGB).  
- Classes with fewer than **30 images** are skipped.  
- Dataset is split into **train / val / test** and stored in `Outputs/rock_dataset_split/`.  
- A `class_distribution.csv` file is created for transparency.  

---

## Data Loading (Step 3)

Implemented in **Step3_DataLoader.py**:  
- Training data augmented with random flips, rotations, zooms, and brightness changes.  
- Validation/test datasets normalized only.  
- Outputs TensorFlow `tf.data.Dataset` pipelines.  

---

## Model Training

- Backbone: **MobileNetV2** (pretrained on ImageNet).  
- Two heads:
  - `rock_output`: fine-grained rock classification.  
  - `type_output`: broader rock type classification.  
- Handles class imbalance via **weighted loss**.  
- Training done in two phases:
  1. **Frozen backbone** (train classifier only).  
  2. **Fine-tuning** last 40 layers for better accuracy.  

### Metrics
- Rock accuracy (top-1)  
- Rock top-3 accuracy  
- Rock type accuracy  

---

## Results (V1)

Example (your run may differ):

- **Rock accuracy**: ~41%  
- **Rock top-3 accuracy**: ~59%  
- **Rock type accuracy (Igneous, Metamorphic, Sedimentary)**: ~67%  

### Version 2 (V2) – Enhancements  
- **Augmentation:** stronger (rotation/contrast/brightness).  
- **Epochs:** increased to **25 (frozen) + 8 (fine-tune)**.  
- **Unfreezing:** last **80 layers** (vs. 40 in V1).  

**Outcome:**  
- Rock accuracy (Top-1): **~39%** (slight dip)  
- Rock accuracy (Top-3): **~58%**  
- Type accuracy: **~65%**  

*The result dipped slightly, showing that heavier augmentation and longer training don’t always guarantee improvements. V2, however, established a more robust training pipeline and stored outputs in structured folders for reproducibility.*  

---

## Saved Outputs

After training, you’ll find:

- `rock_classifier_multitask.keras` → lightweight modern Keras model  
- `SavedModel_RockClassifier/` → full TF SavedModel (for Serving / TFLite)  
- `label_maps.json` → maps classes & types for inference  


- `Outputs/V2/models/rock_classifier_multitask.keras`  
- `Outputs/V2/models/SavedModel_RockClassifier_V2/`  
- `Outputs/V2/label_maps.json` 
---

## Credits

- Built with **TensorFlow / Keras**  
- Dataset: Rocks and Minerals (custom cleaned dataset)  
- Author: Jubemi Anthony Pajiah 
- Contact: jubemi.pajiah@eng.uniben.edu || pajiahjubemi@yahoo.com 
