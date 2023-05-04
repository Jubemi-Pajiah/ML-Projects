# 🪨 Rock Identification Project

This project trains a deep learning model to **classify rocks and minerals** into their specific types (e.g., Basalt, Granite, Coal) and also their broader geological families (**Igneous, Metamorphic, Sedimentary**).  

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

## 📂 Project Structure

```
Rock_Identification_Poject/
│
├── Input_Resouces/         # Rock labels (CSV), Raw images used (ignored in git).
├── Outputs/                # Cleaned & split dataset (ignored in git).
│   ├── rock_dataset_clean/
│   └── rock_dataset_split/
│
├── config.py                       # Centralized config (paths, hyperparams).
├── Step1_NormalizeImages.py        # Preprocess raw rock images: resize to uniform size, normalize, rename, and organize them into clean class folders.
├── Step2_ImageClassification.py    # Generate metadata (CSV of rock labels, class distributions, etc.) and prepare for classification tasks.
├── Step3_DataLoader.py             # Loads dataset with augmentation.
├── TrainModel.py                   # Training pipeline (Steps 4–7).
├── label_maps.json                 # Saved mapping of classes and rock families.
│
├── requirements.txt                # Python dependencies.
└── README.md                       # Project documentation.
```

---

## ⚙️ Setup

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

## 🧹 Data Preparation (Step 1–2)

- Normalize images (resize to `224x224`, RGB).  
- Classes with fewer than **30 images** are skipped.  
- Dataset is split into **train / val / test** and stored in `Outputs/rock_dataset_split/`.  
- A `class_distribution.csv` file is created for transparency.  

---

## 📦 Data Loading (Step 3)

Implemented in **Step3_DataLoader.py**:  
- Training data augmented with random flips, rotations, zooms, and brightness changes.  
- Validation/test datasets normalized only.  
- Outputs TensorFlow `tf.data.Dataset` pipelines.  

---

## 🧠 Model Training (Step 4–7)

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

## 📊 Results (V1)

Example (your run may differ):

- **Rock accuracy**: ~41%  
- **Rock top-3 accuracy**: ~59%  
- **Rock type accuracy (Igneous, Metamorphic, Sedimentary)**: ~67%  

---

## 💾 Saved Outputs

After training, you’ll find:

- `rock_classifier_multitask.keras` → lightweight modern Keras model  
- `SavedModel_RockClassifier/` → full TF SavedModel (for Serving / TFLite)  
- `label_maps.json` → maps classes & types for inference  

---

## 🙌 Credits

- Built with **TensorFlow / Keras**  
- Dataset: Rocks and Minerals (custom cleaned dataset)  
- Author: Jubemi Anthony Pajiah 
- Contact: jubemi.pajiah@eng.uniben.edu || pajiahjubemi@yahoo.com 
