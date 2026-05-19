# PROJECT X-RAY: Plant Species Classification Experiment

Generated: 2026-05-19
By: Antigravity — XRAY Protocol (Vibecode Kit v5.0 Framework)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Quick Start](#2-quick-start)
3. [Architecture](#3-architecture)
4. [Key Components](#4-key-components)
5. [API Reference](#5-api-reference)
6. [Database Schema (Dataset Structure)](#6-database-schema-dataset-structure)
7. [Environment Variables](#7-environment-variables)
8. [Build History (TIP Traceability)](#8-build-history-tip-traceability)
9. [Requirements Traceability (Research Hypotheses)](#9-requirements-traceability-research-hypotheses)
10. [Deployment](#10-deployment)
11. [Common Tasks](#11-common-tasks)
12. [Troubleshooting](#12-troubleshooting)
13. [Future Improvements](#13-future-improvements)

---

## 1. Overview

This project is part of a machine learning internship evaluating **Plant Species Classification**. Specifically, it implements an empirical study comparing:
1. A **Custom Convolutional Neural Network (CNN)** trained from scratch.
2. A **Pretrained ResNet50** utilizing **Feature Extraction** (frozen backbone, trained linear classifier).
3. A **Pretrained ResNet50** utilizing **Fine-tuning** (progressive unfreezing of deep layers).

The study aims to validate transfer learning benefits regarding classification accuracy, generalization gap (overfitting risk), training time/efficiency, and F1-score optimization under domain-shifted conditions. 

### Key Dataset Merging Logic:
The project uses a merged dataset combining:
- **PlantVillage**: Grouped at crop-level (Pepper: 2,475 images, Potato: 2,152 images, Tomato: 16,011 images) by discarding individual disease subcategories.
- **LeafSnap**: Grouped at genus-level (including Abies, Acer, Betula, Magnolia, Pinus, Quercus, Ulmus) by grouping species names by scientific genus prefix.

---

## 2. Quick Start

Follow these steps to run the pipeline locally:

### Step 1: Set Up Virtual Environment (venv)
Activate the pre-configured virtual environment:
```powershell
# In PowerShell (Windows)
venv\Scripts\activate
```
*(On Linux/macOS, run: `source venv/bin/activate`)*

### Step 2: Install Dependencies
Install all required machine learning and data processing libraries:
```bash
pip install -r requirements.txt
```

### Step 3: Verify PyTorch & GPU (CUDA) Support
Confirm that PyTorch has loaded successfully and detects your GPU:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Step 4: Extract and Restructure Dataset
If starting with raw archives:
```bash
# Extract zip archive (trims problematic trailing spaces from member paths)
python 01_extract_clean.py

# Group classes by crop (PlantVillage) and genus (LeafSnap), then structure training folder
python 02_restructure_dataset.py
```
*Note: The processed dataset `dataset_plant_classification` containing 76 classes and 51,504 images will be generated automatically.*

### Step 5: Launch Jupyter & Run Experiment
Start the notebook server:
```bash
jupyter notebook Phase_2_Plant_Classification.ipynb
```
Execute all cells sequentially to run the models and generate comparative figures:
- `training_curves.png`
- `confusion_matrices.png`
- `metrics_comparison.png`

---

## 3. Architecture

### Directory Tree
```text
internship/
├── .codex/                        # Vibecode Kit v6.0 skills and master prompts
│   ├── Masters/                   # XRAY, QA, and DEBUG Master Protocols
│   └── skill-v6/                  # Core skill definitions and references
├── dataset/                       # Raw source datasets
│   ├── PlantVillage/              # 15 classes, 20,639 images (crop diseases)
│   └── leafsnap-dataset/          # images/field (7.7K) & images/lab (23.1K) tree species
├── dataset_clean/                 # Intermediary clean extracted dataset
├── dataset_plant_classification/  # Final structured training dataset (76 classes, 51.5K images)
├── guide/                         # Documentation, guides, and internship abstracts
├── scratch/                       # Temporary scripts and parsed proposal texts
├── venv/                          # Python virtual environment
├── 01_extract_clean.py            # Dataset extraction script
├── 02_restructure_dataset.py      # Dataset consolidation script
├── clean.py                       # Helper script to clean invalid string characters in paths
├── update_nb.py                   # Script updating notebook path variables
├── add_visualization_cells.py     # Script injecting visualization cells into notebook
├── fix_notebook_path.py           # Helper to resolve hardcoded notebook paths
├── Phase_1_Literature_Review_Report.md # Theoretical foundation & review
├── Phase_2_Plant_Classification.ipynb  # Primary Jupyter notebook containing code & experiments
├── IMPLEMENTATION_SUMMARY.md      # Summary of the Phase 2 dataset architecture
└── README.md                      # General user-facing overview
```

### Data Pipeline & Model Training Flow
```text
Raw Archives (dataset_clean.zip / dataset.zip)
                 │
                 ▼ [01_extract_clean.py]
          dataset_clean/
                 │
                 ▼ [02_restructure_dataset.py] (Crop/Genus grouping)
     dataset_plant_classification/ ──► Stratified Split (70% Train, 15% Val, 15% Test)
                 │
  ┌──────────────┼──────────────────────────────┐
  ▼              ▼                              ▼
Custom CNN    ResNet50 Feature Extraction    ResNet50 Fine-Tuning
(Scratch)     (Frozen backbone, train head)  (Deeps layer unfreezed, low LR)
  │              │                              │
  └──────────────┼──────────────────────────────┘
                 ▼
          Evaluation Head
  (Accuracy, Precision, Recall, F1-Score)
                 │
                 ▼
          Visual Outcomes
 (Curves, Confusion Matrices, Comparison PNGs)
```

---

## 4. Key Components

### 1. Data Preparation Scripts
* **`01_extract_clean.py`**: Handles decompression of the raw zipped dataset. It includes custom sanitization logic that strips trailing spaces (`member.filename.rstrip()`) from pathnames during extraction to avoid Windows filesystem exceptions.
* **`02_restructure_dataset.py`**: Restructures class folders into the final unified format.
  * Collapses PlantVillage labels: `Potato___Early_blight` -> `Potato`, `Tomato___Bacterial_spot` -> `Tomato`.
  * Collapses LeafSnap labels to genus-level: `acer_rubrum` -> `Acer`, `quercus_alba` -> `Quercus`.
  * Merges LeafSnap field and lab images into a single directory.
* **`clean.py`**: Scans the PlantVillage directory and replaces non-breaking space characters (`\xa0`) or trailing whitespaces with empty characters.

### 2. Notebook Management & Hotfix Tools
* **`update_nb.py`**: Utility to update paths inside the JSON file of the notebook, replacing references from `dataset/PlantVillage` to the structured `dataset_plant_classification` path.
* **`fix_notebook_path.py`**: Ensures the data load path parameter inside `Phase_2_Plant_Classification.ipynb` does not contain local absolute file references.
* **`add_visualization_cells.py`**: Programmatically appends metrics tables, training curve plots, confusion matrices, and conclusions to the backup notebook to form the final publication notebook.

### 3. Primary Executable Notebook
* **`Phase_2_Plant_Classification.ipynb`**:
  * Loads images using a custom `Dataset` class (`PlantDiseaseDataset`) with Pillow.
  * Implements stratified splits by ensuring class distributions remain consistent across train, validation, and test datasets.
  * Implements image augmentation (Resize 256x256, Random Flips, Rotations, Color Jitter, and ImageNet standardization).
  * Defines the three candidate architectures and executes training/evaluation metrics logs.

---

## 5. API Reference

*No API endpoints or backend routing are defined. This repository is purely a Deep Learning model training pipeline.*

---

## 6. Database Schema (Dataset Structure)

Data is stored as raw images structured into folders. The final directory layout inside `dataset_plant_classification` acts as the label index. All metadata and `.txt` files have been removed, leaving only pure image files.

```text
dataset_plant_classification/
├── Abies/
├── Acer/
├── Betula/
├── Magnolia/
├── Pinus/
├── Quercus/
├── Ulmus/
├── Pepper/
├── Potato/
├── Tomato/
└── [Remaining 66 classes ...]
```

### Dataset Statistics
* **Total Image Files**: 51,504 (pure image files only, zero metadata `.txt` files remaining)
  - `.jpg`: 51,502
  - `.jpeg`: 1
  - `.png`: 1
* **Total Genus/Crop Classes**: 76
* **Class Breakdowns**:
  - **PlantVillage Crop Classes**:
    - `Pepper`: 2,475 images
    - `Potato`: 2,152 images
    - `Tomato`: 16,011 images
  - **Leafsnap Tree Genus Classes**: Abies, Acer, Betula, Magnolia, Pinus, Quercus, Ulmus, and 66 other genera.
* **Form Factor**: Standardized to `256 x 256` in PyTorch loaders.

---

## 7. Environment Variables

There are **no environment variables** required for running the pipeline. Configuration is managed via local python parameters inside the scripts and the notebook.

---

## 8. Build History (TIP Traceability)

*Note: This project was not originally constructed using the Vibecode 8-step build sequence, so it does not contain a formal `task.md` or Contractor TIP trace logs.*

However, the historical pipeline changes can be traced as follows:
* **Initial Setup**: Phase 1 Literature review established theoretical goals.
* **Phase 2 Pipeline**: Initial scripts (`01_extract_clean.py`, `02_restructure_dataset.py`) created to handle dataset processing.
* **Refactoring Phase**: Helper scripts (`update_nb.py`, `fix_notebook_path.py`, `add_visualization_cells.py`) implemented to resolve pathing issues and add visual comparison outputs to the Jupyter notebook.

---

## 9. Requirements Traceability (Research Hypotheses)

The empirical validation in the Jupyter notebook maps back to the Research Hypotheses defined in the **Phase 1 Literature Review**:

| Hypothesis | Theoretical Claim | Implemented Validation Cell | Result status |
| :--- | :--- | :--- | :--- |
| **H1** | Pretrained models outperform scratch models in accuracy. | Section 9 & 10 (Comparison Bar Chart & Results Table) | **Confirmed** (ResNet50 FT achieves higher test accuracy) |
| **H2** | Pretrained models exhibit smaller generalization gaps. | Section 9 & 10 (Train Acc - Val Acc metrics calculation) | **Confirmed** (Lower train-val discrepancy than custom CNN) |
| **H3** | Pretrained models converge faster. | Section 9 & 10 (Training time outputs in seconds) | **Confirmed** (Feature Extraction converges ~10x faster) |
| **H4** | Pretrained models achieve higher F1-scores. | Section 9 & 10 (Weighted average F1-score evaluation) | **Confirmed** |

---

## 10. Deployment

Since this is an evaluation experiment, there is no real-time web deployment. For production tasks or deployment in remote GPU environments (e.g. Kaggle / Colab / AWS):

### 1. Exporting Trained Weights
To save a model's state dictionary after training:
```python
torch.save(model_ft.state_dict(), 'resnet50_plant_classification_finetuned.pth')
```

### 2. Running Headless (No Jupyter UI)
To execute the pipeline directly in terminal (e.g. via SSH on a remote server):
```bash
jupyter nbconvert --to script Phase_2_Plant_Classification.ipynb
python Phase_2_Plant_Classification.py
```

### 3. Kaggle GPU Training (Configured)
The notebook features dynamic environment detection. When uploaded to Kaggle:
- It checks for the existence of `/kaggle/input`.
- It automatically scans `/kaggle/input/*` to locate the correct directory for the `dataset_plant_classification` dataset, resolving nested folder scenarios automatically.
- Output plots and model weights will be exported directly to `/kaggle/working/`, ready for download.

---

## 11. Common Tasks

### Task A: Adjusting Training Hyperparameters
Open the notebook, go to Sections 6-8, and edit hyperparameters in model initializers:
```python
# Reduce batch size if CUDA Out of Memory occurs
batch_size = 16 

# Adjust learning rate for fine-tuning
optimizer = optim.Adam(model_ft.parameters(), lr=0.00005)
```

### Task B: Evaluating a New Pretrained Model (e.g., EfficientNet)
To swap out ResNet50 for another state-of-the-art model:
1. Import the model from torchvision:
   ```python
   import torchvision.models as models
   model = models.efficientnet_b0(pretrained=True)
   ```
2. Modify the classifier head to output `num_classes`:
   ```python
   model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
   ```

---

## 12. Troubleshooting

### 1. CUDA Out of Memory (OOM)
* **Symptoms**: `RuntimeError: CUDA out of memory.`
* **Solution**:
  1. Open Section 3 in the notebook.
  2. Change `batch_size = 32` to `batch_size = 16` or `batch_size = 8`.
  3. Restart the kernel and run again.

### 2. Missing Dataset Folder
* **Symptoms**: `FileNotFoundError: [Errno 2] No such file or directory: 'dataset_plant_classification'`
* **Solution**: Ensure you have executed `02_restructure_dataset.py` before launching the notebook. If the raw archives are in zip format, run `01_extract_clean.py` first.

### 3. File Extract / Zip Name Mismatch
* **Symptoms**: `FileNotFoundError: [Errno 2] No such file or directory: 'dataset.zip'` in `01_extract_clean.py`.
* **Solution**: Change the `zip_path` variable in line 4 of `01_extract_clean.py` to match your local zip name (e.g. `"dataset_clean.zip"`).

---

## 13. Future Improvements

1. **Modular Code Structure**: Move model definitions, custom dataset loaders, and evaluation functions from notebook cells into separate, clean Python module files (`models.py`, `dataset.py`, `utils.py`) to reduce duplication.
2. **Standard Configuration File**: Implement a `config.yaml` to centralize parameters (learning rates, batch sizes, dataset paths, training epochs) instead of hardcoding them in scripts.
3. **Experiment Tracking**: Integrate Weights & Biases (W&B) or MLflow to track runs, compare logs, and version model checkpoints automatically.
4. **Enhanced Data Augmentation**: Introduce advanced techniques like Mixup, CutMix, or RandAugment to improve generalization further.
5. **Class Imbalance Handling**: Use weighted CrossEntropyLoss (`WeightedRandomSampler`) to account for differences in representation between rare leaf genera and common crops.

---
*Generated by Antigravity — XRAY Protocol*
