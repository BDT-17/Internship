# Phase 2: Plant Species Classification - Implementation

This repository currently implements a Phase 2 experiment that compares a CNN trained from scratch against pretrained ResNet50 variants for plant species classification.

Important: this project is a plant classification project. The current notebook uses a merged dataset with:
- 3 crop-level classes from PlantVillage: `Pepper` (2,475 images), `Potato` (2,152 images), `Tomato` (16,011 images)
- 73 tree-genus classes from Leafsnap (including `Abies`, `Acer`, `Betula`, `Magnolia`, `Pinus`, `Quercus`, `Ulmus`)
- 76 total classes in `dataset_plant_classification` (pure image files only, metadata `.txt` files removed)

## Current project structure

```text
internship/
|-- requirements.txt
|-- notebooks/
|   |-- Phase_2_Plant_Classification.ipynb
|   |-- Kaggle_Full_Training_Plant_Classification.ipynb
|   |-- Kaggle_Showcase_Inference.ipynb
|   `-- Smoke_Test_Kaggle.ipynb
|-- scripts/
|   |-- 01_extract_clean.py
|   |-- 02_restructure_dataset.py
|   |-- add_visualization_cells.py
|   |-- fix_notebook_path.py
|   `-- update_nb.py
|-- docs/
|   |-- CHANGELOG.md
|   |-- IMPLEMENTATION_SUMMARY.md
|   |-- PROJECT_XRAY.md
|   |-- admin/
|   |-- guides/
|   `-- reports/
|-- data/
|   `-- archives/
|       `-- dataset_plant_classification.zip
|-- archive/
|   |-- Phase_2_Plant_Classification_backup.ipynb
|   |-- drafts/
|   `-- notebook_checkpoints/
|-- dataset/
|   |-- PlantVillage/
|   `-- leafsnap-dataset/
|-- dataset_clean/
|   `-- dataset/
`-- dataset_plant_classification/
```

## Dataset status

### 1. Original dataset folders

`dataset/PlantVillage`
- 15 classes
- 20,639 images total
- Class breakdown:
  - Pepper: 2 source folders
  - Potato: 3 source folders
  - Tomato: 10 source folders

`dataset/leafsnap-dataset`
- `images/lab`: 185 species, 23,147 images
- `images/field`: 184 species, 7,719 images
- This is a tree-species dataset that extends the project beyond crop-only labels

### 2. Cleaned extraction folder

`dataset_clean/dataset`
- This is a cleaned extraction of the same raw dataset package structure
- It still contains:
  - `PlantVillage` with 15 classes
  - `leafsnap-dataset` with Leafsnap content
- It does not reduce the data by itself

### 3. Training dataset used by the notebook

`dataset_plant_classification`
- 76 classes total
- 51,504 images total (all metadata `.txt` files removed)
- built from:
  - PlantVillage grouped by crop name (Pepper: 2,475 images, Potato: 2,152 images, Tomato: 16,011 images)
  - Leafsnap grouped by genus name (including Abies, Acer, Betula, Magnolia, Pinus, Quercus, Ulmus)
- file types inside output:
  - `.jpg`: 51,502
  - `.jpeg`: 1
  - `.png`: 1

This dataset is produced by `02_restructure_dataset.py`.

## How the merged labels work

The merged label logic comes from `02_restructure_dataset.py`.

### PlantVillage
- original PlantVillage subcategory labels are collapsed
- folders are grouped by crop name only

Examples:
- `Pepper__bell___Bacterial_spot` -> `Pepper`
- `Pepper__bell___healthy` -> `Pepper`
- all 3 potato source folders -> `Potato`
- all 10 tomato source folders -> `Tomato`

### Leafsnap
- species labels are grouped by genus
- both `field` and `lab` images are included

Examples:
- `acer_rubrum` -> `Acer`
- `quercus_alba` -> `Quercus`
- `pinus_strobus` -> `Pinus`

This keeps the dataset broad without exploding into hundreds of species-specific folders.

## Environment setup

### 1. Activate virtual environment

```bash
cd d:/DS/Internship
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify PyTorch

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## Running the experiment

```bash
jupyter notebook notebooks/Phase_2_Plant_Classification.ipynb
```

## Notebook contents

### Sections 1-5
- imports and setup
- dataset loading with stratified split
- augmentation pipeline
- model definitions
- training and evaluation helpers

### Sections 6-8
- Experiment 1: custom CNN from scratch
- Experiment 2: ResNet50 feature extraction
- Experiment 3: ResNet50 fine-tuning

### Sections 9-10
- metrics comparison
- training curves
- confusion matrices
- hypothesis review
- conclusions

## Expected outputs

After running the notebook you should get:
- `training_curves.png`
- `confusion_matrices.png`
- `metrics_comparison.png`

## Current research framing

This repository is framed as a broad plant classification project. The final dataset combines:
- crop-level PlantVillage labels: `Pepper`, `Potato`, `Tomato`
- tree-genus Leafsnap labels such as `Acer`, `Pinus`, and `Quercus`

The recommended research target is therefore:
1. compare custom CNN and pretrained CNN backbones on the 76-class plant classification task
2. analyze accuracy, macro F1-score, class imbalance, and training efficiency
3. select the strongest pretrained model and compare it deeply against the custom CNN baseline

## Troubleshooting

### CUDA out of memory

Reduce the batch size in the notebook, for example:

```python
batch_size = 16
```

### Dataset path not found

The current notebook expects:

```python
dataset_path = "dataset_plant_classification"
```

## References

- [PROJECT_XRAY.md](docs/PROJECT_XRAY.md)
- [CHANGELOG.md](docs/CHANGELOG.md)
- [Phase_1_Literature_Review_Report.md](docs/reports/Phase_1_Literature_Review_Report.md)
- [IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md)
- [ACTION_PLAN_Next_Steps_PLANT.txt](docs/guides/ACTION_PLAN_Next_Steps_PLANT.txt)
- [dataset_overview.md](docs/guides/dataset_overview.md)
