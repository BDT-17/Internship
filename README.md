# Phase 2: Plant Species Classification - Implementation

This repository currently implements a Phase 2 experiment that compares a CNN trained from scratch against pretrained ResNet50 variants for plant species classification.

Important: the current notebook no longer uses the original 15 PlantVillage disease classes directly. It now uses a merged dataset with:
- 3 crop-level classes from PlantVillage: `Pepper` (2,475 images), `Potato` (2,152 images), `Tomato` (16,011 images)
- 73 tree-genus classes from Leafsnap (including `Abies`, `Acer`, `Betula`, `Magnolia`, `Pinus`, `Quercus`, `Ulmus`)
- 76 total classes in `dataset_plant_classification` (pure image files only, metadata `.txt` files removed)

## Current project structure

```text
internship/
|-- Phase_1_Literature_Review_Report.md
|-- Phase_2_Plant_Classification.ipynb
|-- requirements.txt
|-- dataset/
|   |-- PlantVillage/
|   `-- leafsnap-dataset/
|-- dataset_clean/
|   `-- dataset/
|-- dataset_plant_classification/
|-- 01_extract_clean.py
|-- 02_restructure_dataset.py
|-- IMPLEMENTATION_SUMMARY.md
`-- guide/
```

## Dataset status

### 1. Original dataset folders

`dataset/PlantVillage`
- 15 classes
- 20,639 images total
- Class breakdown:
  - Pepper: 2 disease folders
  - Potato: 3 disease folders
  - Tomato: 10 disease folders

`dataset/leafsnap-dataset`
- `images/lab`: 185 species, 23,147 images
- `images/field`: 184 species, 7,719 images
- This is a tree-species dataset, not a crop disease dataset

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
- disease labels are discarded
- folders are grouped by crop name only

Examples:
- `Pepper__bell___Bacterial_spot` -> `Pepper`
- `Pepper__bell___healthy` -> `Pepper`
- all 3 potato disease folders -> `Potato`
- all 10 tomato disease folders -> `Tomato`

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
jupyter notebook Phase_2_Plant_Classification.ipynb
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

## Current research caveat

If the internship goal is plant disease classification, the current merged dataset is still not a clean disease-classification benchmark, because:
- PlantVillage labels were collapsed from disease level to crop level
- Leafsnap labels are tree genera, not crop diseases

For a stronger internship project, a better target is one of these:
1. keep the current 15 PlantVillage classes and study disease classification properly
2. use the current 76-class merged dataset as a broader plant classification experiment
3. keep both setups and compare disease-level vs genus/crop-level classification

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

- [PROJECT_XRAY.md](PROJECT_XRAY.md)
- [CHANGELOG.md](CHANGELOG.md)
- [Phase_1_Literature_Review_Report.md](Phase_1_Literature_Review_Report.md)
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- [guide/ACTION_PLAN_Next_Steps_PLANT.txt](guide/ACTION_PLAN_Next_Steps_PLANT.txt)
- [guide/dataset_overview.md](guide/dataset_overview.md)
