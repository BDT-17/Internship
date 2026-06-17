# Phase 2 Implementation Summary

## Current implementation state

### Main notebook

`notebooks/Phase_2_Plant_Classification.ipynb`
- dataset loading with stratified split
- 3 model settings:
  - custom CNN from scratch
  - ResNet50 feature extraction
  - ResNet50 fine-tuning
- training loop with early stopping
- evaluation metrics and comparison plots
- confusion matrices and final summary cells

### Support files

`requirements.txt`
- torch
- torchvision
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- Pillow
- jupyter
- ipykernel

`scripts/01_extract_clean.py`
- extracts the raw zip into `dataset_clean/`
- trims problematic trailing spaces from archive member names

`scripts/02_restructure_dataset.py`
- converts the source data into `dataset_plant_classification/`
- collapses PlantVillage source folders into crop-level folders
- groups Leafsnap species folders into genus-level folders
- produces the merged training dataset currently used by the notebook

## Dataset audit

### Raw source folders

`dataset/PlantVillage`
- 15 classes
- 20,639 images total
- scope limited to:
  - Pepper: 2 classes
  - Potato: 3 classes
  - Tomato: 10 classes

`dataset/leafsnap-dataset`
- `images/lab`: 185 species, 23,147 images
- `images/field`: 184 species, 7,719 images
- tree species dataset for domain shift / generalization analysis

### Clean extraction folder

`dataset_clean/dataset`
- mirrors the raw package structure after filename cleanup
- still contains `PlantVillage` and `leafsnap-dataset`
- still effectively contains the same 15 PlantVillage classes

### Notebook training dataset

`dataset_plant_classification`
- 76 classes
- 51,504 images total (all metadata `.txt` files removed)
- built from:
  - PlantVillage grouped by crop name (Pepper: 2,475 images, Potato: 2,152 images, Tomato: 16,011 images)
  - Leafsnap grouped by genus name (including Abies, Acer, Betula, Magnolia, Pinus, Quercus, Ulmus)
- output contains image files only:
  - `.jpg`: 51,502
  - `.jpeg`: 1
  - `.png`: 1

## How the merged dataset is built

The current result is caused by the design of `scripts/02_restructure_dataset.py`.

What the script does:
1. rebuilds `dataset_plant_classification` from scratch
2. keeps only image files
3. groups PlantVillage source folders by crop name
4. groups Leafsnap species folders by genus
5. merges both `field` and `lab` Leafsnap images into the same genus folders

Examples:
- `Pepper__bell___Bacterial_spot` -> `Pepper`
- `Tomato_Late_blight` -> `Tomato`
- `acer_rubrum` -> `Acer`
- `quercus_alba` -> `Quercus`
- `pinus_strobus` -> `Pinus`

## Interpretation for the internship

For an internship project, this is now a healthy dataset size for broad plant classification.

Why this matters:
- the task is now mixed crop-level and tree-genus classification
- it is broader and more realistic than the previous 3-class setup
- it should be described as crop/genus plant classification

When this merged dataset is useful:
- as a stronger plant classification internship dataset
- as a transfer learning benchmark across many label groups
- as a practical dataset for a demo classifier

## Recommended next move

Best next option:
- keep the 76-class merged dataset as the main benchmark
- add two pretrained backbones such as EfficientNet-B0 and DenseNet121
- evaluate the best model against the custom CNN baseline using test accuracy, macro F1-score, confusion matrices, and class-level analysis

Reason:
- this matches the current dataset and notebook implementation
- it gives the internship report a clear plant classification story
