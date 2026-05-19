# Phase 2 Implementation Summary

## Current implementation state

### Main notebook

`Phase_2_Plant_Classification.ipynb`
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

`01_extract_clean.py`
- extracts the raw zip into `dataset_clean/`
- trims problematic trailing spaces from archive member names

`02_restructure_dataset.py`
- converts the source data into `dataset_plant_classification/`
- collapses PlantVillage disease folders into crop-level folders
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

The current result is caused by the design of `02_restructure_dataset.py`.

What the script does:
1. rebuilds `dataset_plant_classification` from scratch
2. keeps only image files
3. groups PlantVillage disease folders by crop name
4. groups Leafsnap species folders by genus
5. merges both `field` and `lab` Leafsnap images into the same genus folders

Examples:
- `Pepper__bell___Bacterial_spot` -> `Pepper`
- `Tomato_Late_blight` -> `Tomato`
- `acer_rubrum` -> `Acer`
- `quercus_alba` -> `Quercus`
- `pinus_strobus` -> `Pinus`

## Interpretation for the internship

For an internship project, this is now a much healthier dataset size for general plant classification, but it is still a different task from plant disease classification.

Why this matters:
- the task is now mixed crop-level and tree-genus classification
- it is broader and more realistic than the previous 3-class setup
- but it no longer matches a pure disease-classification research question

When this merged dataset is useful:
- as a stronger plant classification internship dataset
- as a transfer learning benchmark across many label groups
- as a practical dataset for a demo classifier

## Recommended next move

Best next option:
- decide explicitly between:
  - disease classification using the original 15 PlantVillage classes
  - broad plant classification using the new 76-class merged dataset

Reason:
- both are now technically feasible with the local data you already have
- the right choice depends on whether the internship topic is about diseases or plant identity
