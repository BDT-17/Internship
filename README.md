# Plant Species Classification

Internship project benchmarking a from-scratch CNN against pretrained backbones on a
76-class plant classification task. The dataset merges crop-level labels from
PlantVillage with tree-genus labels from Leafsnap, which makes it strongly
long-tailed (largest class is ~250x the smallest), so **macro F1 matters more than
accuracy** and the pipeline ships loss functions and augmentations aimed at that.

Training is designed to run on Kaggle GPUs; the same code runs locally on CPU for
smoke tests.

## Dataset

`dataset_plant_classification/` holds 76 classes and 51,504 images, produced by
`scripts/02_restructure_dataset.py` from the two raw sources under `dataset/`:

| Source | Grouping | Classes | Examples |
| --- | --- | --- | --- |
| PlantVillage | by crop name (disease labels collapsed) | 3 | `Pepper` (2,475), `Potato` (2,152), `Tomato` (16,011) |
| Leafsnap | by genus (`field` + `lab` images) | 73 | `Acer`, `Pinus`, `Quercus`, `Betula`, `Magnolia` |

Grouping examples: `Pepper__bell___Bacterial_spot` and `Pepper__bell___healthy` both
become `Pepper`; `acer_rubrum` and `acer_saccharum` both become `Acer`. This keeps the
label space broad without exploding into hundreds of species folders. Metadata `.txt`
files are stripped, so the output is images only (`.jpg`, plus one `.jpeg` and one
`.png`).

On Kaggle the dataset lives at
`/kaggle/input/datasets/thngbuduc/plant-classification/dataset_plant_classification`;
`plant_classifier/paths.py` locates it automatically, so you rarely pass `--dataset-dir`.

## Repository layout

```text
Internship/
|-- plant_classifier/          # the library: all real logic lives here
|   |-- config.py              # TrainConfig dataclass (every hyperparameter)
|   |-- data.py                # class discovery, deterministic split, transforms, loaders
|   |-- models.py              # the 6 model variants
|   |-- losses.py              # focal / class-balanced / cb-focal + label smoothing
|   |-- mixup.py               # mixup + cutmix
|   |-- training.py            # train/eval loop, early stopping, artifact writing
|   |-- analysis.py            # post-training per-class / crop-vs-genus / confusion analysis
|   |-- inference.py           # single-image prediction
|   |-- plots.py               # history, confusion matrix, model comparison figures
|   `-- paths.py               # Kaggle vs local path resolution
|-- scripts/                   # thin entry points around the library
|   |-- train_kaggle.py        # training CLI
|   |-- analyze_kaggle.py      # analysis CLI (no retraining)
|   |-- infer_kaggle.py        # inference CLI
|   |-- 01_extract_clean.py    # raw archive extraction
|   |-- 02_restructure_dataset.py  # builds dataset_plant_classification/
|   `-- ...                    # notebook maintenance helpers
|-- notebooks/
|   |-- Kaggle_Run_From_Git.ipynb   # main Kaggle entry: clones this repo, runs training
|   |-- Smoke_Test_Kaggle.ipynb     # 1-epoch sanity check
|   |-- Kaggle_Showcase_Inference.ipynb
|   |-- Kaggle_Full_Training_Plant_Classification.ipynb
|   `-- Phase_2_Plant_Classification.ipynb   # original self-contained notebook
|-- docs/
|-- dataset/                   # raw PlantVillage + Leafsnap
|-- dataset_plant_classification/   # the training dataset
|-- data/archives/
`-- archive/                   # notebook backups and drafts
```

The notebooks call into `plant_classifier/`, they do not duplicate it. `Kaggle_Run_From_Git.ipynb`
clones the repo fresh on every Kaggle run, so pushing to `main` is how you ship a change
to the GPU — see `docs/KAGGLE_WORKFLOW.md`.

## Models

Selected with `--models`. All pretrained backbones fall back to random init if weights
cannot be downloaded, so an offline run still completes.

| Name | What it is | Trainable parts |
| --- | --- | --- |
| `custom_cnn` | 4-conv baseline, from scratch | everything |
| `custom_cnn_v2` | residual + Squeeze-Excitation, ~SE-ResNet-18 scale | everything |
| `resnet50_feature_extraction` | frozen ResNet50 | head only |
| `resnet50_fine_tuning` | ResNet50 | `layer4` + head |
| `efficientnet_v2_s` | EfficientNetV2-S | last 2 feature stages + head |
| `convnext_tiny` | ConvNeXt-Tiny | last 2 feature stages + head |

`DEFAULT_MODEL_NAMES` in `config.py` is still the original three (`custom_cnn`,
`resnet50_feature_extraction`, `resnet50_fine_tuning`), so the newer models only run when
you name them explicitly with `--models`.

## Setup

```bash
cd d:/DS/Internship
venv\Scripts\activate
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Training

```bash
python scripts/train_kaggle.py \
    --dataset-dir dataset_plant_classification \
    --output-dir plant_training_outputs \
    --models custom_cnn_v2 \
    --epochs 50
```

Splits are 70/15/15 stratified per class, seeded (`--seed 42`), and deterministic — which
is what lets `analyze_kaggle.py` reconstruct the exact test set later without retraining.
Early stopping uses patience 7 on validation accuracy. Mixed precision is on by default on
CUDA (`--no-amp` to disable).

### Tier-1 options

These all default to the original baseline behaviour, so they are opt-in and each one can
be ablated independently:

| Flag | Values | Default |
| --- | --- | --- |
| `--optimizer` | `adam`, `adamw` | `adam` |
| `--lr-scheduler` | `none`, `cosine`, `cosine_warmup` | `none` |
| `--warmup-epochs` | int | `3` |
| `--label-smoothing` | float | `0.0` |
| `--loss` | `ce`, `focal`, `class_balanced`, `cb_focal` | `ce` |
| `--focal-gamma` | float | `2.0` |
| `--cb-beta` | float | `0.9999` |
| `--mixup-alpha` / `--cutmix-alpha` | float | `0.0` (off) |
| `--mixup-prob` | float | `0.5` |

`class_balanced` and `cb_focal` weight classes by effective-number-of-samples (Cui et al.,
CVPR 2019) using the *training-split* class counts; `focal` is Lin et al. (ICCV 2017).
Both target the long tail.

Two things to know when reading the logs of a mixup run: training accuracy looks alarmingly
low (it is measured against the original labels on mixed images — this is expected), and
`cb_focal` validation loss is not comparable to a plain CE run. Compare validation accuracy
and macro F1 instead. Mixup also converges slowly, so 50 epochs may not be enough.

### Outputs

Per model, under `<output-dir>/<model_name>/`: `best_model.pth`, `epoch_NN.pth` (unless
`--no-save-every-epoch`), `history.csv`, `history.png`, `confusion_matrix.png`,
`test_predictions.csv`, `summary.json`. At the run root: `class_names.json`,
`class_counts.csv`, `all_models_summary.csv`/`.json`, `model_comparison.png`.

Note that `--output-dir` is not namespaced per run, so two runs with the same output
directory overwrite each other. Give each ablation its own `--output-dir`.

## Post-training analysis

`analyze_kaggle.py` does not retrain. It rebuilds the deterministic test split, reruns each
saved `best_model.pth`, and writes the metrics the report actually needs given the class
imbalance:

```bash
python scripts/analyze_kaggle.py \
    --dataset-dir dataset_plant_classification \
    --model-root plant_training_outputs \
    --output-dir plant_training_outputs \
    --models custom_cnn_v2 resnet50_fine_tuning
```

Produces `per_class_metrics.csv` (precision/recall/F1/support per class),
`group_analysis.csv` (crop vs tree-genus performance), `most_confused_pairs.csv`,
`per_class_f1_vs_size.png`, and `analysis_summary.json`.

On Kaggle the input tree is read-only, so pass an `--output-dir` under `/kaggle/working`;
the script mirrors the weights across before writing.

## Inference

```bash
python scripts/infer_kaggle.py \
    --model-name resnet50_fine_tuning \
    --model-root plant_training_outputs \
    --image-path path/to/leaf.jpg \
    --top-k 5
```

## Results so far

50-epoch benchmark, validation accuracy:

| Model | Val accuracy |
| --- | --- |
| `resnet50_fine_tuning` | 0.984 |
| `custom_cnn` | 0.920 |
| `resnet50_feature_extraction` | 0.916 |

Fine-tuning clearly wins; feature extraction buys nothing over a custom CNN trained from
scratch on this dataset. The `custom_cnn_v2` and Tier-1 runs are the current work in
progress.

## Research questions

1. Compare a from-scratch CNN against pretrained backbones on 76-class plant classification.
2. Analyze accuracy, macro F1, class imbalance, and training efficiency.
3. Take the strongest pretrained model and compare it deeply against the custom baseline.

## Troubleshooting

**CUDA out of memory** — lower `--batch-size` (e.g. `16`), or `--image-size 160`.

**Dataset not found** — `find_dataset_dir` searches `/kaggle/input` recursively and then the
working directory for a folder named `dataset_plant_classification`. Pass `--dataset-dir`
explicitly if your layout differs.

## References

- [KAGGLE_WORKFLOW.md](docs/KAGGLE_WORKFLOW.md) — how the run-from-git notebook works
- [NEXT_RESEARCH_MODELS.md](docs/NEXT_RESEARCH_MODELS.md)
- [PROJECT_XRAY.md](docs/PROJECT_XRAY.md)
- [CHANGELOG.md](docs/CHANGELOG.md)
- [IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md)
- [Phase_1_Literature_Review_Report.md](docs/reports/Phase_1_Literature_Review_Report.md)
- [dataset_overview.md](docs/guides/dataset_overview.md)
