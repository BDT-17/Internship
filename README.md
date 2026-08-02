# Plant Image Classification: From Scratch versus Pretrained CNNs

Bachelor thesis project. Four CNNs are trained on a 76-class plant dataset that merges
crop labels from PlantVillage with tree-genus labels from Leafsnap, and compared under
one deterministic split.

The headline comparison — scratch versus pretrained — is the expected one. The question
the project actually asks is **what the models learn**: because Leafsnap contains *lab*
and *field* photographs of the *same genera*, the accuracy difference between those two
domains measures how much a model depends on capture conditions rather than on leaf
shape, with the label space held fixed.

## Results

| Model | Top-1 | Macro F1 | Lab → field gap | Grad-CAM attention drift |
| --- | --- | --- | --- | --- |
| Custom CNN | 0.9222 | 0.8730 | −17.5 pts | −0.229 |
| Custom CNN v2 (scratch) | 0.9655 | 0.9637 | −6.7 pts | −0.137 |
| ResNet50 feature extraction | 0.9167 | 0.8737 | −7.1 pts | −0.115 |
| **ResNet50 fine-tuning** | **0.9849** | **0.9747** | **−4.0 pts** | **+0.076** |

Single seed, 7,763 held-out test images, 76 classes.

Three findings:

1. **Frozen ImageNet features are worth about as much as training from scratch**
   (0.8737 vs 0.8730 macro F1). The advantage of transfer learning comes from
   *adapting* the representation, not from possessing it.
2. **A modern training recipe substitutes for pretraining.** Custom CNN v2 uses no
   pretrained weights and lands within 1.1 macro-F1 points of fine-tuning.
3. **Every model degrades from lab to field imagery**, so condition sensitivity is a
   property of the task, not of one architecture — but the gap shrinks as
   representations improve, and Grad-CAM shows the weakest model's attention leaving
   the leaf entirely on field images it gets wrong.

Full write-up: [`docs/reports/report_draft.tex`](docs/reports/report_draft.tex).
Architecture comparison: [`docs/reports/custom_cnn_architectures.md`](docs/reports/custom_cnn_architectures.md).

## Dataset

76 classes, 51,504 images, built by `scripts/02_restructure_dataset.py` from two raw
sources:

| Source | Grouping | Classes | Note |
| --- | --- | --- | --- |
| PlantVillage | by crop, disease labels collapsed | 3 | uniform 256×256, plain background |
| Leafsnap | by genus, `lab` + `field` kept distinguishable | 73 | lab is controlled; field is phone photos outdoors |

`Pepper__bell___Bacterial_spot` and `Pepper__bell___healthy` both become `Pepper`;
`acer_rubrum` and `acer_saccharum` both become `Acer`.

Two properties shape everything downstream:

- **Imbalance is severe** — roughly 250:1 (Tomato 16,011 images, Toona 64), and 45 of
  76 classes hold fewer than 300 images. Model selection and early stopping therefore
  use **validation macro F1**, never accuracy.
- **Source is partly confounded with the label**, since PlantVillage and Leafsnap differ
  in resolution and background. This is why the lab-versus-field comparison *within*
  Leafsnap, where genera are identical, is the measurement that matters.

Leafsnap filenames keep their `ls_lab_` / `ls_field_` prefix after restructuring, which
is what makes the source-conditioned analysis possible after training.

## Layout

```text
plant_classifier/        the library — all real logic lives here
  config.py              TrainConfig: every hyperparameter in one dataclass
  data.py                class discovery, deterministic split, transforms, loaders
  models.py              the model variants
  losses.py              focal / class-balanced / cb-focal, label smoothing
  mixup.py               mixup + cutmix
  training.py            train/eval loop, early stopping, artifact writing
  analysis.py            per-class, crop-vs-genus, lab-vs-field, confusion analysis
  gradcam.py             Grad-CAM saliency on matched lab/field pairs
  inference.py           single-image prediction
  plots.py               history, confusion matrix, comparison figures
  paths.py               locates the dataset and model root on Kaggle or locally

scripts/                 thin CLIs — argument parsing only, no logic
  01_extract_clean.py    unpack and clean the raw archives
  02_restructure_dataset.py  build dataset_plant_classification/
  train_kaggle.py        training
  analyze_kaggle.py      post-training analysis, no retraining
  gradcam_kaggle.py      Grad-CAM panels, no retraining
  infer_kaggle.py        inference

notebooks/
  Kaggle_Run_From_Git.ipynb   the only notebook: pulls this repo, then trains or analyzes

docs/reports/            thesis source, figures, architecture note
```

Anything not listed here — raw data, trained weights, downloaded result bundles, the
compiled PDF — is generated and deliberately untracked. See `.gitignore`.

## Running it

### On Kaggle (normal path)

1. Import `notebooks/Kaggle_Run_From_Git.ipynb` from GitHub. Kaggle does not pull the
   notebook itself on later runs, so **re-import it whenever the notebook changes** —
   the Python code *is* pulled fresh by the second cell.
2. Attach the image dataset and the four model datasets.
3. Leave `MODE = "analyze"` and run every cell. The training cell skips itself in this
   mode, so a top-to-bottom run is safe.
4. Download `plant_research_bundle.zip` from the output panel.

Set `MODE = "train"` to train instead. `SMOKE = True` runs a single epoch — a pipeline
check whose numbers must never reach the report.

### Locally

```bash
pip install -r requirements.txt

# rebuild the dataset from raw archives
python scripts/01_extract_clean.py
python scripts/02_restructure_dataset.py

# analyze already-trained weights (CPU is fine)
python scripts/analyze_kaggle.py \
    --dataset-dir dataset_plant_classification \
    --model-root plant_training_outputs \
    --output-dir plant_training_outputs \
    --models custom_cnn custom_cnn_v2 resnet50_feature_extraction resnet50_fine_tuning

# Grad-CAM panels — only a few dozen images, so this is quick without a GPU
python scripts/gradcam_kaggle.py \
    --dataset-dir dataset_plant_classification \
    --model-root plant_training_outputs \
    --output-dir plant_training_outputs \
    --models custom_cnn custom_cnn_v2 resnet50_feature_extraction resnet50_fine_tuning \
    --max-pairs 6
```

> `DEFAULT_MODEL_NAMES` in `config.py` lists only three models — it omits
> `custom_cnn_v2`. Pass `--models` explicitly, as above, or that model is silently
> skipped.

Training needs a GPU. Analysis and Grad-CAM do not.

## Reproducibility

The train/validation/test split is stratified 70/15/15 and driven by a fixed seed (42),
so the exact held-out test set can be rebuilt from the seed alone on any machine. That
is what lets `analyze_kaggle.py` and `gradcam_kaggle.py` re-evaluate saved checkpoints
long after training, on different hardware, without retraining anything — every number
in the report was produced that way.

## Building the report

```bash
cd docs/reports
pdflatex report_draft.tex   # run three times so the ToC and \ref links settle
```

Needs a LaTeX distribution with `tikz`, `booktabs`, and `titlesec`. The PDF is not
tracked.
