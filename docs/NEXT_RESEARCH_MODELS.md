# Next Research Direction: EfficientNetV2-S and ConvNeXt-Tiny

## 1. Context

This project has already completed the first major benchmark for the 76-class plant classification task. The current pipeline compares:

- Custom CNN trained from scratch
- ResNet50 feature extraction
- ResNet50 fine-tuning

The dataset is a merged plant classification dataset:

- 3 crop-level classes from PlantVillage: `Pepper`, `Potato`, `Tomato`
- 73 tree-genus classes from Leafsnap
- 76 total classes
- 51,504 image files

The next research step should not simply add many random pretrained models. The stronger direction is to add two classification-focused modern backbones, train them under the same pipeline, and then analyze why they perform better or worse than ResNet50.

Recommended new models:

1. EfficientNetV2-S
2. ConvNeXt-Tiny

These two models give a clean research story:

- EfficientNetV2-S tests whether a parameter-efficient modern CNN improves plant classification under limited compute.
- ConvNeXt-Tiny tests whether a modernized pure ConvNet can outperform the older ResNet50 baseline.

## 2. Research Objective

The next phase should answer this question:

> Do modern classification backbones, specifically EfficientNetV2-S and ConvNeXt-Tiny, improve accuracy, macro F1-score, convergence speed, and class-level robustness compared with ResNet50 on the 76-class plant classification dataset?

This is a better research question than only asking which model has the highest accuracy. The dataset is imbalanced and mixes crop-level and tree-genus labels, so macro F1-score and class-level behavior matter.

## 3. Why Not Add Too Many Models

Adding many models can make the project look broad but shallow. For an internship report, a stronger structure is:

- fewer models
- controlled experiments
- deeper error analysis
- clear explanation of architecture differences
- practical conclusion for plant classification

Recommended final benchmark set:

| Group | Model | Purpose |
|---|---|---|
| Baseline | Custom CNN | Shows performance without transfer learning |
| Classic transfer learning | ResNet50 feature extraction | Measures frozen-backbone transfer |
| Classic fine-tuning | ResNet50 fine-tuning | Current pretrained CNN baseline |
| Modern efficient CNN | EfficientNetV2-S fine-tuning | Tests parameter-efficient classification |
| Modern ConvNet | ConvNeXt-Tiny fine-tuning | Tests modernized convolutional design |

## 4. Model 1: EfficientNetV2-S

### 4.1 Summary

EfficientNetV2 is a family of convolutional networks designed for faster training and better parameter efficiency. It uses training-aware neural architecture search and scaling. The paper introduces Fused-MBConv operations and progressive learning ideas to improve speed and accuracy.

EfficientNetV2-S is a good fit for this project because:

- It is designed for image classification.
- It is more modern than ResNet50.
- It is relatively compact for its accuracy.
- It is available directly in TorchVision.
- It can run on Kaggle GPU without becoming too large.

### 4.2 TorchVision reference numbers

According to the TorchVision documentation for `efficientnet_v2_s`:

| Item | Value |
|---|---:|
| TorchVision model name | `efficientnet_v2_s` |
| Weights enum | `EfficientNet_V2_S_Weights.IMAGENET1K_V1` |
| ImageNet-1K acc@1 | 84.228 |
| ImageNet-1K acc@5 | 96.878 |
| Parameters | 21,458,488 |
| GFLOPS | 8.37 |
| File size | 82.7 MB |
| Default inference crop size | 384 |

### 4.3 Why it is useful for plant classification

The current dataset has many classes but limited domain-specific data compared with ImageNet-scale training. EfficientNetV2-S is useful because it provides a strong pretrained representation without requiring a very large model.

Expected strengths:

- Strong classification accuracy.
- Better parameter efficiency than larger CNNs.
- Good transfer learning behavior on medium-sized datasets.
- Potentially strong macro F1-score if fine-tuned carefully.

Potential risks:

- Default TorchVision inference transform uses a larger crop size than the current 224 pipeline.
- Higher GFLOPS than ConvNeXt-Tiny in the TorchVision reference table.
- May require lower batch size on Kaggle if using 384 image size.

### 4.4 Recommended project usage

Use EfficientNetV2-S in two possible ways:

1. Practical benchmark mode:
   - Keep `IMAGE_SIZE = 224`
   - Easier comparison with existing ResNet50 runs
   - Lower memory use

2. Architecture-faithful mode:
   - Use `IMAGE_SIZE = 384`
   - Closer to TorchVision pretrained evaluation transform
   - More expensive on Kaggle

Recommended first run:

```bash
python scripts/train_kaggle.py --epochs 50 --batch-size 16 --models efficientnet_v2_s
```

If Kaggle memory is stable, run:

```bash
python scripts/train_kaggle.py --epochs 50 --batch-size 16 --image-size 384 --models efficientnet_v2_s
```

## 5. Model 2: ConvNeXt-Tiny

### 5.1 Summary

ConvNeXt is a pure convolutional model family that modernizes ResNet-style ConvNets using design ideas from the Transformer era. The paper reexamines ConvNet design and shows that a well-designed convolutional architecture can compete strongly with vision Transformers.

ConvNeXt-Tiny is a good fit for this project because:

- It is still a CNN, so it is easy to compare against ResNet50.
- It is designed as a modern classification backbone.
- It has strong ImageNet performance.
- It is available directly in TorchVision.
- It is a clean research contrast: old ConvNet baseline vs modern ConvNet.

### 5.2 TorchVision reference numbers

According to the TorchVision documentation for `convnext_tiny`:

| Item | Value |
|---|---:|
| TorchVision model name | `convnext_tiny` |
| Weights enum | `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` |
| ImageNet-1K acc@1 | 82.52 |
| ImageNet-1K acc@5 | 96.146 |
| Parameters | 28,589,128 |
| GFLOPS | 4.46 |
| File size | 109.1 MB |
| Default inference crop size | 224 |

### 5.3 Why it is useful for plant classification

ConvNeXt-Tiny is useful because it isolates an important research question:

> If the project keeps a convolutional backbone, does a modern ConvNet architecture outperform ResNet50 on plant classification?

Expected strengths:

- Strong classification backbone.
- Lower GFLOPS than EfficientNetV2-S in TorchVision reference numbers.
- Default crop size aligns well with the current 224 image pipeline.
- Good architectural comparison against ResNet50.

Potential risks:

- More parameters than EfficientNetV2-S.
- May overfit if the head is too flexible or training is too long.
- Needs the correct classifier replacement because TorchVision ConvNeXt uses a different classifier structure from ResNet.

### 5.4 Recommended project usage

Recommended first run:

```bash
python scripts/train_kaggle.py --epochs 50 --batch-size 16 --models convnext_tiny
```

If memory allows:

```bash
python scripts/train_kaggle.py --epochs 50 --batch-size 32 --models convnext_tiny
```

## 6. Why Swin Transformer Should Be Optional

Swin Transformer is also a strong vision backbone, but it should be treated as an optional extension rather than the next immediate step.

Reasons:

- It changes the research comparison from CNN-focused to CNN vs Transformer.
- It can make the report broader but harder to explain deeply.
- EfficientNetV2-S and ConvNeXt-Tiny already provide a strong next phase.
- Kaggle compute limits make it better to first finish a clean CNN-based benchmark.

Recommended use of Swin:

- Add only after EfficientNetV2-S and ConvNeXt-Tiny are trained and analyzed.
- Use it as a final extension section: "Transformer-based future work".

## 7. Experimental Design

### 7.1 Controlled setup

All models should use the same:

- Dataset split
- Random seed
- Train/validation/test ratio
- Optimizer family
- Early stopping rule
- Evaluation metrics
- Output format

Recommended split:

| Split | Ratio |
|---|---:|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

Recommended seed:

```text
42
```

### 7.2 Training settings

Initial recommended settings:

| Setting | Value |
|---|---:|
| Epochs | 50 |
| Batch size | 16 or 32 |
| Optimizer | Adam |
| Weight decay | 1e-4 |
| Early stopping patience | 7 |
| AMP | Enabled on CUDA |
| Image size | 224 first; 384 only for EfficientNetV2-S follow-up |

### 7.3 Learning rates

Recommended learning rates:

| Model | First learning rate |
|---|---:|
| ResNet50 fine-tuning | 1e-4 |
| EfficientNetV2-S | 1e-4 |
| ConvNeXt-Tiny | 1e-4 |

If validation loss is unstable:

```text
Try 5e-5
```

If validation accuracy improves too slowly:

```text
Try 2e-4, but monitor overfitting.
```

## 8. Metrics to Report

Do not report only accuracy. Because the dataset is imbalanced, the report should include both global and class-sensitive metrics.

Required metrics:

| Metric | Why it matters |
|---|---|
| Test accuracy | Overall classification performance |
| Macro F1-score | Treats each class equally; important for rare classes |
| Precision macro | Shows false positive behavior across classes |
| Recall macro | Shows whether minority classes are missed |
| Runtime minutes | Practical training cost |
| Epochs completed | Shows convergence and early stopping behavior |
| Best validation accuracy | Model selection signal |

Recommended additional metrics:

| Metric | Why it matters |
|---|---|
| Weighted F1-score | Balances class frequency with F1 |
| Per-class F1-score | Identifies weak plant classes |
| Confusion matrix | Shows which genera/crops are confused |
| Parameter count | Compares model size |
| Model file size | Deployment relevance |

## 9. Deeper Analysis After Training

### 9.1 Crop vs tree-genus analysis

The dataset mixes two types of labels:

- PlantVillage crop-level labels
- Leafsnap genus-level labels

The report should separate these groups.

Questions to answer:

- Does the model perform better on crop classes or tree-genus classes?
- Are `Pepper`, `Potato`, and `Tomato` easier because they have more data?
- Are rare Leafsnap genera responsible for lower macro F1-score?

Suggested output table:

| Group | Accuracy | Macro F1 | Number of classes | Number of images |
|---|---:|---:|---:|---:|
| PlantVillage crop classes | TBD | TBD | 3 | TBD |
| Leafsnap genus classes | TBD | TBD | 73 | TBD |

### 9.2 Class imbalance analysis

The current dataset has large class imbalance. `Tomato` has many more images than smaller tree genera.

Questions to answer:

- Does the best model mainly improve frequent classes?
- Which model has the best macro F1-score?
- Does a model with slightly lower accuracy have better minority-class behavior?

Recommended plot:

- x-axis: number of images per class
- y-axis: per-class F1-score
- color: model name

### 9.3 Confusion matrix analysis

The confusion matrix should not only be included as a figure. It should be interpreted.

Questions to answer:

- Which genera are most often confused?
- Are visually similar tree genera grouped together by mistake?
- Does ConvNeXt-Tiny reduce confusion compared with ResNet50?
- Does EfficientNetV2-S improve rare-class recall?

### 9.4 Convergence analysis

Compare training curves:

- train loss
- validation loss
- train accuracy
- validation accuracy
- validation macro F1-score if available

Questions to answer:

- Which model converges fastest?
- Which model overfits earliest?
- Does validation loss continue improving after accuracy plateaus?
- Does EfficientNetV2-S justify its higher GFLOPS?

### 9.5 Explainability analysis

After selecting the best model, add visual explanation.

Recommended method:

- Grad-CAM for CNN-like models

Questions to answer:

- Does the model focus on leaves, stems, fruit, or background?
- Are wrong predictions caused by background bias?
- Does the model focus differently for PlantVillage vs Leafsnap images?

This analysis can strengthen the internship report because it connects model performance to visual evidence.

## 10. Expected Outcomes

### 10.1 If EfficientNetV2-S wins

Likely interpretation:

- EfficientNetV2-S has strong transfer learning ability.
- Parameter-efficient design works well for plant classification.
- Larger input size may help capture fine-grained leaf texture.

Report angle:

> EfficientNetV2-S provides the best trade-off between classification performance and model size for the 76-class plant dataset.

### 10.2 If ConvNeXt-Tiny wins

Likely interpretation:

- Modern ConvNet design improves over ResNet50.
- ConvNeXt-Tiny captures plant texture and shape better than the older ResNet design.
- It may provide a good balance between accuracy and runtime.

Report angle:

> ConvNeXt-Tiny shows that modern convolutional architectures remain highly competitive for plant classification without requiring Transformer-based models.

### 10.3 If ResNet50 still wins

Likely interpretation:

- ResNet50 may be sufficient for this dataset size and label structure.
- Newer models may need different hyperparameters.
- The dataset may be limited by class imbalance and label granularity rather than architecture.

Report angle:

> The benefit of newer architectures is constrained by dataset imbalance and mixed crop/genus label design.

## 11. Implementation Plan

### 11.1 Add model builders

Update:

```text
plant_classifier/models.py
```

Add:

```python
from torchvision.models import EfficientNet_V2_S_Weights, ConvNeXt_Tiny_Weights
```

Add model names:

```text
efficientnet_v2_s
convnext_tiny
```

Implementation notes:

- EfficientNet classifier is usually in `model.classifier`.
- ConvNeXt classifier is also in `model.classifier`, but its structure differs from ResNet.
- ResNet uses `model.fc`, so do not reuse ResNet head replacement code directly.

### 11.2 Add learning rates

Update:

```text
plant_classifier/config.py
plant_classifier/training.py
```

Add:

```text
learning_rate_efficientnet_v2_s = 1e-4
learning_rate_convnext_tiny = 1e-4
```

### 11.3 Add CLI examples

EfficientNetV2-S only:

```bash
python scripts/train_kaggle.py --epochs 50 --batch-size 16 --models efficientnet_v2_s
```

ConvNeXt-Tiny only:

```bash
python scripts/train_kaggle.py --epochs 50 --batch-size 16 --models convnext_tiny
```

Both new models:

```bash
python scripts/train_kaggle.py --epochs 50 --batch-size 16 --models efficientnet_v2_s convnext_tiny
```

Full benchmark:

```bash
python scripts/train_kaggle.py --epochs 50 --batch-size 16 --models custom_cnn resnet50_fine_tuning efficientnet_v2_s convnext_tiny
```

### 11.4 Update outputs

The current output structure can stay the same:

```text
plant_training_outputs/
|-- efficientnet_v2_s/
|   |-- best_model.pth
|   |-- history.csv
|   |-- history.png
|   |-- confusion_matrix.png
|   `-- summary.json
|-- convnext_tiny/
|   |-- best_model.pth
|   |-- history.csv
|   |-- history.png
|   |-- confusion_matrix.png
|   `-- summary.json
|-- all_models_summary.csv
|-- all_models_summary.json
`-- model_comparison.png
```

## 12. Suggested Report Structure

Use this structure for the next report chapter:

1. Motivation
   - Current ResNet50 baseline is complete.
   - Need to test modern classification architectures.

2. Model selection
   - EfficientNetV2-S: parameter-efficient CNN.
   - ConvNeXt-Tiny: modernized ConvNet.

3. Experimental setup
   - Dataset
   - Split
   - Training configuration
   - Hardware/Kaggle environment

4. Quantitative results
   - Accuracy
   - Macro F1
   - Runtime
   - Best validation accuracy

5. Class-level analysis
   - Crop vs tree-genus performance
   - Minority-class behavior

6. Error analysis
   - Confusion matrix
   - Most confused classes

7. Explainability
   - Grad-CAM examples for correct and incorrect predictions

8. Conclusion
   - Best model
   - Practical recommendation
   - Limitations
   - Future work

## 13. Decision

Recommended next action:

1. Add `efficientnet_v2_s` and `convnext_tiny` to the modular Python pipeline.
2. Run a smoke test with 1 epoch.
3. Run each new model separately on Kaggle.
4. Compare them against the existing ResNet50 results.
5. Write a deeper analysis focused on macro F1, confusion matrix, and crop-vs-genus behavior.

Do not add Swin Transformer yet unless the EfficientNetV2-S and ConvNeXt-Tiny results are already stable.

## 14. References

- Tan, M., & Le, Q. V. EfficientNetV2: Smaller Models and Faster Training. arXiv: https://arxiv.org/abs/2104.00298
- Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. A ConvNet for the 2020s. arXiv: https://arxiv.org/abs/2201.03545
- TorchVision `efficientnet_v2_s` documentation: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_s.html
- TorchVision `convnext_tiny` documentation: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html
