Plant Classification: Scratch vs Pretrained Models
Experimental Design Foundation

1. Problem Definition and Scope Clarification

1.1 Plant Classification in Computer Vision

Plant classification is a broad computer vision task that assigns plant-related images to predefined categories such as crop type, tree genus, species, or cultivar. In this internship project, the task is defined as broad plant image classification over crop-level and genus-level labels.

PlantVillage is used as one source of crop leaf images, with its original subcategory labels collapsed into crop-level labels: Pepper, Potato, and Tomato. Leafsnap contributes tree images grouped by genus. The final task is therefore a 76-class crop/genus classification problem.

1.2 Research Objective

The objective of this study is to systematically compare convolutional neural networks trained from scratch with pretrained CNNs for plant image classification, focusing on:
- classification accuracy
- macro precision, recall, and F1-score
- generalization ability
- training efficiency
- robustness under class imbalance

2. Dataset Foundation Analysis

2.1 PlantVillage Source Images

Dataset characteristics used in this project:
- local subset: 20,639 images
- source folders: 15
- crop groups after restructuring: Pepper, Potato, Tomato
- acquisition: controlled leaf images with relatively clean backgrounds

Role in this project:
- provides crop leaf images for the merged plant classification dataset
- contributes three crop-level classes after label restructuring
- supports controlled-environment training examples

2.2 Leafsnap Dataset

Dataset characteristics:
- approximately 30,000 images
- 185 plant species
- two domains:
  - lab images with controlled backgrounds
  - field images captured in natural outdoor conditions

Role in this project:
- contributes tree categories grouped by genus
- introduces real-world variation in lighting, occlusion, background, and viewpoint
- expands the classification task beyond crop leaves

2.3 Final Dataset Selection Rationale

The final dataset combines PlantVillage crop groups and Leafsnap genus groups into a 76-class plant classification benchmark. This creates a larger and more diverse dataset than a crop-only setup, while keeping the label space manageable for an internship project.

Key dataset implications:
- the task is crop/genus plant classification
- labels mix crop-level and genus-level categories
- class imbalance is significant and must be reported
- macro F1-score is important because accuracy alone can hide poor minority-class performance

3. Transfer Learning Fundamentals

3.1 Feature Transferability

Deep CNNs learn hierarchical representations:
- low-level layers: edges, textures, color gradients
- mid-level layers: shapes and object parts
- high-level layers: task-specific visual semantics

For plant images, pretrained CNNs can reuse general visual features such as leaf edges, venation-like textures, shapes, and color patterns. Fine-tuning allows the model to adapt these general representations to the 76-class plant classification task.

3.2 Transfer Learning Strategies

Strategy 1: Feature extraction
- freeze pretrained backbone
- train only the classifier head
- fast convergence but limited domain adaptation

Strategy 2: Fine-tuning
- update selected deeper layers and classifier head
- better adaptation to plant images
- requires careful learning-rate control and regularization

Strategy 3: Full fine-tuning
- update all layers
- can improve accuracy when enough data is available
- more expensive and more prone to overfitting

3.3 Why Transfer Learning Works for Plant Classification

ImageNet-pretrained models have learned broad visual features from natural images. These features transfer well to plant classification because the task depends on shape, texture, color, and object-level visual cues. Compared with a CNN trained from scratch, pretrained models usually converge faster and generalize better.

4. Comparative Study Motivation

4.1 Model Families

The project currently compares:
- custom CNN trained from scratch
- ResNet50 feature extraction
- ResNet50 fine-tuning

Recommended next model additions:
- EfficientNet-B0 for parameter-efficient transfer learning
- DenseNet121 for dense feature reuse comparison

4.2 Expected Comparison Axes

The comparison should cover:
- validation and test accuracy
- macro precision, recall, and F1-score
- top-5 accuracy
- confusion matrix patterns
- per-class performance
- training time
- model size and deployment practicality

5. Experimental Design Framework

5.1 Fair Comparison Principles

To ensure validity:
- use identical train/validation/test splits
- use the same augmentation pipeline where possible
- report the same metrics for all models
- use early stopping consistently
- evaluate the selected best checkpoint on the held-out test set

5.2 Research Hypotheses

H1: Pretrained models outperform the custom CNN baseline in validation and test accuracy.

H2: Fine-tuning outperforms feature extraction because deeper layers adapt to plant-specific visual patterns.

H3: Pretrained models converge faster than the custom CNN trained from scratch.

H4: Pretrained models achieve stronger macro F1-score, especially on minority classes.

6. Expected Outcomes

Expected pattern:
- custom CNN provides a useful baseline and interpretability point
- feature extraction provides a strong and efficient transfer learning baseline
- fine-tuned pretrained models provide the highest accuracy
- EfficientNet or DenseNet may improve the best-model comparison beyond ResNet50

7. Phase 2 Implementation Roadmap

Current implementation:
- custom CNN from scratch
- ResNet50 feature extraction
- ResNet50 fine-tuning
- stratified split
- training history and model comparison outputs

Recommended next implementation:
- add EfficientNet-B0
- add DenseNet121
- run final test-set evaluation for all models
- add class imbalance analysis
- add per-class metrics and confusion matrix discussion
- compare custom CNN against the best pretrained model in depth

8. Conclusion

This Phase 1 report establishes the theoretical and empirical foundation for comparing scratch-trained and pretrained CNNs in plant classification. The project is framed as a 76-class crop/genus classification task built from PlantVillage and Leafsnap sources. Phase 2 should empirically validate the expected benefits of transfer learning through controlled experiments, test-set evaluation, and class-level error analysis.

9. References

Hughes, D. P., & Salathe, M. (2015). An open access repository of images on plant health to aid in crop image research. arXiv:1511.08060.

Kumar, N., et al. (2012). LeafSnap: A Computer Vision System for Automatic Plant Species Identification. ECCV.

Yosinski, J., et al. (2014). How transferable are features in deep neural networks? NeurIPS.

Akhand, A. S., et al. (2025). A comparative study of custom CNNs, pre-trained models, and transfer learning approaches for plant image classification.
