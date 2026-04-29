Plant Classification: Scratch vs Pretrained Models
Experimental Design Foundation

1. Problem Definition & Scope Clarification
1.1 Plant Classification in Computer Vision
Plant classification is a broad computer vision task that involves assigning plant-related images to predefined categories. Depending on the granularity, it includes:
Plant species classification (e.g., LeafSnap)
Plant disease classification (e.g., PlantVillage)
In this project, plant disease classification is selected as a fine-grained plant classification problem, where images of the same plant species must be distinguished based on subtle disease-related visual patterns.
1.2 Research Objective
The objective of this study is to systematically compare convolutional neural networks (CNNs) trained from scratch with pretrained CNNs for plant image classification, focusing on:
Classification accuracy
Generalization ability
Training efficiency
Robustness to dataset bias

2. Dataset Foundation Analysis
2.1 PlantVillage Dataset (Hughes & Salathé, 2015)
Dataset Characteristics
Total images: 54,306
Classes: 38 plant disease categories
Plant species: 14 (tomato, potato, corn, grape, apple, etc.)
Image size: 256×256 RGB
Acquisition: Controlled laboratory conditions
Key Properties
Uniform backgrounds and standardized lighting
High image quality and low noise
Open-access and widely used for benchmarking
Implications for Plant Classification
Enables fair comparison between learning strategies
Introduces controlled-environment bias, limiting real-world generalization
Suitable for evaluating feature learning efficiency rather than deployment readiness

2.2 LeafSnap Dataset (Kumar et al., 2012)
Dataset Characteristics
~30,000 images
185 plant species
Two domains:
Lab images (controlled background)
Field images (natural conditions)
Critical Observation: Lab vs Field Gap
Field images introduce variations in lighting, occlusion, and leaf damage
Performance typically drops by 5–10% when models trained on lab data are tested on field images
Role in This Project
LeafSnap field images are used to evaluate cross-domain generalization of plant classification models

2.3 Dataset Selection Rationale
Criterion
PlantVillage
LeafSnap
Task complexity
Medium–High
High
Dataset size
Large (54K)
Medium (30K)
Benchmark usage
Extensive
Moderate
Field realism
Low
High

Decision:
PlantVillage as the primary training and benchmarking dataset
LeafSnap field images for generalization assessment

3. Transfer Learning Fundamentals
3.1 Feature Transferability (Yosinski et al., 2014)
Deep CNNs learn hierarchical representations:
Low-level layers: edges, textures, color gradients (highly transferable)
Mid-level layers: shapes and object parts (moderately transferable)
High-level layers: task-specific semantics (low transferability)
Transferability decreases as domain distance increases:
\text{Transferability} \propto \frac{1}{1 + \text{domain\_distance}}

3.2 Transfer Learning Strategies
Strategy 1: Feature Extraction
Freeze pretrained backbone
Train only task-specific classifier
Fast convergence, limited adaptation
Strategy 2: Fine-tuning (Progressive Unfreezing)
Freeze early layers
Fine-tune deeper layers with lower learning rates
Best balance between accuracy and generalization
Strategy 3: Full Fine-tuning
Update all layers
Requires large datasets and careful regularization

3.3 Why Transfer Learning Works for Plant Classification
ImageNet contains diverse natural images, including plants
Low-level visual features align well with leaf textures and disease patterns
Pretrained weights act as a strong inductive bias
Faster convergence and reduced overfitting

4. State-of-the-Art Comparative Study
4.1 Reference Study: Akhand et al. (2025)
Experimental Setup
Datasets: PlantVillage, LeafSnap
Models: Custom CNNs, VGG16, ResNet50, InceptionV3, MobileNetV2
Metrics: Accuracy, Precision, Recall, F1-score
Key Results
Approach
PlantVillage Accuracy
LeafSnap Accuracy
Training Time
Scratch CNN
~87%
~72%
4–6 hours
Feature Extraction
~94%
~85%
15–30 min
Fine-tuning
96–97%
~90%
1–2 hours


4.2 Generalization Gap Analysis
\text{Generalization Gap} = \text{Train Accuracy} - \text{Test Accuracy}
Scratch models: 8–12%
Feature extraction: 2–4%
Fine-tuning: 1–3%
Interpretation: Pretrained models generalize significantly better due to implicit regularization.

5. Experimental Design Framework
5.1 Fair Comparison Principles
To ensure validity:
Identical data splits (70/15/15, stratified)
Same augmentation pipeline
Matched architecture depth and parameter count
Same optimizer and early stopping criteria
Multiple runs with different random seeds

5.2 Research Hypotheses
H1: Pretrained models outperform scratch models in accuracy
H2: Pretrained models exhibit smaller generalization gaps
H3: Pretrained models converge faster
H4: Pretrained models achieve higher F1-scores on minority classes

6. Expected Outcomes
Metric
Scratch CNN
Pretrained (FE)
Pretrained (FT)
Accuracy
87–90%
94–96%
96–98%
F1-score
0.86–0.89
0.93–0.95
0.95–0.97
Training time
4–6h
15–30m
1–2h
Overfitting risk
High
Low
Low


7. Phase 2 Implementation Roadmap
Implement custom CNN trained from scratch
Implement pretrained CNNs with feature extraction
Apply fine-tuning with progressive unfreezing
Evaluate using multiple metrics and confusion matrices
Test cross-dataset generalization on LeafSnap field images

8. Conclusion
This Phase 1 report establishes a solid theoretical and empirical foundation for comparing scratch-trained and pretrained CNNs in plant classification. Literature evidence strongly indicates that transfer learning provides superior accuracy, efficiency, and generalization, particularly for fine-grained plant image classification tasks.
Phase 2 will empirically validate these findings through controlled experiments and comprehensive evaluation.

9. References
Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to aid in crop disease diagnosis. arXiv:1511.08060.
Kumar, N., et al. (2012). LeafSnap: A Computer Vision System for Automatic Plant Species Identification. ECCV.
Yosinski, J., et al. (2014). How transferable are features in deep neural networks? NeurIPS.
Akhand, A. S., et al. (2025). A comparative study of custom CNNs, pre-trained models, and transfer learning approaches for plant disease classification. Journal of Agricultural AI.

