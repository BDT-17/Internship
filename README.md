# Phase 2: Plant Disease Classification - Implementation

Dự án so sánh hiệu suất giữa CNN trained from scratch và pretrained models (ResNet50) cho bài toán phân loại bệnh cây trồng.

## Cấu trúc dự án

```
internship/
├── Phase_1_Literature_Review_Report.md    # Báo cáo nghiên cứu lý thuyết
├── Phase_2_Plant_Classification.ipynb     # Notebook thực nghiệm chính
├── requirements.txt                        # Dependencies cần cài đặt
├── dataset/
│   ├── PlantVillage/                      # Dataset chính (15 classes)
│   └── leafsnap-dataset/                  # Dataset test cross-domain
└── guide/                                  # Tài liệu hướng dẫn
```

## Cài đặt môi trường

### 1. Kích hoạt virtual environment (nếu có)
```bash
cd d:/DS/internship
source venv/Scripts/activate  # hoặc venv\Scripts\activate trên Windows
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

**Lưu ý**: Nếu bạn có GPU NVIDIA, cài đặt PyTorch với CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Kiểm tra cài đặt
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## Chạy thực nghiệm

### Mở Jupyter Notebook
```bash
jupyter notebook Phase_2_Plant_Classification.ipynb
```

### Cấu trúc notebook

**Sections 1-5**: Setup và chuẩn bị
- Import libraries
- Load PlantVillage dataset (70/15/15 split)
- Data augmentation
- Model architectures (Scratch CNN, ResNet50 FE, ResNet50 FT)
- Training functions

**Sections 6-8**: Ba thực nghiệm chính
- Experiment 1: Custom CNN from scratch
- Experiment 2: ResNet50 Feature Extraction (frozen backbone)
- Experiment 3: ResNet50 Fine-tuning (progressive unfreezing)

**Sections 9-10**: Phân tích kết quả
- Comparison table
- Training curves visualization
- Confusion matrices
- Metrics comparison bar charts
- Hypothesis validation
- Conclusions

### Thời gian chạy dự kiến

| Model | Training Time (CPU) | Training Time (GPU) |
|-------|---------------------|---------------------|
| Scratch CNN | 4-6 hours | 30-60 min |
| ResNet50 FE | 1-2 hours | 15-30 min |
| ResNet50 FT | 2-3 hours | 30-60 min |

## Dataset

### PlantVillage
- **Classes**: 15 (Pepper: 2, Potato: 3, Tomato: 10)
- **Total images**: ~10,000+
- **Format**: 256×256 RGB JPG
- **Condition**: Controlled lab environment

### LeafSnap (Cross-domain testing)
- **Species**: 185 tree species
- **Field images**: ~7,700 real-world photos
- **Purpose**: Test generalization to natural conditions

## Kết quả mong đợi

Dựa trên Phase 1 Literature Review:

| Metric | Scratch CNN | ResNet50 FE | ResNet50 FT |
|--------|-------------|-------------|-------------|
| Accuracy | 87-90% | 94-96% | 96-98% |
| F1-score | 0.86-0.89 | 0.93-0.95 | 0.95-0.97 |
| Training time | 4-6h | 15-30m | 1-2h |
| Generalization gap | 8-12% | 2-4% | 1-3% |

## Outputs

Sau khi chạy notebook, bạn sẽ có:
- `training_curves.png` - Biểu đồ loss và accuracy qua các epochs
- `confusion_matrices.png` - Ma trận nhầm lẫn cho 3 models
- `metrics_comparison.png` - So sánh metrics và training time
- Console output với detailed analysis và hypothesis validation

## Troubleshooting

### Lỗi CUDA out of memory
Giảm batch size trong cell 6:
```python
batch_size = 16  # thay vì 32
```

### Lỗi "No module named 'torch'"
```bash
pip install torch torchvision
```

### Dataset không tìm thấy
Kiểm tra đường dẫn trong cell 4:
```python
dataset_path = 'd:/DS/internship/dataset/PlantVillage'
```

## Tham khảo

- Phase 1 Literature Review: [Phase_1_Literature_Review_Report.md](Phase_1_Literature_Review_Report.md)
- Action Plan: [guide/ACTION_PLAN_Next_Steps_PLANT.txt](guide/ACTION_PLAN_Next_Steps_PLANT.txt)
- Dataset Overview: [guide/dataset_overview.md](guide/dataset_overview.md)

## Liên hệ

Nếu có vấn đề kỹ thuật, tham khảo:
- PyTorch Documentation: https://pytorch.org/docs/
- PlantVillage Dataset: https://www.kaggle.com/datasets/emmareid/plantvillage-dataset
