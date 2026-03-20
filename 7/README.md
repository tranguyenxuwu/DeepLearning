# Bài tập tuần 7 — CNN trên 3 tập dữ liệu

## Tổng quan

Xây dựng mô hình CNN tự thiết kế (không dùng pretrained) để phân loại ảnh trên 3 tập dữ liệu khác nhau.

| Dataset | Số lớp | Input | Kiến trúc | Mục tiêu |
|---------|--------|-------|-----------|----------|
| CIFAR-10 | 10 | 32×32 | 3-block CNN (32→64→128) | > 90% |
| Cat & Dog | 2 | 64×64 | 4-block CNN (32→64→128→256) | > 90% |
| PlantVillage | 38 | 64×64 | 4-block CNN (32→64→128→256) | > 90% |

## Cấu trúc thư mục

```
7/
├── CNN_CIFAR10.ipynb        # CIFAR-10 (auto download)
├── CNN_CatDog.ipynb         # Cat & Dog (Kaggle)
├── CNN_PlantVillage.ipynb   # PlantVillage (Kaggle)
└── README.md
```

## Chi tiết từng notebook

### 1. CNN_CIFAR10.ipynb

- **Dữ liệu**: CIFAR-10 (60,000 ảnh, 10 lớp), tự tải qua `torchvision.datasets`
- **Kiến trúc**: 3 khối Conv (32→64→128) + FC, ~200K params
- **Xử lý mất cân bằng**: Dữ liệu cân bằng sẵn, không cần xử lý
- **Augmentation**: RandomCrop, RandomHorizontalFlip
- **Training**: Adam + ReduceLROnPlateau, 30 epochs
- **Đánh giá**: Confusion matrix, classification report, feature map visualization

### 2. CNN_CatDog.ipynb

- **Dữ liệu**: Dogs vs Cats (Kaggle), cần tải thủ công
- **Kiến trúc**: 4 khối Conv (32→64→128→256) + GAP + FC, ~700K params
- **Xử lý mất cân bằng**: WeightedRandomSampler
- **Augmentation**: RandomHorizontalFlip, RandomRotation, ColorJitter
- **Training**: Adam + ReduceLROnPlateau, 30 epochs
- **Đánh giá**: Confusion matrix, feature maps, hiển thị dự đoán sai

### 3. CNN_PlantVillage.ipynb

- **Dữ liệu**: PlantVillage (Kaggle), 38 lớp bệnh lá cây
- **Kiến trúc**: 4 khối Conv (32→64→128→256) + GAP + FC(256→38), ~800K params
- **Xử lý mất cân bằng**: WeightedRandomSampler + class weights trong CrossEntropyLoss
- **Augmentation**: RandomFlip (H+V), RandomRotation, ColorJitter mạnh
- **Training**: Adam + ReduceLROnPlateau, 30 epochs
- **Đánh giá**: Confusion matrix 38×38, accuracy theo từng lớp, feature maps

## Kỹ thuật chung

| Kỹ thuật | Mục đích |
|----------|----------|
| BatchNorm | Ổn định gradient, tăng tốc convergence |
| Dropout / Dropout2d | Giảm overfitting |
| Data Augmentation | Tăng đa dạng dữ liệu, giảm overfitting |
| WeightedRandomSampler | Xử lý mất cân bằng lớp |
| ReduceLROnPlateau | Giảm learning rate khi accuracy không tăng |
| GlobalAvgPool (GAP) | Giảm params so với Flatten, tránh overfitting |
