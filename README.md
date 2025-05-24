
# Hybrid Deep Learning Image Classifier

This project implements a **hybrid image classification system** for vegetables using **ResNet50** and **InceptionV3** backbones with **Focal Loss**, **transfer learning**, and a **soft voting ensemble** strategy.

---

## 🔍 Overview

- Uses **transfer learning** from pre-trained ResNet50 and InceptionV3 models.
- Applies **data augmentation** via `ImageDataGenerator`.
- Custom **Focal Loss** for handling class imbalance.
- Combines individual models into a **hybrid model** by feature concatenation.
- Implements **soft voting ensemble** to improve final predictions.
- Visualizes accuracy, loss, and confusion matrix.

---

## 🧠 Model Architecture

- **ResNet50 and InceptionV3** are used as frozen feature extractors.
- After training, the last 50 layers are unfrozen for **fine-tuning**.
- The **hybrid model** merges both backbones’ outputs.
- **Focal loss** is used to combat class imbalance in multiclass classification.

---

## 🗂 Dataset Structure

Expected dataset folder format:

```
dataset/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

Update the dataset path in the script:
```python
train_dataset_path = "E:/BTP_sem8/dataset/vegetable_data_modified/train"
test_dataset_path = "E:/BTP_sem8/dataset/vegetable_data_modified/test"
```

---

## ⚙️ Dependencies

Ensure you have the following packages installed:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

---

## 🚀 Running the Project

1. Clone the repository.
2. Place your dataset in the specified directory structure.
3. Run the script:
```bash
python hybrid_classifier.py
```

---

## 📊 Outputs

- Trained model weights:
  - `resnet_model.h5`
  - `inception_model.h5`
  - `resnet_fine_tuned_model.h5`
  - `inception_fine_tuned_model.h5`
  - `final_hybrid_model.h5`

- Evaluation metrics:
  - Classification report
  - Confusion matrix
  - Accuracy and loss plots

---

## 📈 Visualizations

- Confusion matrix to visualize model performance.
- Training and validation accuracy/loss plots for all models.

---

## 📌 Notes

- Uses **categorical cross-entropy with focal loss** for imbalanced data.
- **Soft Voting Ensemble** improves generalization by averaging predictions from all models.
- Training time might be high depending on system specs.

---
