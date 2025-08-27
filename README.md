# 🩺 Skin Lesion Segmentation using Deep Learning

This project focuses on **automatic skin lesion segmentation** using deep learning methods.  
Segmentation of skin lesions plays a crucial role in **computer-aided diagnosis (CAD)** of melanoma and other types of skin cancer.  
We leverage the **HAM10000 dataset**, apply **preprocessing (hair removal, resizing, normalization)**, and train **U-Net–based architectures** implemented with **Segmentation Models PyTorch (SMP)**.

---

## 📖 Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Preprocessing](#preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training Pipeline](#training-pipeline)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Installation](#installation)



---

## 🔍 Overview
- **Goal**: To accurately segment lesions from dermoscopic images for better diagnosis.
- **Frameworks**: PyTorch, Segmentation Models PyTorch (SMP), Albumentations, OpenCV.
- **Key Steps**:
  1. Data preprocessing (hair removal, resizing, normalization).
  2. Dataset creation with image-mask pairs.
  3. Model training using U-Net–based architectures.
  4. Evaluation with standard segmentation metrics.
  5. Visualization of results with overlays and confusion matrices.

---

## 📂 Dataset
We use the **HAM10000 dataset** (Human Against Machine with 10,000 training images).  
It includes:
- **Images**: Dermoscopic skin lesion images.
- **Segmentation Masks**: Ground-truth lesion masks.


---

## 🧹 Preprocessing
1. **Hair Removal (DullRazor Algorithm)**
   - Convert to grayscale.
   - Detect hair-like structures using edge detection (Canny).
   - Dilate + inpaint to remove hair artifacts.
2. **Resizing**
   - Images and masks resized to `256x256`.
3. **Normalization**
   - Pixel values scaled to `[0,1]`.
4. **Storage**
   - Preprocessed images and masks saved as `.npy` arrays for efficient loading.

---

## 🏗️ Model Architecture
- Implemented using **Segmentation Models PyTorch (SMP)**.
- Backbone architectures tested:
  - **U-Net**
  - **U-Net++**
  - **DeepLabV3+**
- Encoder options include ResNet34/50, EfficientNet, etc.
- Loss functions:
  - **Dice Loss**
  - **Binary Cross Entropy (BCE)**
  - Hybrid: `BCE + Dice`
- Optimizer: **Adam**
- Learning rate scheduling with **ReduceLROnPlateau**.

---

## 🏋️ Training Pipeline
1. **Dataset Splitting**
   - Train / Validation split (e.g., 80% / 20%).
2. **Augmentation** (using Albumentations)
   - Horizontal/Vertical flips
   - Random rotations
   - Brightness & contrast adjustments
   - Elastic transforms
3. **Batching & Dataloaders**
   - Pytorch `Dataset` and `DataLoader` classes for efficient training.
4. **Training**
   - Train for ~50 epochs.
   - Early stopping based on validation loss.

---

## 📊 Evaluation Metrics
We evaluate model performance using:
- **Intersection over Union (IoU)**
- **Dice Coefficient**
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- Confusion Matrix (pixel-wise)


---

## ⚙️ Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/skin-lesion-segmentation.git
cd skin-lesion-segmentation

pip install -r requirements.txt

