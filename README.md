# TransCapsNet: Explainable Hybrid Deep Learning for Colorectal Cancer Classification

This project presents an explainable hybrid deep learning–based medical image
classification system for colorectal cancer and colon disease detection.
The proposed model, **TransCapsNet**, combines Convolutional Neural Networks (CNN)
and Capsule Networks (CapsNet) to achieve robust, accurate, and interpretable
predictions from medical images.

---

## 🎯 Project Objective

The primary objective of this project is to design an AI-based system capable of
automatically classifying colorectal and colon disease images to assist medical
diagnosis and academic research.

Key objectives include:
- Accurate classification of colon disease images
- Reduction of manual diagnostic effort
- Improved transparency using Explainable AI (XAI)
- Reliable and reproducible performance suitable for academic use

---

## 🗂️ Dataset Description

- Medical image dataset related to colon and colorectal diseases
- Images are organized into class-wise folders
- Dataset is split into training and testing sets
- Data preprocessing includes resizing, normalization, and augmentation

### Disease Classes:
- Normal
- Ulcerative Colitis
- Polyps
- Esophagitis

---

## 🧠 Model Used (Hybrid Architecture – TransCapsNet)

This project uses a **hybrid deep learning architecture called TransCapsNet**.

### Model Components:
- **Convolutional Neural Network (CNN)**  
  Extracts low-level and high-level spatial features from colon images.

- **Capsule Network (CapsNet)**  
  Preserves spatial relationships between features and improves robustness to
  rotation, scale, and viewpoint variations.

- **Fully Connected Layers**  
  Perform final classification into disease categories.

The hybrid CNN + CapsNet architecture improves feature representation and
generalization compared to traditional CNN-only models.

---

## 🔍 Explainable AI (XAI)

To ensure transparency, trust, and clinical interpretability, Explainable AI
techniques are integrated into this project.

### XAI Techniques Used:
- **Grad-CAM (Gradient-weighted Class Activation Mapping)**  
  Visualizes important image regions influencing the model’s predictions.

- **LIME (Local Interpretable Model-agnostic Explanations)**  
  Explains individual predictions by approximating the model locally.

- **SHAP (SHapley Additive exPlanations)**  
  Quantifies the contribution of input features to the final prediction.

These techniques help clinicians and researchers understand *why* a specific
disease class is predicted, increasing confidence in AI-assisted diagnosis.

---

## ⚙️ Training Details

- Framework: PyTorch
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Learning rate scheduling applied
- GPU acceleration supported (CUDA)
- Early stopping used to prevent overfitting

---

## 📊 Evaluation Metrics

The model performance is evaluated using multiple quantitative metrics and
validation strategies to ensure robustness and generalization.

### 🔹 Classification Metrics
- **Accuracy**
- **Precision**
- **Recall (Sensitivity)**
- **F1-score**
- **Confusion Matrix**

---

### 📈 ROC Curve and AUC Analysis
- **ROC Curve (Receiver Operating Characteristic Curve)**  
  Evaluates the trade-off between true positive rate and false positive rate.

- **AUC (Area Under the Curve)**  
  Measures the overall discriminative ability of the classifier.

---

### 📉 Training and Validation Curves
- Training Accuracy Curve
- Validation Accuracy Curve
- Training Loss Curve
- Validation Loss Curve

These curves are used to monitor convergence, detect overfitting, and analyze
training stability.

---

### 🔁 Cross-Validation Strategy
- **10-Fold Cross Validation**

The dataset is divided into 10 folds. The model is trained on 9 folds and tested
on the remaining fold. This process is repeated 10 times, and the final
performance is reported as the average across all folds.

---

## 🧪 Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- KAggle

