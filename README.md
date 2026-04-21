# 🩺 Abdominal Trauma Detection

A deep learning-based system for **automated detection and classification of abdominal injuries from CT scan images**. This project leverages Convolutional Neural Networks (CNNs) and transfer learning techniques to improve diagnostic accuracy and assist healthcare professionals.

---

## 📌 Overview

Abdominal trauma is a life-threatening condition that requires **rapid and accurate diagnosis**. Traditional diagnosis relies on manual interpretation of CT scans, which is time-consuming and dependent on radiologist expertise.

This project presents an **AI-powered solution** that:

* Automatically analyzes CT scan images
* Detects and classifies abdominal injuries
* Improves diagnostic speed and consistency

The system is designed to support **clinical decision-making and emergency care**. 

---

## 🎯 Problem Statement

Develop a robust deep learning model capable of:

* Detecting abdominal trauma in CT scans
* Identifying injuries in organs like liver, spleen, kidney, and bowel
* Classifying injury types and severity

The goal is to reduce diagnostic time and improve accuracy compared to manual analysis. 

---

## 🎯 Objectives

* Detect and localize abdominal injuries automatically
* Classify trauma into meaningful categories
* Improve diagnostic precision and consistency
* Reduce dependency on manual interpretation
* Enable faster clinical decision-making

---

## 🧠 Approach

The project uses **deep learning and computer vision techniques**:

### 🔹 Core Components

* Convolutional Neural Networks (CNNs)
* Transfer Learning
* Data Preprocessing & Augmentation

### 🔹 Models Implemented

* Custom CNN (Baseline)
* EfficientNetB1
* ResNet50

### 🔹 Data Augmentation

* Image transformations
* CutMix augmentation for better generalization

These approaches help the model learn complex patterns in CT scan images for accurate detection. 

---

## 🏗️ System Workflow

```
CT Scan Images
      ↓
Data Cleaning & Preprocessing
      ↓
Dataset Preparation
      ↓
Model Training
   ├── Custom CNN
   ├── EfficientNetB1
   └── ResNet50
      ↓
Validation & Evaluation
      ↓
Prediction on New Data
```

---

## 📂 Dataset

* Source: RSNA 2023 Abdominal Trauma Detection Dataset (Kaggle)
* Original dataset: ~1.5M images
* Used subset: ~84K images

### 🏷️ Labels Include:

* Bowel (Healthy / Injury)
* Extravasation (Healthy / Injury)
* Kidney (Healthy / Low / High)
* Liver (Healthy / Low / High)
* Spleen (Healthy / Low / High)

The dataset enables **multi-label classification across multiple organs**. 

---

## ⚙️ Tech Stack

* **Language:** Python
* **Framework:** PyTorch
* **Libraries:** NumPy, Pandas, Matplotlib, Scikit-learn
* **Tools:** Kaggle, Weights & Biases (wandb)

---

## 🧪 Training Pipeline

* Data cleaning and validation
* Dataset splitting (Train / Validation / Test)
* Data augmentation
* Model training with hyperparameter tuning
* Performance evaluation using multiple metrics

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* AUC-ROC Curve
* Confusion Matrix

---

## 📈 Results & Insights

* Transfer learning models (**EfficientNetB1, ResNet50**) outperform baseline CNN
* Data augmentation (CutMix) improves generalization
* Model shows strong performance in detecting multiple injury types

However, further improvements are required for:

* Better consistency across unseen data
* Real-world clinical deployment



---

## 🚀 Future Scope

* Use larger multi-center datasets
* Improve model robustness and generalization
* Real-time deployment in healthcare systems
* Integration with clinical workflows
* Explainable AI for better interpretability

---

## 👨‍💻 Contributors

* **Jayant Arsode**
* **Hemant Tekade**

---

## 📜 License

This project is developed for **academic and research purposes**.

---

## 🙏 Acknowledgements

* Yeshwantrao Chavan College of Engineering
* Dr. Lalit B. Damahe (Project Guide)
* RSNA Dataset (Kaggle)

---

## 💡 Conclusion

This project demonstrates the potential of **AI in medical imaging**, enabling faster, more accurate detection of abdominal trauma and supporting improved patient outcomes.

---
