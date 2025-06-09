#Dry Bean Classification Using Machine Learning

This project applies supervised machine learning to classify seven types of dry beans using morphological features. The models compared include C5.0 Decision Tree, XGBoost, and Support Vector Machine (SVM) with a Radial Basis Function (RBF) kernel. The project is part of an academic coursework submission for DS7003.

---

##Project Structure

- `Script.R` – Main R script for data loading, preprocessing, model training, and evaluation.
- `DryBeanDataset.xlsx` – Dataset used for classification, obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset).
- `Dry-Beans-Classification` – Full project report with methodology, analysis, and results.

---

##Models Implemented

- **C5.0 Decision Tree** – Interpretable rule-based classifier with boosting.
- **XGBoost** – Gradient boosting framework known for performance and accuracy.
- **SVM (Radial Kernel)** – Effective for non-linear classification, though less effective in this context.

---

##Key Results

| Model              | Accuracy |
|-------------------|----------|
| XGBoost           | 92.87%   |
| C5.0 Decision Tree| 92.39%   |
| SVM (Radial)      | 66.80%   |

XGBoost achieved the best overall accuracy, with both tree-based models outperforming SVM in most class-wise metrics.

---

##How to Run

1. Open RStudio or your preferred R environment.
2. Place `Script.R` and `DryBeanDataset.xlsx` in the same working directory.
3. Run the script:
   ```r
   source("Script.R")
