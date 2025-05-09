# cpeptide_code
# 📊 Predicting Stimulated C-Peptide in Type 1 Diabetes Using Machine Learning: A Web-Based Tool from the T1D Exchange Registry

This repository contains Python code for applying a **Random Forest machine learning model** on a diabetes-related dataset to predict an ordered categorical outcome (`PEAK_K_3`).

---

## 📦 Dataset Overview

The dataset includes the following columns (original names in Turkish, English translations provided):

* `diyabet_tani_yasi` → **Age at diabetes diagnosis**
* `diyabet_suresi` → **Diabetes duration**
* `GLU` → **Non-fasting plasma glucose**
* `A1C` → **HbA1c level**
* `CPEP` → **Non-fasting C-peptide level**
* `PEAK_K_3` → **Outcome class (ordered categorical variable)**

---

## ⚙️ Workflow

1️⃣ **Data Preprocessing**

* Load data and drop rows with missing outcome (`PEAK_K_3`)
* Apply `RobustScaler` to numeric features to reduce the influence of outliers

2️⃣ **Model Development**

* Use `RandomForestClassifier` from scikit-learn
* Optimize hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`) using `GridSearchCV` (5-fold stratified cross-validation)

3️⃣ **Model Evaluation**

* Apply 10-fold stratified cross-validation on the training set
* Calculate:

  * Accuracy
  * Area Under the Curve (AUC, one-vs-rest)
  * Recall (Sensitivity, macro-average)
  * Precision (macro-average)
  * F1 Score (macro-average)
  * Cohen’s Kappa
  * Matthews Correlation Coefficient (MCC)
  * Specificity (computed from confusion matrix)

4️⃣ **Test Set Performance**

* Evaluate the final optimized model on the hold-out test set (30% split)

---

## 🏁 Final Output

The script prints a **summary table** showing:

| Metric | 10-Fold CV (Mean ± 95% CI) | Test Set Result |
| ------ | -------------------------- | --------------- |

---

## 🛠 Requirements

* Python
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn

---

## 💡 Notes

* The script is designed to run both locally and on Google Colab (you can adapt the file paths accordingly).
* Random seeds are fixed (`random_state=42`) for reproducibility.
* `RobustScaler` is used because it scales based on the median and interquartile range, making it more robust to extreme values.

---

## 📈 How to Run

1️⃣ Place your data file as `data.xlsx` in the working directory.
2️⃣ Run the Python script provided in this repository.
3️⃣ Review the printed results for optimized model performance on both cross-validation and test data.
