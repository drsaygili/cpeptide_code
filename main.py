import numpy as np
import pandas as pd

# Initial data loading and preparation
df = pd.read_excel("data.xlsx")
selected_columns_rfe = ['diyabet_tani_yasi', 'diyabet_suresi', 'GLU', 'A1C', 'CPEP', 'PEAK_K_3']
# Variable names translations:
# diyabet_tani_yasi = diabetes_diagnosis_age
# diyabet_suresi = diabetes_duration
# GLU = non-fasting Glucose
# A1C = Hemoglobin A1C
# CPEP = non-fasting C-Peptide
# PEAK_K_3 = Peak C-Peptide category after MMTT stimulation

df_rfe = df[selected_columns_rfe]
df_rfe = df_rfe.dropna()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, cohen_kappa_score, matthews_corrcoef, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import time

# 🔹 Data Preparation

df = df_rfe.copy()
df = df[df["PEAK_K_3"].notnull()]

X = df.drop("PEAK_K_3", axis=1)
y = pd.Series(pd.Categorical(df["PEAK_K_3"], ordered=True), index=df.index)

# 🔹 Numerical variables (RobustScaler)

numeric_features = X.columns.tolist()
scaler = RobustScaler()
X_numeric_scaled = pd.DataFrame(scaler.fit_transform(X[numeric_features]), columns=numeric_features, index=X.index)

# 🔹 Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X_numeric_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# 🔹 Metric calculation functions

def mean_ci(vals):
    mean = np.mean(vals)
    se = np.std(vals, ddof=1)/np.sqrt(len(vals))
    return f"{mean:.2f} ({mean - 1.96*se:.2f} – {mean + 1.96*se:.2f})"

def compute_sens_spec_multiclass(cm):
    sensitivities, specificities = [], []
    for i in range(cm.shape[0]):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        sens = TP / (TP + FN) if (TP+FN)>0 else 0
        spec = TN / (TN + FP) if (TN+FP)>0 else 0
        sensitivities.append(sens)
        specificities.append(spec)
    return np.mean(sensitivities), np.mean(specificities)

# 🔹 Hyperparameter grid search

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# 🔹 Hyperparameter optimization with GridSearchCV

start_time = time.time()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Hyperparameter optimization time: {(time.time() - start_time) / 60:.2f} minutes")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best F1 score (CV): {grid_search.best_score_:.4f}")

# 🔹 10-Fold Cross-Validation with the best model

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

metrics_cv = {"Accuracy": [], "AUC": [], "Recall": [], "Precision": [],
              "F1 Score": [], "Kappa": [], "MCC": [], "Specificity": []}

best_model = RandomForestClassifier(**grid_search.best_params_, random_state=42)

for train_idx, val_idx in cv.split(X_train, y_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    best_model.fit(X_tr, y_tr)
    
    y_pred_prob = best_model.predict_proba(X_val)
    y_pred = best_model.predict(X_val)
    
    metrics_cv["Accuracy"].append(accuracy_score(y_val, y_pred))
    metrics_cv["AUC"].append(roc_auc_score(pd.get_dummies(y_val), y_pred_prob, multi_class='ovr', average='macro'))
    metrics_cv["Recall"].append(recall_score(y_val, y_pred, average='macro'))
    metrics_cv["Precision"].append(precision_score(y_val, y_pred, average='macro'))
    metrics_cv["F1 Score"].append(f1_score(y_val, y_pred, average='macro'))
    metrics_cv["Kappa"].append(cohen_kappa_score(y_val, y_pred))
    metrics_cv["MCC"].append(matthews_corrcoef(y_val, y_pred))
    _, spec = compute_sens_spec_multiclass(confusion_matrix(y_val, y_pred))
    metrics_cv["Specificity"].append(spec)

# 🔹 Test set evaluation

final_model = RandomForestClassifier(**grid_search.best_params_, random_state=42)
final_model.fit(X_train, y_train)

y_test_pred_prob = final_model.predict_proba(X_test)
y_test_pred = final_model.predict(X_test)

metrics_test = {
    "Accuracy": accuracy_score(y_test, y_test_pred),
    "AUC": roc_auc_score(pd.get_dummies(y_test), y_test_pred_prob, multi_class='ovr', average='macro'),
    "Recall": recall_score(y_test, y_test_pred, average='macro'),
    "Precision": precision_score(y_test, y_test_pred, average='macro'),
    "F1 Score": f1_score(y_test, y_test_pred, average='macro'),
    "Kappa": cohen_kappa_score(y_test, y_test_pred),
    "MCC": matthews_corrcoef(y_test, y_test_pred),
    "Specificity": compute_sens_spec_multiclass(confusion_matrix(y_test, y_test_pred))[1]
}

# 🔹 Results table

df_final = pd.DataFrame({
    "Metric": metrics_test.keys(),
    "Training (CV 10-Fold)": [mean_ci(metrics_cv[m]) for m in metrics_test.keys()],
    "Test": [f"{val:.2f}" for val in metrics_test.values()]
})

print("\nResults for the optimized RF model:")
print(df_final.set_index("Metric"))
