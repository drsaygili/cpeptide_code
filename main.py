import numpy as np
import pandas as pd

# Initial data loading and preparation
df = pd.read_excel("data.xlsx")
# Variable names translations:
# diyabet_tani_yasi = diabetes_diagnosis_age
# diyabet_suresi = diabetes_duration
# GLU = non-fasting Glucose
# A1C = Hemoglobin A1C
# CPEP = non-fasting C-Peptide
# PEAK_K_3 = Peak C-Peptide category after MMTT stimulation

selected_columns = ['diyabet_tani_yasi', 'diyabet_suresi', 'GLU', 'A1C', 'CPEP', 'PEAK_K_3', 'DKAMeetCritEver2', 'SHSeizLoseConscEver2', 'duz_totalbasal_kg_K','GENDER','BKI']

numeric_features = ['diyabet_tani_yasi', 'diyabet_suresi', 'GLU', 'A1C', 'CPEP', 'BKI']

# Create DataFrame with selected columns
df_selected_columns = df[selected_columns]

# Categorical variable list
categorical_columns = ['GENDER', 'duz_totalbasal_kg_K', 'DKAMeetCritEver2', 'SHSeizLoseConscEver2', 'PEAK_K_3']


# Import required libraries
from sklearn.impute import SimpleImputer

# Check missing data percentages
missing_percentages = df_selected_columns.isnull().mean() * 100
print("Missing data percentage for each column:")
print(missing_percentages)

# Identify columns with missing values for imputation
numeric_cols = [col for col in numeric_features if df_selected_columns[col].isnull().any()]
categorical_cols = [col for col in categorical_columns if df_selected_columns[col].isnull().any()]

print(f"\nNumeric columns for median imputation: {numeric_cols}")
print(f"Categorical columns for mode imputation: {categorical_cols}")

# Apply median imputation to numeric variables
if numeric_cols:
    print("\nApplying median imputation to numeric columns with missing values")
    median_imputer = SimpleImputer(strategy='median')
    df_selected_columns.loc[:, numeric_cols] = median_imputer.fit_transform(df_selected_columns[numeric_cols])

# Apply mode imputation to categorical variables
if categorical_cols:
    print("\nApplying mode imputation to categorical columns with missing values")
    mode_imputer = SimpleImputer(strategy='most_frequent')
    df_selected_columns.loc[:, categorical_cols] = mode_imputer.fit_transform(df_selected_columns[categorical_cols])

# Check if any missing values remain
missing_after = df_selected_columns.isnull().sum().sum()
print(f"\nRemaining missing values after imputation: {missing_after}")

# Ensure the target variable (PEAK_K_3) has no missing values
if df_selected_columns["PEAK_K_3"].isnull().sum() > 0:
    print("Warning: Target variable still has missing values. Removing these rows.")
    df_selected_columns = df_selected_columns[df_selected_columns["PEAK_K_3"].notnull()]

# Feature selection
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier

# Create a copy of the dataframe
new_df2 = df_selected_columns.copy()

#  Target variable
target = 'PEAK_K_3'

# --- Feature Selection ---
X = new_df2.drop(columns=[target])
y = new_df2[target]

# Random Forest Feature Importances
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X, y)
rf_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nRandom Forest Feature Importance:\n", rf_importance)

# Recursive Feature Elimination (RFE)
rfe = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=5)
rfe.fit(X, y)
rfe_selected = X.columns[rfe.support_]
print("\nFeatures selected by RFE:\n", rfe_selected)

# Selected features from RFE

selected_features = ['diyabet_tani_yasi', 'diyabet_suresi', 'GLU', 'A1C', 'CPEP', 'PEAK_K_3']
df_rfe = df_selected_columns[selected_features]

# Model building and evaluation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, roc_auc_score
import time

# 🔹 Data Preparation
df = df_rfe.copy()
df = df[df["PEAK_K_3"].notnull()]

X = df.drop("PEAK_K_3", axis=1)
y = pd.Series(pd.Categorical(df["PEAK_K_3"], ordered=True), index=df.index)

# 🔹 Scale numerical variables with RobustScaler
numeric_features = X.columns.tolist()
scaler = RobustScaler()
X_numeric_scaled = pd.DataFrame(scaler.fit_transform(X[numeric_features]), 
                               columns=numeric_features, index=X.index)

# 🔹 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_numeric_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# 🔹 Metric calculation functions
def mean_ci(vals):
    """Calculate mean and 95% confidence interval"""
    mean = np.mean(vals)
    se = np.std(vals, ddof=1)/np.sqrt(len(vals))
    return f"{mean:.2f} ({mean - 1.96*se:.2f} – {mean + 1.96*se:.2f})"

def compute_sens_spec_multiclass(cm):
    """Compute sensitivity and specificity for multiclass classification"""
    sensitivities, specificities = [], []
    for i in range(cm.shape[0]):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        sens = TP / (TP + FN) if (TP+FN) > 0 else 0
        spec = TN / (TN + FP) if (TN+FP) > 0 else 0
        sensitivities.append(sens)
        specificities.append(spec)
    return np.mean(sensitivities), np.mean(specificities)

# 🔹 Hyperparameter grid search 
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

# 🔹 Define custom AUC scorer for multi-class classification
from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelBinarizer

def multiclass_auc_scorer(estimator, X, y):
    y_pred_proba = estimator.predict_proba(X)
    lb = LabelBinarizer()
    lb.fit(y)
    y_bin = lb.transform(y)
    return roc_auc_score(y_bin, y_pred_proba, multi_class='ovr', average='macro')

auc_scorer = make_scorer(multiclass_auc_scorer, needs_proba=True)

# 🔹 Hyperparameter optimization with GridSearchCV - optimized for speed and AUC
start_time = time.time()
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced from 5 to 3 splits for speed

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring=auc_scorer,  # Using AUC for scoring
    n_jobs=-1,
    verbose=1)

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
    metrics_cv["AUC"].append(roc_auc_score(pd.get_dummies(y_val), y_pred_prob, 
                                          multi_class='ovr', average='macro'))
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
    "AUC": roc_auc_score(pd.get_dummies(y_test), y_test_pred_prob, 
                        multi_class='ovr', average='macro'),
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

