import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix)

# SMOTE Import
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("Error: Library not found. Run 'pip install imbalanced-learn' in terminal.")
    exit()

# ==========================================
# 1. LOAD ORIGINAL DATA
# ==========================================
# We use the original file. We do NOT use the pre-balanced CSV.
try:
    df = pd.read_csv("Flood_Datasets.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'Flood_Datasets.csv' not found.")
    exit()

# ==========================================
# 2. PREPROCESSING
# ==========================================
# One-hot encode Location if it exists
if 'Location' in df.columns:
    df = pd.get_dummies(df, columns=["Location"], drop_first=True)

X = df.drop(columns=["FloodOccurrence", "Date"], errors='ignore')
y = df["FloodOccurrence"]

# ==========================================
# 3. THE GOLDEN RULE (Split First!)
# ==========================================
# We split the data BEFORE creating any fake samples.
# This ensures the Test Set is 100% real and has never "met" the training data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nOriginal Training Data: {len(X_train)} rows")
print(f"Test Data (Locked):     {len(X_test)} rows")

# ==========================================
# 4. APPLY SMOTE (Only to Training Data)
# ==========================================
print("\n--- Applying SMOTE to Training Data ---")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"New Training Size:      {len(X_train_smote)} rows (Balanced 50/50)")
print("Note: The Test Data was NOT touched by SMOTE.")

# ==========================================
# 5. DEFINE PIPELINES
# ==========================================
# Logistic Regression
log_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(random_state=42, max_iter=1000))
])

# Random Forest
rf_pipeline = Pipeline([
    ("model", RandomForestClassifier(n_estimators=200, random_state=42))
])

# SVM (Linear Kernel)
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(kernel="linear", random_state=42)) 
])

# ==========================================
# 6. TRAIN MODELS
# ==========================================
print("\nTraining Logistic Regression...")
log_pipeline.fit(X_train_smote, y_train_smote)

print("Training Random Forest...")
rf_pipeline.fit(X_train_smote, y_train_smote)

print("Training SVM...")
svm_pipeline.fit(X_train_smote, y_train_smote)

# ==========================================
# 7. EVALUATION FUNCTION
# ==========================================
def evaluate_model(name, model, X_test, y_test):
    # Predict on the REAL test set
    y_pred = model.predict(X_test)
    
    print(f"\n================ {name} REPORT ================")
    # Scores
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.2%}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.2%} (Target: >70%)")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.2%}")
    print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n--- Truth Table (Real Data) ---")
    print(f"Caught Floods (TP):     {tp}")
    print(f"Missed Floods (FN):     {fn}  <-- DANGER ZONE")
    print(f"False Alarms (FP):      {fp}")
    print(f"Correct Non-Floods (TN):{tn}")

# Run Evaluation
evaluate_model("Logistic Regression", log_pipeline, X_test, y_test)
evaluate_model("Random Forest", rf_pipeline, X_test, y_test)
evaluate_model("SVM", svm_pipeline, X_test, y_test)

# ==========================================
# 8. VISUALIZATION (Feature Importance)
# ==========================================
print("\nGenerating Graphs...")
feature_names = X.columns

# --- A. Random Forest Importance ---
rf_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_pipeline.named_steps['model'].feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=rf_importances, x='Importance', y='Feature', palette='viridis')
plt.title("Random Forest: What matters most?")
plt.tight_layout()
plt.show()

# --- B. Logistic Regression Coefficients ---
log_coefs = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': log_pipeline.named_steps['model'].coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=log_coefs, x='Coefficient', y='Feature', palette='coolwarm')
plt.title("Logistic Regression: Flood Drivers (Pos) vs Preventers (Neg)")
plt.axvline(x=0, color='black', linestyle='--')
plt.tight_layout()
plt.show()

# --- C. SVM Coefficients ---
svm_coefs = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': svm_pipeline.named_steps['model'].coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=svm_coefs, x='Coefficient', y='Feature', palette='magma')
plt.title("SVM: Flood Drivers (Pos) vs Preventers (Neg)")
plt.axvline(x=0, color='black', linestyle='--')
plt.tight_layout()
plt.show()

print("\nProcess Complete. No laws of data science were broken.")

print("\n--- FORCING OUTPUT FOR MISSING MODELS ---\n")

# 1. Random Forest
y_pred_rf = rf_pipeline.predict(X_test)
print(">>> RANDOM FOREST RESULTS")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_rf):.2%}")
print(f"Recall:    {recall_score(y_test, y_pred_rf, zero_division=0):.2%}")
print(f"Precision: {precision_score(y_test, y_pred_rf, zero_division=0):.2%}")
print(f"F1 Score:  {f1_score(y_test, y_pred_rf, zero_division=0):.4f}") # <--- Added this
print("Truth Table:", confusion_matrix(y_test, y_pred_rf).ravel())

# 2. SVM
y_pred_svm = svm_pipeline.predict(X_test)
print("\n>>> SVM RESULTS")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_svm):.2%}")
print(f"Recall:    {recall_score(y_test, y_pred_svm, zero_division=0):.2%}")
print(f"Precision: {precision_score(y_test, y_pred_svm, zero_division=0):.2%}")
print(f"F1 Score:  {f1_score(y_test, y_pred_svm, zero_division=0):.4f}") # <--- Added this
print("Truth Table:", confusion_matrix(y_test, y_pred_svm).ravel())