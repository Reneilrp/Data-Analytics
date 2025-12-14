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

# ==========================================
# CONFIGURATION
# ==========================================
# Make sure this matches the name of your NEW balanced file
DATASET_FILENAME = "Flood_Datasets_Balanced_SMOTE.csv" 

# ==========================================
# 1. LOAD DATA
# ==========================================
try:
    df = pd.read_csv(DATASET_FILENAME)
    print(f"Successfully loaded '{DATASET_FILENAME}'")
    print(f"Dataset Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Could not find '{DATASET_FILENAME}'.")
    print("Make sure the file is in the same folder as this script.")
    exit()

# ==========================================
# 2. PREPROCESSING
# ==========================================
# Handle categorical data (if it still exists in the SMOTE file)
if 'Location' in df.columns:
    df = pd.get_dummies(df, columns=["Location"], drop_first=True)

# Split Features and Target
X = df.drop(columns=["FloodOccurrence", "Date"], errors='ignore')
y = df["FloodOccurrence"]

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. DEFINE MODEL PIPELINES
# ==========================================
# We use Pipelines to ensure Scaling happens automatically and correctly.

# A. Logistic Regression
log_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(random_state=42, max_iter=1000))
])

# B. Random Forest (Scaling not strictly needed, but doesn't hurt)
rf_pipeline = Pipeline([
    ("model", RandomForestClassifier(n_estimators=200, random_state=42))
])

# C. SVM (Linear Kernel is CRITICAL for seeing feature importance)
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(kernel="linear", random_state=42)) 
])

# ==========================================
# 4. TRAIN MODELS
# ==========================================
print("\nTraining Logistic Regression...")
log_pipeline.fit(X_train, y_train)

print("Training Random Forest...")
rf_pipeline.fit(X_train, y_train)

print("Training SVM...")
svm_pipeline.fit(X_train, y_train)

# ==========================================
# 5. SUPERIOR EVALUATION FUNCTION
# ==========================================
def evaluate_model(name, model, X_test, y_test):
    """
    Prints a report combining Scores (for the boss) and 
    Confusion Matrix (for the engineer).
    """
    # Generate Predictions
    y_pred = model.predict(X_test)
    
    print(f"\n================ {name} REPORT ================")
    
    # Part 1: The Scores
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.2%}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.2%} (Prioritize this for safety!)")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.2%}")
    print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    
    # Part 2: The Truth Table (Confusion Matrix)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n--- Truth Table ---")
    print(f"Caught Floods (TP):     {tp}")
    print(f"Missed Floods (FN):     {fn}  <-- DANGER ZONE")
    print(f"False Alarms (FP):      {fp}")
    print(f"Correct Non-Floods (TN):{tn}")

# Run Evaluation
evaluate_model("Logistic Regression", log_pipeline, X_test, y_test)
evaluate_model("Random Forest", rf_pipeline, X_test, y_test)
evaluate_model("SVM", svm_pipeline, X_test, y_test)

# ==========================================
# 6. VISUALIZATION (Feature Importance)
# ==========================================
print("\nGenerating Feature Importance Graphs...")
feature_names = X.columns

# --- Plot 1: Random Forest ---
# Access the model step inside the pipeline using .named_steps['model']
rf_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_pipeline.named_steps['model'].feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=rf_importances, x='Importance', y='Feature', palette='viridis')
plt.title("Random Forest: Feature Importance (Magnitude Only)")
plt.tight_layout()
plt.show()

# --- Plot 2: Logistic Regression Coefficients ---
log_coefs = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': log_pipeline.named_steps['model'].coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=log_coefs, x='Coefficient', y='Feature', palette='coolwarm')
plt.title("Logistic Regression Coefficients (Pos=Flood, Neg=Safe)")
plt.axvline(x=0, color='black', linestyle='--')
plt.tight_layout()
plt.show()

# --- Plot 3: SVM Coefficients ---
svm_coefs = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': svm_pipeline.named_steps['model'].coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=svm_coefs, x='Coefficient', y='Feature', palette='magma')
plt.title("SVM Coefficients (Pos=Flood, Neg=Safe)")
plt.axvline(x=0, color='black', linestyle='--')
plt.tight_layout()
plt.show()

print("\nAnalysis Complete.")