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
                             f1_score, classification_report, confusion_matrix)

# ==========================================
# 1. LOAD DATA
# ==========================================
# Try/Except block to handle file loading gracefully
try:
    df = pd.read_csv("Flood_Datasets.csv")
except FileNotFoundError:
    print("Error: 'Flood_Datasets.csv' not found. Please upload the file.")
    exit()

# ==========================================
# 2. PREPROCESSING
# ==========================================
df = pd.get_dummies(df, columns=["Location"], drop_first=True)

# Split Data
# We use a fixed random_state=42 for reproducibility during testing.
# In production, you might remove this to test robustness.
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

X_train = train_df.drop(columns=["FloodOccurrence", "Date"])
y_train = train_df["FloodOccurrence"]

X_test = test_df.drop(columns=["FloodOccurrence", "Date"])
y_test = test_df["FloodOccurrence"]

# ==========================================
# 3. DEFINE MODELS (With Imbalance Fix)
# ==========================================

# Logistic Regression Pipeline
log_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        class_weight='balanced',  # <--- MAGIC SWITCH (Fixes Imbalance)
        random_state=42, 
        max_iter=1000
    ))
])

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200, 
    class_weight='balanced',      # <--- MAGIC SWITCH (Fixes Imbalance)
    random_state=42
)

# SVM Pipeline
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="linear", 
        class_weight='balanced',  # <--- MAGIC SWITCH (Fixes Imbalance)
        random_state=42
    ))
])

# ==========================================
# 4. TRAIN MODELS
# ==========================================
print("Training Logistic Regression...")
log_pipeline.fit(X_train, y_train)

print("Training Random Forest...")
rf_model.fit(X_train, y_train)

print("Training SVM...")
svm_pipeline.fit(X_train, y_train)

# Generate Predictions
y_pred_log = log_pipeline.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_svm = svm_pipeline.predict(X_test)

# ==========================================
# 5. EVALUATION FUNCTION (With Matrix)
# ==========================================
def evaluate(name, y_pred):
    print(f"\n================ {name} ================")
    
    # 1. Score Metrics
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.2%}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.2%}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.2%} (Target: High)")
    print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    
    # 2. Confusion Matrix (The Truth Table)
    cm = confusion_matrix(y_test, y_pred)
    
    # Visualizing the Matrix textually
    tn, fp, fn, tp = cm.ravel()
    print(f"\n--- Confusion Matrix ---")
    print(f"True Negatives (Correct Non-Flood): {tn}")
    print(f"False Positives (False Alarm):      {fp}")
    print(f"False Negatives (MISSED FLOOD):     {fn}  <-- We want this to be 0!")
    print(f"True Positives (Caught Flood):      {tp}")

# Evaluate all
evaluate("Logistic Regression", y_pred_log)
evaluate("Random Forest", y_pred_rf)
evaluate("SVM", y_pred_svm)

print("\n(Note: If Recall is high but Precision is low, the model is 'playing it safe' by predicting floods more often.)")