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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE


# ==========================================
# 1. LOAD DATA
# ==========================================
df = pd.read_csv("Flood_Datasets.csv")

# ==========================================
# 2. PREPROCESSING
# ==========================================
# One-hot encode Location
df = pd.get_dummies(df, columns=["Location"], drop_first=True)

# Split Features and Target
X = df.drop(columns=["FloodOccurrence", "Date"])
y = df["FloodOccurrence"]

# SPLIT FIRST (Crucial Step!)
# We split before SMOTE so we don't leak fake data into the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Original Training Flood Count: {sum(y_train)}")
print(f"Original Training Size: {len(X_train)}")

# ==========================================
# 3. APPLY SMOTE (Synthetic Data Generation)
# ==========================================
print("\n--- Applying SMOTE... ---")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"New Synthetic Flood Count: {sum(y_train_smote)}")
print(f"New Training Size: {len(X_train_smote)}")
print("(Notice how the Flood count is now equal to the Non-Flood count!)")

# ==========================================
# 4. DEFINE MODELS (No Class Weight Needed!)
# ==========================================
# Since data is now 50/50, we remove "class_weight='balanced'"

# Logistic Regression
log_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(random_state=42, max_iter=1000))
])

# Random Forest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)

# SVM (Linear for visibility)
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear", random_state=42)) # No class_weight needed now
])

# ==========================================
# 5. TRAIN MODELS
# ==========================================
print("\nTraining models on SMOTE data...")

log_pipeline.fit(X_train_smote, y_train_smote)
rf_model.fit(X_train_smote, y_train_smote)
svm_pipeline.fit(X_train_smote, y_train_smote)

# ==========================================
# 6. EVALUATION
# ==========================================
def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    print(f"\n================ {name} (SMOTE) ================")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.2%}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.2%} (Target: High)")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.2%}")
    print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"Caught Floods (TP): {tp} | Missed Floods (FN): {fn}")
    print(f"False Alarms (FP):  {fp}")

evaluate("Logistic Regression", log_pipeline, X_test, y_test)
evaluate("Random Forest", rf_model, X_test, y_test)
evaluate("SVM", svm_pipeline, X_test, y_test)

# OPTIONAL: Save the synthetic training data to a new file
# Combine X and y back together just for saving
synthetic_df = pd.concat([X_train_smote, y_train_smote], axis=1)

# Save to a NEW file (so we don't touch the original)
synthetic_df.to_csv("Flood_Datasets_Balanced_SMOTE.csv", index=False)

print("Success! I saved the new balanced dataset to 'Flood_Datasets_Balanced_SMOTE.csv'")

# OPTIONAL: Save the synthetic training data to a new file
# Combine X and y back together just for saving
synthetic_df = pd.concat([X_train_smote, y_train_smote], axis=1)

# Save to a NEW file (so we don't touch the original)
synthetic_df.to_csv("Flood_Datasets_Balanced_SMOTE.csv", index=False)

print("Success! I saved the new balanced dataset to 'Flood_Datasets_Balanced_SMOTE.csv'")