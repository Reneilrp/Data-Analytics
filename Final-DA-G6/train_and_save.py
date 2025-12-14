import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib  # This library saves your models to files

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
try:
    df = pd.read_csv("Flood_Datasets.csv")
    print("Dataset loaded.")
except FileNotFoundError:
    print("Error: 'Flood_Datasets.csv' not found.")
    exit()

# Preprocessing
if 'Location' in df.columns:
    df = pd.get_dummies(df, columns=["Location"], drop_first=True)

X = df.drop(columns=["FloodOccurrence", "Date"], errors='ignore')
y = df["FloodOccurrence"]

# ==========================================
# 2. THE GOLDEN RULE SPLIT
# ==========================================
# We split first. The Test data is separated immediately.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SAVE THE TEST DATA FOR FILE 2
# We save the questions (X) and answers (y) separately for the second script
X_test.to_csv("test_features.csv", index=False)
y_test.to_csv("test_labels.csv", index=False)
print("Locked Test Data saved to 'test_features.csv' and 'test_labels.csv'.")

# ==========================================
# 3. SMOTE (Training Data Only)
# ==========================================
print("Applying SMOTE to Training set...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# ==========================================
# 4. TRAIN & SAVE MODELS
# ==========================================
# Define the 3 Pipelines
models = {
    "Logistic_Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42, max_iter=1000))
    ]),
    "Random_Forest": Pipeline([
        ("model", RandomForestClassifier(n_estimators=200, random_state=42))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="linear", random_state=42))
    ])
}

print("\n--- Training and Saving Models ---")
for name, pipeline in models.items():
    # Train on SMOTE data
    pipeline.fit(X_train_smote, y_train_smote)
    
    # Save the trained model file (.pkl)
    filename = f"{name}_model.pkl"
    joblib.dump(pipeline, filename)
    print(f"✅ {name} trained and saved as '{filename}'")

print("\nStep 1 Complete. You can now run the second file.")