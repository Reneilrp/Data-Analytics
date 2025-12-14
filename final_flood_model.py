import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

# ==========================================
# 1. PREPARE THE DATA (One Time Setup)
# ==========================================
df = pd.read_csv("Flood_Datasets.csv")

# Preprocessing
if 'Location' in df.columns:
    df = pd.get_dummies(df, columns=["Location"], drop_first=True)

X = df.drop(columns=["FloodOccurrence", "Date"], errors='ignore')
y = df["FloodOccurrence"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. CREATE THE "BLIND" FILES
# ==========================================
# File 1: The Questions (What the models see)
X_test.to_csv("New_Data_To_Predict.csv", index=False)

# File 2: The Answer Key (Hidden)
y_test.to_csv("Hidden_Answer_Key.csv", index=False)

print("Test Files Created: 'New_Data_To_Predict.csv' & 'Hidden_Answer_Key.csv'")

# ==========================================
# 3. TRAIN ALL MODELS (On Training Data Only)
# ==========================================
print("\n--- Training Phase (With SMOTE) ---")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Define all 3 models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42, max_iter=1000))
    ]),
    "Random Forest": Pipeline([
        ("model", RandomForestClassifier(n_estimators=200, random_state=42))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="linear", random_state=42))
    ])
}

# Train them all
for name, pipeline in models.items():
    print(f"Training {name}...")
    pipeline.fit(X_train_smote, y_train_smote)

# ==========================================
# 4. THE REAL-WORLD EXAM
# ==========================================
print("\n--- DEPLOYMENT & TESTING ---")

# Load the "Blind" data
new_data = pd.read_csv("New_Data_To_Predict.csv")
# Load the "Hidden" answers
answer_key = pd.read_csv("Hidden_Answer_Key.csv")["FloodOccurrence"]

# Test each model
for name, pipeline in models.items():
    print(f"\n>>> TESTING: {name}")
    
    # 1. Predict
    predictions = pipeline.predict(new_data)
    
    # 2. Grade
    acc = accuracy_score(answer_key, predictions)
    recall = recall_score(answer_key, predictions, zero_division=0)
    cm = confusion_matrix(answer_key, predictions)
    
    # 3. Report
    print(f"Accuracy: {acc:.2%}")
    print(f"Recall:   {recall:.2%} (Floods Caught)")
    print("Truth Table:")
    print(cm)
    
    # Optional: Save individual results
    # pd.DataFrame(predictions).to_csv(f"Predictions_{name}.csv", index=False)

print("\nAll models have been graded against the hidden key.")