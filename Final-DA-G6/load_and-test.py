import pandas as pd
import joblib
# Added f1_score to the imports below
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score

# ==========================================
# 1. LOAD THE TEST DATA
# ==========================================
try:
    X_test = pd.read_csv("test_features.csv")
    y_test = pd.read_csv("test_labels.csv").values.ravel() # Flatten to array
    print("Test data loaded successfully.")
except FileNotFoundError:
    print("Error: Test files not found. Please run 'train_and_save.py' first.")
    exit()

# ==========================================
# 2. LOAD & GRADE MODELS
# ==========================================
model_names = ["Logistic_Regression", "Random_Forest", "SVM"]

print("\n--- GRADING RESULTS ---")

for name in model_names:
    try:
        # Load the model from the hard drive
        model = joblib.load(f"{name}_model.pkl")
        
        # Make Predictions
        y_pred = model.predict(X_test)
        
        # Calculate Scores
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0) # <--- Calculated F1 here
        
        # Print Report
        print(f"\n>>> {name.replace('_', ' ')}")
        print(f"Accuracy:  {acc:.2%}")
        print(f"Recall:    {rec:.2%} (Floods Caught)")
        print(f"Precision: {prec:.2%}")
        print(f"F1 Score:  {f1:.4f}") # <--- Printed F1 here
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"Caught: {tp} | Missed: {fn} | False Alarms: {fp}")

    except FileNotFoundError:
        print(f"Could not find model file: {name}_model.pkl")

print("\nTesting Complete.")