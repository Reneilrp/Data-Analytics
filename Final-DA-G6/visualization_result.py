import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. LOAD DATA & MODELS
# ==========================================
try:
    X_test = pd.read_csv("test_features.csv")
    y_test = pd.read_csv("test_labels.csv").values.ravel()
    
    models = {
        "Logistic Regression": joblib.load("Logistic_Regression_model.pkl"),
        "Random Forest": joblib.load("Random_Forest_model.pkl"),
        "SVM": joblib.load("SVM_model.pkl")
    }
    print("✅ Models and Test Data loaded successfully.")
except FileNotFoundError:
    print("❌ Error: Missing files. Run 'train_and_save.py' first.")
    exit()

# ==========================================
# 2. GENERATE CONFUSION MATRICES
# ==========================================
print("Generating Confusion Matrices...")
plt.figure(figsize=(18, 5))

for i, (name, model) in enumerate(models.items()):
    plt.subplot(1, 3, i+1)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 14})
    
    # --- CHANGE IS HERE: ONLY MODEL NAME ---
    plt.title(name, fontsize=16, fontweight='bold') 
    
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.xticks([0.5, 1.5], ["Safe", "Flood"], fontsize=11)
    plt.yticks([0.5, 1.5], ["Safe", "Flood"], fontsize=11)

plt.tight_layout()
plt.savefig("Final_Confusion_Matrices.png", dpi=300)
print("✅ Saved 'Final_Confusion_Matrices.png'")
plt.show()

# ==========================================
# 3. GENERATE FEATURE IMPORTANCE (RF Only)
# ==========================================
print("Generating Feature Importance...")
rf_model = models["Random Forest"]
importances = rf_model.named_steps['model'].feature_importances_
feature_names = X_test.columns

# Create DataFrame for plotting
df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
df_imp = df_imp.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_imp, x='Importance', y='Feature', palette='viridis')
plt.title("Random Forest Feature Importance", fontsize=16, fontweight='bold')
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.savefig("Final_Feature_Importance.png", dpi=300)
print("✅ Saved 'Final_Feature_Importance.png'")
plt.show()