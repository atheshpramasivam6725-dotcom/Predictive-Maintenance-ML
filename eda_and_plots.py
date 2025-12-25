import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -------------------------------
# Base project directory
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = r"C:\Users\Athesh\Downloads\predictive_maintenance\Predictive_maintenance_synthetic.csv"
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Create plots folder
os.makedirs(PLOTS_DIR, exist_ok=True)

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(DATA_PATH)

# -------------------------------
# 1. Target Distribution
# -------------------------------
plt.figure(figsize=(6, 4))
df["Target"].value_counts().plot(kind="bar")
plt.title("Target Distribution (Failure vs No Failure)")
plt.xlabel("Target")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "target_distribution.png"))
plt.close()

# -------------------------------
# 2. Correlation Heatmap (numeric only)
# -------------------------------
numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"))
plt.close()

# -------------------------------
# 3. Feature Importance
# -------------------------------
model = joblib.load(
    os.path.join(MODELS_DIR, "predictive_maintenance_rf_model.pkl")
)
features = joblib.load(
    os.path.join(MODELS_DIR, "feature_names.pkl")
)

importances = model.feature_importances_
fi = pd.Series(importances, index=features).sort_values()

plt.figure(figsize=(8, 6))
fi.plot(kind="barh")
plt.title("Feature Importance - Predictive Maintenance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"))
plt.close()

print("✅ All plots saved successfully in the 'plots/' folder.")
