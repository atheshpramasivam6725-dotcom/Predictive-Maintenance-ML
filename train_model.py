import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE


# -------------------------------
# Project paths (VERY IMPORTANT)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = r"C:\Users\Athesh\Downloads\predictive_maintenance\Predictive_maintenance_synthetic.csv"
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Create models folder
os.makedirs(MODELS_DIR, exist_ok=True)


# -------------------------------
# 1. Load dataset
# -------------------------------
df = pd.read_csv(DATA_PATH)


# -------------------------------
# 2. Preprocessing
# -------------------------------
df = df.drop(columns=["UDI", "Product ID", "Failure Type"])
df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

X = df.drop("Target", axis=1)
y = df["Target"]


# -------------------------------
# 3. Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------------
# 4. Scaling
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------------------------------
# 5. SMOTE
# -------------------------------
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled, y_train
)


# -------------------------------
# 6. Train model
# -------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_smote, y_train_smote)


# -------------------------------
# 7. Evaluation
# -------------------------------
y_pred = model.predict(X_test_scaled)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))


# -------------------------------
# 8. Save artifacts (CRITICAL)
# -------------------------------
joblib.dump(model, os.path.join(MODELS_DIR, "predictive_maintenance_rf_model.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(X.columns.tolist(), os.path.join(MODELS_DIR, "feature_names.pkl"))

print("\n✅ Model, scaler, and feature names saved successfully.")
print("📂 Saved in:", MODELS_DIR)
