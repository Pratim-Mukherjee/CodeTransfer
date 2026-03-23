import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Import feature extraction from model.py
from model import extract_enhanced_features

# ─── Command-line argument for dataset ───────────────
if len(sys.argv) < 2:
    print("Usage: python predict_age_from_pickle.py <dataset.xlsx>")
    sys.exit(1)

DATASET_PATH = Path(sys.argv[1]).resolve()
if not DATASET_PATH.exists():
    print(f"❌ ERROR: File not found!\nExpected: {DATASET_PATH}")
    sys.exit(1)

print(f"✅ Using file: {DATASET_PATH}")

# ─── Load Data ───────────────────────────────────────
#df_ppg = pd.read_excel(DATASET_PATH, sheet_name="Sheet1", header=None, skiprows=1)
if DATASET_PATH.endswith('.csv'):
    df_ppg = pd.read_csv(DATASET_PATH, header=None, skiprows=1)
else:
    df_ppg = pd.read_excel(DATASET_PATH, sheet_name="Sheet1", header=None, skiprows=1)
df_ppg = df_ppg.iloc[:, 1:]
X_raw = df_ppg.to_numpy(dtype=np.float32)
print(f"PPG signals shape: {X_raw.shape}")

# ─── Extract Features ───────────────────────────────
print("\nExtracting features...")
feature_list = [extract_enhanced_features(sig) for sig in X_raw]
df_feat = pd.DataFrame(feature_list)
feature_cols = [c for c in df_feat.columns if c not in ['quality_flag']]
nan_counts = df_feat[feature_cols].isna().sum(axis=1)
valid_rows = nan_counts <= 0.3 * len(feature_cols)
df_clean = df_feat[valid_rows].copy()
for col in feature_cols:
    median_val = df_clean[col].median(skipna=True)
    df_clean[col] = df_clean[col].fillna(median_val)
X = df_clean[feature_cols].values

# ─── Load Encoder & Transform Features ───────────────
encoder = load_model("deep_encoder_model.keras")
import joblib   # at top
scaler = joblib.load("feature_scaler.pkl")   # or pickle.load
X_sc = scaler.transform(X)   # ✅ transform only, no fit!
encoded_features = encoder.predict(X_sc)

# ─── Load Supervised Model & Predict Age ─────────────
with open("rf_encoder_model.pkl", "rb") as f:
    rf = pickle.load(f)

age_pred = rf.predict(encoded_features)

print("\nPredicted ages:")
print(age_pred)

# ─── Show Output ────────────────────────────────────
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.hist(age_pred, bins=20, color='#1f77b4', alpha=0.7)
plt.xlabel('Predicted Age (years)')
plt.ylabel('Count')
plt.title('Predicted Age Distribution')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n✅ Age prediction complete! Output shown.")
