#!/usr/bin/env python3
"""
Train autoencoder + Random Forest for age prediction from PPG signals.
Saves all necessary artifacts for prediction.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import pickle

# Import feature extraction
from features import extract_enhanced_features

# -------------------- Paths --------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_FILE = SCRIPT_DIR / "train8.xlsx"                     # PPG training file
META_FILE = SCRIPT_DIR / "archive (1)" / "subjects_metadata.xlsx"   # metadata

if not DATA_FILE.exists():
    print(f"❌ ERROR: {DATA_FILE} not found!")
    sys.exit(1)
if not META_FILE.exists():
    print(f"❌ ERROR: {META_FILE} not found!")
    sys.exit(1)

print(f"✅ Using PPG data: {DATA_FILE}")
print(f"✅ Using metadata: {META_FILE}")

# Sampling rate of training data (adjust if different)
FS_TRAIN = 50

# -------------------- Load PPG signals --------------------
df_ppg = pd.read_excel(DATA_FILE, sheet_name="Sheet1", header=None, skiprows=1)
df_ppg = df_ppg.iloc[:, 1:]                     # drop first column (index)
X_raw = df_ppg.to_numpy(dtype=np.float32)
print(f"PPG signals shape: {X_raw.shape}")

# -------------------- Load metadata (subject IDs and ages) --------------------
meta = pd.read_excel(META_FILE)
# Expect columns: 'subjectcode', 'age'
subject_ids = meta['subjectcode'].values
ages = meta['age'].values

# Each subject has multiple PPG segments. Assign a subject ID to each row.
n_subjects = len(meta)
segments_per_subject = len(X_raw) // n_subjects
subject_labels = []
for i, sid in enumerate(subject_ids):
    subject_labels.extend([sid] * segments_per_subject)
# Handle any remainder
remainder = len(X_raw) - len(subject_labels)
if remainder > 0:
    subject_labels.extend([subject_ids[-1]] * remainder)
subject_labels = np.array(subject_labels)

# Create age array aligned with each segment
age_per_segment = []
for sid in subject_labels:
    age = meta.loc[meta['subjectcode'] == sid, 'age'].values[0]
    age_per_segment.append(age)
y = np.array(age_per_segment, dtype=np.float32)

print(f"Subjects: {n_subjects}, segments per subject: {segments_per_subject}")

# -------------------- Feature extraction --------------------
print("\nExtracting features...")
feature_list = [extract_enhanced_features(sig, fs=FS_TRAIN) for sig in X_raw]
df_feat = pd.DataFrame(feature_list)

feature_cols = [c for c in df_feat.columns if c != 'quality_flag']

# Keep rows with ≤30% NaNs
nan_counts = df_feat[feature_cols].isna().sum(axis=1)
valid_mask = nan_counts <= 0.3 * len(feature_cols)
df_clean = df_feat[valid_mask].copy()
y_clean = y[valid_mask]
subject_labels_clean = subject_labels[valid_mask]

# Fill remaining NaNs with median of that feature (and save medians)
feature_medians = {}
for col in feature_cols:
    median_val = df_clean[col].median(skipna=True)
    feature_medians[col] = median_val
    df_clean[col] = df_clean[col].fillna(median_val)

X = df_clean[feature_cols].values
print(f"Final feature matrix shape: {X.shape}")

# -------------------- Train/Test split by subject --------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(gss.split(X, y_clean, groups=subject_labels_clean))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_clean[train_idx], y_clean[test_idx]

print(f"Train subjects: {len(np.unique(subject_labels_clean[train_idx]))}")
print(f"Test subjects:  {len(np.unique(subject_labels_clean[test_idx]))}")

# -------------------- Scale features --------------------
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "feature_scaler.pkl")
print("✅ Scaler saved as 'feature_scaler.pkl'")

# -------------------- Build a simple autoencoder --------------------
input_dim = X_train_sc.shape[1]
bottleneck_dim = max(8, input_dim // 4)   # e.g., 8-16

input_layer = Input(shape=(input_dim,))
# Encoder
x = Dense(128, activation='relu')(input_layer)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
encoded = Dense(bottleneck_dim, activation='linear', name='bottleneck')(x)

# Decoder
x = Dense(32, activation='relu')(encoded)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
decoded = Dense(input_dim, activation='linear')(x)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

print("\nTraining simple autoencoder...")
early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
history = autoencoder.fit(
    X_train_sc, X_train_sc,
    epochs=300,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# -------------------- Extract encoder features --------------------
encoder = Model(input_layer, encoded)
encoded_train = encoder.predict(X_train_sc)
encoded_test = encoder.predict(X_test_sc)

# -------------------- Train Random Forest on encoded features --------------------
print("\nTraining Random Forest on encoder features...")
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(encoded_train, y_train)

y_pred = rf.predict(encoded_test)
mae = np.mean(np.abs(y_pred - y_test))
print(f"Random Forest MAE on test set: {mae:.2f} years")

# Save models
encoder.save("encoder_model.keras")
with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)
print("✅ Encoder saved as 'encoder_model.keras'")
print("✅ Random Forest saved as 'rf_model.pkl'")

# Save feature medians and feature columns for later imputation
with open("feature_medians.pkl", "wb") as f:
    pickle.dump(feature_medians, f)
with open("feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

# Save training parameters (fs and expected signal length)
training_params = {
    'fs': FS_TRAIN,
    'expected_length': X_raw.shape[1]  # number of samples per signal
}
with open("training_params.pkl", "wb") as f:
    pickle.dump(training_params, f)

print("✅ Feature medians, columns, and training parameters saved.")