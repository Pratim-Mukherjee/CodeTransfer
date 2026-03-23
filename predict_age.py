#!/usr/bin/env python3
"""
Predict age from PPG signal(s) using pre‑trained encoder + Random Forest.
Handles:
  - Variable sampling rate (--fs)
  - Signals of any length (with warning if too short/long)
  - Robust CSV loading (header/no header, single/multiple signals)
  - Consistent imputation using training medians
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import pickle
from tensorflow.keras.models import load_model

# Import feature extraction
from features import extract_enhanced_features

def load_signals_from_file(file_path):
    """
    Intelligently load PPG signals from a CSV file.
    Returns (signals_array, patient_ids) where signals_array shape = (n_signals, n_samples).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # 1. Check if it's a merged file with a 'patient_id' column
    try:
        df_sample = pd.read_csv(file_path, nrows=0)
        if 'patient_id' in df_sample.columns:
            df = pd.read_csv(file_path)
            patient_ids = df['patient_id'].values
            numeric_df = df.drop(columns=['patient_id']).select_dtypes(include=[np.number])
            if numeric_df.shape[1] == 0:
                raise ValueError("No numeric columns after dropping 'patient_id'")
            signals = numeric_df.to_numpy(dtype=np.float32)
            print(f"  → Merged file: {signals.shape[0]} signals, {signals.shape[1]} samples each")
            return signals, patient_ids
    except Exception:
        pass

    # 2. Try reading with header=0 (first row as header)
    try:
        df = pd.read_csv(file_path, header=0)
        df_numeric = df.select_dtypes(include=[np.number])
        if not df_numeric.empty:
            data = df_numeric.to_numpy(dtype=np.float32)
            # If there is only one column, shape will be (n_rows, 1)
            if data.shape[1] == 1:
                data = data.flatten().reshape(1, -1)
                patient_ids = [file_path.stem]
            else:
                # Multiple columns: each row is a separate signal
                patient_ids = [file_path.stem] * data.shape[0]
            print(f"  → Loaded with header: {data.shape[0]} signals, {data.shape[1]} samples each")
            return data, patient_ids
    except Exception:
        pass

    # 3. Fallback: read with header=None, then try to drop the first row if it's non-numeric
    df = pd.read_csv(file_path, header=None)
    # Check if first row contains any non-numeric (string) values
    try:
        # Attempt to convert entire DataFrame to float; if fails, first row is likely header
        df_float = df.astype(float)
        # If all rows convert, we have pure numbers
        data = df_float.to_numpy(dtype=np.float32)
    except ValueError:
        # First row is probably a header – drop it and use the rest
        df = pd.read_csv(file_path, header=None, skiprows=1)
        df_float = df.astype(float)
        data = df_float.to_numpy(dtype=np.float32)

    # Now data is purely numeric. Determine orientation.
    if data.ndim == 1 or data.shape[0] == 1:
        # Single signal
        signals = data.reshape(1, -1)
        patient_ids = [file_path.stem]
    elif data.shape[1] == 1:
        # One column, many rows → single signal (time points as rows)
        signals = data.flatten().reshape(1, -1)
        patient_ids = [file_path.stem]
    else:
        # Multiple rows and columns → each row is a separate signal
        signals = data
        patient_ids = [file_path.stem] * data.shape[0]

    print(f"  → Loaded without header: {signals.shape[0]} signals, {signals.shape[1]} samples each")
    return signals, patient_ids

def extract_features_from_signals(signals, fs, feature_cols, feature_medians):
    """Extract features for all signals, fill NaNs with training medians."""
    feature_list = [extract_enhanced_features(sig, fs=fs) for sig in signals]
    df_feat = pd.DataFrame(feature_list)
    # Ensure all required columns exist (fill missing with NaN)
    for col in feature_cols:
        if col not in df_feat.columns:
            df_feat[col] = np.nan
    # Impute with saved medians
    for col in feature_cols:
        df_feat[col] = df_feat[col].fillna(feature_medians[col])
    # Return in correct order
    return df_feat[feature_cols].values

def main():
    parser = argparse.ArgumentParser(description="Predict age from PPG signals")
    parser.add_argument("input", help="Path to CSV file or directory containing CSV files")
    parser.add_argument("--fs", type=float, default=50.0,
                        help="Sampling rate of the input signals (default: 50 Hz)")
    parser.add_argument("--metadata", help="Optional Excel file with true ages for evaluation")
    args = parser.parse_args()

    input_path = Path(args.input)
    fs = args.fs

    # Load all signals
    if input_path.is_dir():
        all_signals = []
        all_ids = []
        for csv_file in sorted(input_path.glob("*.csv")):
            print(f"\n📄 Processing {csv_file.name}")
            try:
                signals, ids = load_signals_from_file(csv_file)
                all_signals.append(signals)
                all_ids.extend(ids)
            except Exception as e:
                print(f"⚠️ Skipping {csv_file.name}: {e}")
        if not all_signals:
            print("No valid CSV files found in directory.")
            sys.exit(1)
        signals = np.vstack(all_signals)
        patient_ids = all_ids
    else:
        print(f"\n📄 Processing {input_path.name}")
        signals, patient_ids = load_signals_from_file(input_path)

    print(f"\n✅ Total signals loaded: {signals.shape[0]}")
    print(f"   Each signal length: {signals.shape[1]} samples")

    # Load artifacts from training
    print("\n📦 Loading models and metadata...")
    scaler = joblib.load("feature_scaler.pkl")
    encoder = load_model("encoder_model.keras")
    with open("rf_model.pkl", "rb") as f:
        rf = pickle.load(f)
    with open("feature_medians.pkl", "rb") as f:
        feature_medians = pickle.load(f)
    with open("feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    with open("training_params.pkl", "rb") as f:
        training_params = pickle.load(f)

    expected_length = training_params.get('expected_length', None)
    if expected_length is not None and signals.shape[1] != expected_length:
        print(f"⚠️  Warning: Signal length ({signals.shape[1]}) differs from training length ({expected_length}).")
        print("   Features may be affected. Consider resampling or truncating to match.")

    # Extract features using the correct fs and impute with training medians
    print("\n🔍 Extracting features...")
    X = extract_features_from_signals(signals, fs, feature_cols, feature_medians)
    print(f"   Feature matrix shape: {X.shape}")

    # Scale and encode
    X_sc = scaler.transform(X)
    encoded = encoder.predict(X_sc, verbose=0)
    print(f"   Encoded features shape: {encoded.shape}")

    # Predict
    ages_pred = rf.predict(encoded)
    print(f"\n🎯 Predictions (first 10): {ages_pred[:10]}")

    # Output
    results = pd.DataFrame({
        'patient_id': patient_ids,
        'predicted_age': ages_pred
    })
    print("\n📊 Predictions summary:")
    print(results.describe())

    # Evaluate if metadata provided
    if args.metadata:
        meta_path = Path(args.metadata)
        if not meta_path.exists():
            print(f"⚠️ Metadata file not found: {meta_path}")
        else:
            meta = pd.read_excel(meta_path)
            # Assume metadata has columns 'subjectcode' and 'age'
            merged = results.merge(meta, left_on='patient_id', right_on='subjectcode', how='left')
            if 'age' in merged.columns:
                true_ages = merged['age'].dropna()
                pred_ages = merged.loc[true_ages.index, 'predicted_age']
                if len(true_ages) > 0:
                    mae = np.mean(np.abs(pred_ages - true_ages))
                    print(f"\n✅ MAE against true ages: {mae:.2f} years")
                else:
                    print("No matching subject codes found in metadata.")
            else:
                print("Metadata file does not contain 'age' column.")

    # Save predictions
    results.to_csv("predictions.csv", index=False)
    print("\n💾 Predictions saved to predictions.csv")

if __name__ == "__main__":
    main()