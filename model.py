import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
from scipy.stats import skew, kurtosis, entropy
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore")

# ─── VSCode-Friendly Path Handling ───────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
FILE_PATH = SCRIPT_DIR / "train8.xlsx"

if not FILE_PATH.exists():
    print(f"❌ ERROR: File not found!\nExpected: {FILE_PATH}")
    sys.exit(1)

print(f"✅ Using file: {FILE_PATH}")

# ─── Butterworth bandpass filter (unchanged) ─────────────────────
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# ─── Feature Extraction (exactly the same as before) ─────────────
def extract_enhanced_features(signal, fs=50):
    features = {}
    if np.all(np.isnan(signal)):
        features.update({'mean': 0.0, 'std': 0.0, 'skew': 0.0, 'kurt': 0.0,
                        'zcr': 0.0, 'entropy': 0.0, 'dom_freq': 0.0, 'hr_power': 0.0,
                        'pulse_rate': 0.0, 'sdnn': 0.0, 'rmssd': 0.0,
                        'b_a': np.nan, 'SDAI': np.nan, 'quality_flag': 'all_nan'})
        return features

    sig_detrend = signal - np.nanmean(signal)
    try:
        b, a = butter_bandpass(0.5, 5.0, fs, order=4)
        sig_filt = filtfilt(b, a, sig_detrend)
    except:
        sig_filt = sig_detrend

    sig_std = np.std(sig_filt)
    if np.isnan(sig_std) or sig_std < 1e-6:
        features.update({'mean': 0.0, 'std': 0.0, 'skew': 0.0, 'kurt': 0.0,
                        'zcr': 0.0, 'entropy': 0.0, 'dom_freq': 0.0, 'hr_power': 0.0,
                        'pulse_rate': 0.0, 'sdnn': 0.0, 'rmssd': 0.0,
                        'b_a': np.nan, 'SDAI': np.nan,
                        'quality_flag': 'flat' if sig_std < 1e-6 else 'all_nan'})
        return features

    sig_norm = sig_filt / sig_std

    features['mean'] = np.mean(sig_norm)
    features['std'] = np.std(sig_norm)
    features['skew'] = skew(sig_norm)
    features['kurt'] = kurtosis(sig_norm)

    zero_crossings = np.where(np.diff(np.sign(sig_norm)))[0]
    features['zcr'] = len(zero_crossings) / len(sig_norm)

    hist, _ = np.histogram(sig_norm, bins=20)
    features['entropy'] = entropy(hist + 1e-10)

    n = len(sig_filt)
    fft_vals = np.abs(fft(sig_filt))
    freqs = fftfreq(n, 1/fs)
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    fft_pos = fft_vals[pos_mask]
    hr_mask = (freqs_pos >= 0.5) & (freqs_pos <= 4.0)
    if np.any(hr_mask):
        dom_idx = np.argmax(fft_pos[hr_mask])
        features['dom_freq'] = freqs_pos[hr_mask][dom_idx]
        features['hr_power'] = np.sum(fft_pos[hr_mask])
    else:
        features['dom_freq'] = 0.0
        features['hr_power'] = 0.0

    peaks, _ = find_peaks(sig_filt, distance=fs*0.4, prominence=0.1)
    if len(peaks) >= 2:
        peak_intervals = np.diff(peaks) / fs
        hr_inst = 60.0 / peak_intervals
        features['pulse_rate'] = np.mean(hr_inst)
        features['sdnn'] = np.std(peak_intervals) * 1000
        features['rmssd'] = np.sqrt(np.mean(np.diff(peak_intervals)**2)) * 1000
    else:
        features['pulse_rate'] = 0.0
        features['sdnn'] = 0.0
        features['rmssd'] = 0.0

    vpg = np.gradient(sig_norm)
    apg = np.gradient(vpg)
    apg_smooth = savgol_filter(apg, window_length=15, polyorder=3)
    apg_norm = apg_smooth / (np.max(np.abs(apg_smooth)) + 1e-10)

    pos_peaks, _ = find_peaks(apg_norm, distance=fs*0.18, prominence=0.04, width=2)
    neg_peaks, _ = find_peaks(-apg_norm, distance=fs*0.18, prominence=0.04, width=2)

    features['b_a'] = np.nan
    features['SDAI'] = np.nan
    if len(pos_peaks) >= 2 and len(neg_peaks) >= 2:
        try:
            a_idx = pos_peaks[0]
            b_idx = next((p for p in neg_peaks if p > a_idx), None)
            if b_idx is not None:
                c_idx = next((p for p in pos_peaks if p > b_idx), None)
                if c_idx is not None:
                    d_idx = next((p for p in neg_peaks if p > c_idx), None)
                    if d_idx is not None:
                        e_cand = [p for p in pos_peaks if p > d_idx]
                        e_val = apg_norm[e_cand[0]] if e_cand else 0.0
                        a_val, b_val, c_val, d_val = apg_norm[[a_idx, b_idx, c_idx, d_idx]]
                        denom = a_val + 1e-10
                        features['b_a'] = b_val / denom
                        features['SDAI'] = (b_val - c_val - d_val - e_val) / denom
        except:
            pass

    features['quality_flag'] = 'ok' if len(pos_peaks) >= 2 and len(neg_peaks) >= 2 else 'poor_fiducials'
    return features

# ─── 1-4. Load + Features + Clean + Split (unchanged) ────────────
print("Loading data...")
df_ppg = pd.read_excel(FILE_PATH, sheet_name="Sheet1", header=None, skiprows=1)
df_ppg = df_ppg.iloc[:, 1:]
X_raw = df_ppg.to_numpy(dtype=np.float32)
print(f"PPG signals shape: {X_raw.shape}")

df_info = pd.read_excel(FILE_PATH, sheet_name="Sheet2")
n_segments = len(X_raw)
n_persons = len(df_info)
segments_per_person = n_segments // n_persons
ages = []
for i in range(n_persons):
    count = segments_per_person + (1 if i < (n_segments % n_persons) else 0)
    ages.extend([float(df_info["Age"].iloc[i])] * count)
y = np.array(ages[:n_segments])

print("\nAge distribution:")
print(pd.Series(y).describe())

print("\nExtracting features...")
feature_list = [extract_enhanced_features(sig) for i, sig in enumerate(X_raw)]
df_feat = pd.DataFrame(feature_list)
df_feat['age'] = y

feature_cols = [c for c in df_feat.columns if c not in ['age', 'quality_flag']]
nan_counts = df_feat[feature_cols].isna().sum(axis=1)
valid_rows = nan_counts <= 0.3 * len(feature_cols)
df_clean = df_feat[valid_rows].copy()

for col in feature_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median(skipna=True))

df_clean['b_a_available'] = (~df_feat.loc[valid_rows, 'b_a'].isna()).astype(int)
df_clean['SDAI_available'] = (~df_feat.loc[valid_rows, 'SDAI'].isna()).astype(int)
feature_cols_extended = feature_cols + ['b_a_available', 'SDAI_available']

X_full = df_clean[feature_cols_extended].values
y_age = df_clean['age'].values

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full, y_age, test_size=0.25, random_state=42, stratify=pd.cut(y_age, bins=5)
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_full)
X_test_sc  = scaler.transform(X_test_full)

# ─── 5. RF Baseline + Top-12 Feature Selection ───────────────────
print("\n" + "="*70)
print("  RANDOM FOREST BASELINE + TOP-12 FEATURE SELECTION")
print("="*70)
rf = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train_sc, y_train)
y_pred_rf = rf.predict(X_test_sc)

mae_rf  = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf   = r2_score(y_test, y_pred_rf)

print(f"RF MAE : {mae_rf:.2f} | RMSE : {rmse_rf:.2f} | R² : {r2_rf:.3f}")

# Select top 12 features
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 12
selected_idx = indices[:top_n]
X_train_sel = X_train_sc[:, selected_idx]
X_test_sel  = X_test_sc[:, selected_idx]

# ─── 6. UNSUPERVISED-STYLE DATA AUGMENTATION (Gaussian Noise) ────
print("\nApplying UNSUPERVISED data augmentation (noise ×3)...")
noise_factor = 0.04          # small noise so it doesn't destroy signal
aug_times = 3
X_aug_list = [X_train_sel]
y_aug_list = [y_train]

for _ in range(aug_times):
    noise = np.random.normal(0, noise_factor * X_train_sel.std(axis=0), X_train_sel.shape)
    X_aug_list.append(X_train_sel + noise)
    y_aug_list.append(y_train)

X_train_aug = np.vstack(X_aug_list)
y_train_aug = np.concatenate(y_aug_list)

print(f"Augmented training samples: {len(X_train_aug)} (original {len(X_train_sel)})")

# ─── 7. SHALLOW NEURAL NETWORK (ALL FIXES APPLIED) ───────────────
print("\n" + "="*70)
print("  SHALLOW NEURAL NETWORK (MAE loss + augmentation + simple architecture)")
print("="*70)

model = Sequential([
    Dense(48, activation='relu', input_shape=(12,)),   # very shallow like top papers
    BatchNormalization(),
    Dropout(0.25),
    
    Dense(24, activation='relu'),
    BatchNormalization(),
    Dropout(0.25),
    
    Dense(1)
])

optimizer = Adam(learning_rate=0.0003)   # even lower + stable

model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])   # directly optimize MAE!

early_stop = EarlyStopping(monitor='val_mae', patience=80, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=25, min_lr=1e-6, verbose=1)

history = model.fit(
    X_train_aug, y_train_aug,
    epochs=600,
    batch_size=16,
    validation_split=0.15,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

y_pred_nn = model.predict(X_test_sel, verbose=0).flatten()

mae_nn  = mean_absolute_error(y_test, y_pred_nn)
rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
r2_nn   = r2_score(y_test, y_pred_nn)

print(f"\nFINAL NEURAL NETWORK RESULTS")
print(f"MAE  : {mae_nn:.2f} years")
print(f"RMSE : {rmse_nn:.2f} years")
print(f"R²   : {r2_nn:.3f}")
print(f"Best validation MAE: {min(history.history['val_mae']):.2f}")

# ─── 8-11. Plots (same as before) ────────────────────────────────
# (Model comparison, scatter, history, example signals - identical to previous script)
# ... [plots code omitted here for brevity but included in your file - copy from last version]

plt.show()

print("\n" + "="*70)
print("✅ ALL FIXES IMPLEMENTED:")
print("   • Top-12 RF feature selection")
print("   • Unsupervised Gaussian noise augmentation (×3 data)")
print("   • Shallow 2-hidden-layer NN (exactly like best small-data papers)")
print("   • Direct MAE loss + low LR + strong regularization")
print("   • No PCA (it was hurting small data)")
print("="*70)
print(f"Target check: MAE = {mae_nn:.2f} | R² = {r2_nn:.3f}")