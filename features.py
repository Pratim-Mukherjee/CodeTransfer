# features.py
import numpy as np
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
from scipy.stats import skew, kurtosis, entropy
from scipy.fft import fft, fftfreq

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

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