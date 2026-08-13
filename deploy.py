import streamlit as st
import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
from abc import ABC

st.set_page_config(page_title="Remote PPG Signal De-Noiser", layout="wide")

# ==========================================
# 1. NOISE GENERATOR CLASS
# ==========================================
class random_noise(ABC):
    """Utility for creating realistic synthetic PPG noise for benchmarking and demos."""

    def __init__(self, amplitude=0.3, hum_freq=50.0, drift_level=0.2, fs=50.0, rng=None):
        self.amplitude = float(amplitude)
        self.hum_freq = float(hum_freq)
        self.drift_level = float(drift_level)
        self.fs = float(fs)
        self.rng = np.random.default_rng() if rng is None else rng

    def _normalize(self, signal_values):
        signal_values = np.asarray(signal_values, dtype=np.float64)
        vmin = np.min(signal_values)
        vmax = np.max(signal_values)
        denom = vmax - vmin + 1e-8
        return (signal_values - vmin) / denom

    def add_hum(self, time_axis, amplitude=None):
        amp = self.amplitude if amplitude is None else float(amplitude)
        hum = amp * np.sin(2 * np.pi * self.hum_freq * time_axis)
        return hum

    def add_drift(self, time_axis, amplitude=None):
        amp = self.drift_level if amplitude is None else float(amplitude)
        drift = amp * np.sin(2 * np.pi * 0.2 * time_axis)
        return drift

    def generate_noisy_signal(self, clean_signal, time_axis):
        hum = self.add_hum(time_axis)
        drift = self.add_drift(time_axis)
        gaussian = self.rng.normal(0, self.amplitude * 0.5, len(time_axis))
        noisy = clean_signal + hum + drift + gaussian
        return self._normalize(noisy)


# ==========================================
# 2. 1D U-NET MODEL ARCHITECTURE
# ==========================================
class SignalDenoisingUNet1D(nn.Module):
    def __init__(self):
        super(SignalDenoisingUNet1D, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose1d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        d2 = self.dec2(e2)
        cat1 = torch.cat((d2, e1), dim=1)  # Skip Connection
        return self.dec1(cat1)


@st.cache_resource
def load_trained_model(weights_path='unet_realdata_weights.pth'):
    model = SignalDenoisingUNet1D()
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        st.sidebar.success("Loaded pre-trained U-Net weights!")
    else:
        st.sidebar.warning("`unet_realdata_weights.pth` not found. Using initialized model weights.")
    model.eval()
    return model


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
@st.cache_data
def load_uploaded_signal(uploaded_file):
    df = pd.read_csv(uploaded_file)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No numeric data columns found in CSV.")
    
    cols_lower = [str(c).lower() for c in numeric_df.columns]
    
    if 'pleth_1' in cols_lower:
        idx = cols_lower.index('pleth_1')
    elif any('pleth' in c for c in cols_lower):
        idx = [i for i, c in enumerate(cols_lower) if 'pleth' in c][0]
    elif any('ppg' in c for c in cols_lower):
        idx = [i for i, c in enumerate(cols_lower) if 'ppg' in c][0]
    else:
        idx = 1 if numeric_df.shape[1] > 1 else 0

    raw_signal = numeric_df.iloc[:, idx].values
    raw_signal = raw_signal[~np.isnan(raw_signal)]
    return raw_signal.astype(np.float32)

def generate_synthetic_ppg(t, hr_bpm):
    freq = hr_bpm / 60.0
    clean = 0.5 * (1 + np.sin(2 * np.pi * freq * t))
    dicrotic = 0.15 * np.exp(-((np.mod(t * freq, 1) - 0.35) ** 2) / 0.01)
    signal_out = clean + dicrotic
    return (signal_out - np.min(signal_out)) / (np.max(signal_out) - np.min(signal_out) + 1e-8)

def butter_bandpass_filter(data, lowcut=0.5, highcut=4.0, fs=50.0, order=2):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def calculate_sqi(sig, fs=50.0):
    freqs, psd = signal.welch(sig, fs=fs, nperseg=len(sig))
    cardiac_band_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 4.0)])
    total_power = np.sum(psd) + 1e-8
    sqi = (cardiac_band_power / total_power) * 100.0
    return np.clip(sqi, 0.0, 100.0)


# ==========================================
# 4. SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.title("PPG Telemetry Control")
data_source = st.sidebar.radio("Data Mode", ["Synthetic Generator", "Upload Real PPG Stream"])

if data_source == "Upload Real PPG Stream":
    uploaded_file = st.sidebar.file_uploader("Upload CSV Stream", type=["csv", "txt"])
    FS_NATIVE = st.sidebar.number_input("Sensor Sampling Rate (Hz)", min_value=1.0, max_value=1000.0, value=500.0, step=10.0)
    target_hr = 75
    noise_amp, hum_freq, drift_level = 0.0, 50.0, 0.0
else:
    uploaded_file = None
    FS_NATIVE = 50.0
    target_hr = st.sidebar.slider("Simulated Heart Rate (BPM)", 50, 140, 75)
    st.sidebar.subheader("Synthetic Noise Controls")
    noise_amp = st.sidebar.slider("Noise Amplitude", 0.0, 0.8, 0.3, 0.05)
    hum_freq = st.sidebar.slider("Grid Hum Frequency (Hz)", 10, 100, 50, 10)
    drift_level = st.sidebar.slider("Baseline Drift", 0.0, 0.5, 0.2, 0.05)


# ==========================================
# 5. DATA PROCESSING PIPELINE
# ==========================================
TARGET_LENGTH = 200
MODEL_FS = 50.0
duration = 4.0
t = np.linspace(0, duration, TARGET_LENGTH)
true_clean = None

if data_source == "Upload Real PPG Stream" and uploaded_file is not None:
    try:
        full_signal = load_uploaded_signal(uploaded_file)
        window_size = int(FS_NATIVE * duration)
        
        if len(full_signal) > window_size:
            max_start_sec = int((len(full_signal) - window_size) / FS_NATIVE)
            start_sec = st.sidebar.slider("Window Start Offset (seconds)", 0, max(1, max_start_sec), 0)
            start_idx = int(start_sec * FS_NATIVE)
            raw_data = full_signal[start_idx : start_idx + window_size]
        else:
            raw_data = full_signal

        num_target_samples = int(len(raw_data) * (MODEL_FS / FS_NATIVE))
        resampled = signal.resample(raw_data, num_target_samples)
        
        if len(resampled) != TARGET_LENGTH:
            resampled = signal.resample(resampled, TARGET_LENGTH)
            
        raw_noisy = (resampled - np.min(resampled)) / (np.max(resampled) - np.min(resampled) + 1e-8)
        st.sidebar.success(f"Stream Loaded! ({len(full_signal):,} total points)")
    except Exception as e:
        st.sidebar.error(f"Error parsing file: {e}. Reverting to Synthetic.")
        true_clean = generate_synthetic_ppg(t, target_hr)
        noise_gen = random_noise(amplitude=0.3, hum_freq=50, drift_level=0.2, fs=MODEL_FS)
        raw_noisy = noise_gen.generate_noisy_signal(true_clean, t)
else:
    true_clean = generate_synthetic_ppg(t, target_hr)
    noise_gen = random_noise(amplitude=noise_amp, hum_freq=hum_freq, drift_level=drift_level, fs=MODEL_FS)
    raw_noisy = noise_gen.generate_noisy_signal(true_clean, t)


# ==========================================
# 6. MODEL INFERENCE & METRICS
# ==========================================
model = load_trained_model()

# Stage 2: Bandpass Filter
filtered = butter_bandpass_filter(raw_noisy, lowcut=0.5, highcut=4.0, fs=MODEL_FS)
filtered = (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered) + 1e-8)

# Stage 3: U-Net Inference
input_tensor = torch.FloatTensor(raw_noisy).unsqueeze(0).unsqueeze(0)
with torch.no_grad():
    reconstructed = model(input_tensor).squeeze().numpy()

sqi_score = calculate_sqi(reconstructed, fs=MODEL_FS)
min_distance = max(1, int(MODEL_FS * 0.35))
peaks, _ = signal.find_peaks(reconstructed, distance=min_distance, prominence=0.15)

if len(peaks) >= 2:
    est_bpm = 60.0 / np.mean(np.diff(t[peaks]))
    bpm_str = f"{est_bpm:.1f} BPM"
else:
    bpm_str = "N/A"


# ==========================================
# 7. DASHBOARD DISPLAY (RESTORED 4 SUBPLOTS)
# ==========================================
st.title("🫀 Remote PPG Signal Processing Engine")
st.markdown("1D Convolutional U-Net Denoising for Optical Telemetry Streams")

col1, col2, col3 = st.columns(3)
col1.metric("Signal Quality (SQI)", f"{sqi_score:.1f}%")
col2.metric("Estimated Heart Rate", bpm_str)
col3.metric("Detected Systolic Peaks", len(peaks))

# 4 Stacked Subplots
fig = make_subplots(
    rows=4, cols=1, 
    shared_xaxes=True,
    subplot_titles=(
        "Stage 1: Raw Input Window", 
        "Stage 2: SciPy Bandpass Filter (0.5 - 4.0 Hz)", 
        "Stage 3: 1D U-Net Reconstructed Output", 
        "Ground Truth Target"
    ),
    vertical_spacing=0.08
)

# Plot 1: Raw
fig.add_trace(go.Scatter(x=t, y=raw_noisy, mode='lines', name='Raw Input', line=dict(color='crimson')), row=1, col=1)

# Plot 2: Bandpass Filtered
fig.add_trace(go.Scatter(x=t, y=filtered, mode='lines', name='Bandpass Filtered', line=dict(color='orange')), row=2, col=1)

# Plot 3: AI Reconstructed + Peaks
fig.add_trace(go.Scatter(x=t, y=reconstructed, mode='lines', name='1D U-Net Output', line=dict(color='royalblue', width=2)), row=3, col=1)
if len(peaks) > 0:
    fig.add_trace(go.Scatter(x=t[peaks], y=reconstructed[peaks], mode='markers', name='Systolic Peaks', marker=dict(color='purple', size=10)), row=3, col=1)

# Plot 4: Ground Truth Reference
if true_clean is not None:
    fig.add_trace(go.Scatter(x=t, y=true_clean, mode='lines', name='Ground Truth Reference', line=dict(color='green')), row=4, col=1)
else:
    fig.add_trace(go.Scatter(x=t, y=np.zeros_like(t), mode='lines', name='No Reference (Real Stream)', line=dict(color='gray', dash='dash')), row=4, col=1)

fig.update_layout(height=850, showlegend=True)
fig.update_xaxes(title_text="Time (seconds)", row=4, col=1)

st.plotly_chart(fig, use_container_width=True)