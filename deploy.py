import streamlit as st
import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Remote PPG Signal De-Noiser", layout="wide")

st.title("Remote Biosensor Signal De-Noiser")
st.markdown("**An Edge-AI & Signal Processing Pipeline for Rural Healthcare**")

# --- 1. MODEL DEFINITION ---
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
        cat1 = torch.cat([d2, e1], dim=1)
        d1 = self.dec1(cat1)
        return d1

# --- DYNAMIC PPG GENERATOR ---
def generate_synthetic_ppg(t, hr_bpm):
    freq = hr_bpm / 60.0
    ppg = np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(4 * np.pi * freq * t + 0.5)
    return (ppg - np.min(ppg)) / (np.max(ppg) - np.min(ppg) + 1e-8)

# --- STANDALONE QUALITY METRIC (SQI) ---
def calculate_sqi(sig, fs=100.0):
    freqs, psd = signal.welch(sig, fs=fs, nperseg=len(sig))
    cardiac_band_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 4.0)])
    total_power = np.sum(psd) + 1e-8
    sqi = (cardiac_band_power / total_power) * 100.0
    return np.clip(sqi, 0.0, 100.0)

# --- SIDEBAR CONTROLS (YOUR ORIGINAL VARIABLES) ---
st.sidebar.header("Pipeline Configurations")
FS = 100
duration = 2.0
t = np.linspace(0, duration, int(FS * duration))

target_hr = st.sidebar.slider("Simulated Heart Rate (BPM)", 50, 140, 75)
hum_freq = st.sidebar.slider("Grid Noise (Hz)", 10, 60, 50)
drift_level = st.sidebar.slider("Baseline Drift Severity", 0.0, 2.0, 0.8)
noise_amp = st.sidebar.slider("Sensor Noise Level", 0.0, 1.5, 0.4)

# Signal synthesis and corruption
true_clean = generate_synthetic_ppg(t, target_hr)
grid_hum = noise_amp * np.sin(2 * np.pi * hum_freq * t)
baseline_drift = drift_level * np.sin(2 * np.pi * 0.2 * t)
random_noise = np.random.normal(0, noise_amp * 0.3, len(t))

raw_noisy = true_clean + grid_hum + baseline_drift + random_noise
raw_noisy = (raw_noisy - np.min(raw_noisy)) / (np.max(raw_noisy) - np.min(raw_noisy) + 1e-8)

# Classical Filter Stage
b, a = signal.butter(2, [0.5 / (0.5 * FS), 4.0 / (0.5 * FS)], btype='band')
filtered_signal = signal.filtfilt(b, a, raw_noisy)

# Model Training Setup
model = SignalDenoisingUNet1D()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(40):
    optimizer.zero_grad()
    rand_hr = np.random.randint(50, 130)
    sample_clean = generate_synthetic_ppg(t, rand_hr)
    sample_noise = noise_amp * np.sin(2 * np.pi * hum_freq * t) + drift_level * np.sin(2 * np.pi * 0.2 * t)
    sample_noisy = sample_clean + sample_noise
    sample_noisy = (sample_noisy - np.min(sample_noisy)) / (np.max(sample_noisy) - np.min(sample_noisy) + 1e-8)

    inp = torch.FloatTensor(sample_noisy).unsqueeze(0).unsqueeze(0)
    tgt = torch.FloatTensor(sample_clean).unsqueeze(0).unsqueeze(0)

    out = model(inp)
    loss = criterion(out, tgt)
    loss.backward()
    optimizer.step()

# Inference
model.eval()
with torch.no_grad():
    inp_tensor = torch.FloatTensor(raw_noisy).unsqueeze(0).unsqueeze(0)
    reconstructed = model(inp_tensor).squeeze().numpy()

# Metrics
confidence_score = calculate_sqi(reconstructed, fs=FS)
min_distance = max(1, int(FS * 0.4))
peaks, _ = signal.find_peaks(reconstructed, distance=min_distance, prominence=0.1)

if len(peaks) >= 2:
    rr_intervals = np.diff(t[peaks])
    bpm = 60.0 / np.mean(rr_intervals)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals)))) * 1000.0 if len(rr_intervals) > 1 else 0.0
    triage_msg = f"✅ HEALTHY: Normal Sinus Rhythm ({bpm:.1f} BPM | RMSSD: {rmssd:.1f} ms)" if 60 <= bpm <= 100 else f"🚨 WARNING: Abnormal Pulse Frequency ({bpm:.1f} BPM)!"
else:
    bpm, rmssd = 0.0, 0.0
    triage_msg = "⚠️ WARNING: Insufficient Peaks Detected."

# Metrics Display
col1, col2, col3 = st.columns(3)
col1.metric("Est. Heart Rate (BPM)", f"{bpm:.1f} BPM" if bpm > 0 else "N/A")
col2.metric("Signal Quality Index (SQI)", f"{confidence_score:.1f}%")
col3.metric("HRV RMSSD", f"{rmssd:.1f} ms" if rmssd > 0 else "N/A")

st.info(f"**Diagnostic Triage:** {triage_msg}")

# --- RESTORED 4 STACKED PLOTS ---
fig = make_subplots(rows=4, cols=1, subplot_titles=(
    f"1. Raw Input Signal (Grid Hum: {hum_freq}Hz | Drift: {drift_level:.2f})",
    "2. Stage 2: Classical Bandpass Filter (0.5 - 4.0 Hz)",
    f"3. AI Reconstructed Signal (SQI Score: {confidence_score:.1f}%)",
    "4. Ground Truth Baseline Target"
))

# Subplot 1: Raw Signal
fig.add_trace(go.Scatter(x=t, y=raw_noisy, mode='lines', name='Raw Signal', line=dict(color='crimson')), row=1, col=1)

# Subplot 2: Butterworth Filtered
fig.add_trace(go.Scatter(x=t, y=filtered_signal, mode='lines', name='Filtered', line=dict(color='darkorange')), row=2, col=1)

# Subplot 3: AI Denoised + Peaks
fig.add_trace(go.Scatter(x=t, y=reconstructed, mode='lines', name='AI Reconstructed', line=dict(color='royalblue', width=2)), row=3, col=1)
if len(peaks) > 0:
    fig.add_trace(go.Scatter(x=t[peaks], y=reconstructed[peaks], mode='markers', name='Systolic Peaks',
                             marker=dict(color='purple', size=9, symbol='diamond')), row=3, col=1)

# Subplot 4: Ground Truth
fig.add_trace(go.Scatter(x=t, y=true_clean, mode='lines', name='Target Reference', line=dict(color='forestgreen', dash='dash')), row=4, col=1)

fig.update_layout(height=800, template="plotly_white", showlegend=False)
st.plotly_chart(fig, use_container_width=True)