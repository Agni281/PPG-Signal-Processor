import streamlit as st
import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go

st.set_page_config(page_title="Remote PPG Signal De-Noiser", layout="wide")

st.title("Remote Biosensor Signal De-Noiser")
st.markdown("**An Edge-AI & Signal Processing Pipeline for Rural Healthcare**")

# --- 1. MODEL DEFINITION ---
class SignalDenoisingUNet1D(nn.Module):
    def __init__(self):
        super(SignalDenoisingUNet1D, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2), nn.ReLU()) # preserved dimensions
        self.dec2 = nn.Sequential(nn.ConvTranspose1d(16, 16, kernel_size=5, stride=2, padding=2, output_padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose1d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        d2 = self.dec2(e2)
        cat1 = torch.cat([d2, e1], dim=1)
        d1 = self.dec1(cat1)
        return d1

# --- FIX 1: DYNAMIC PPG SIGNAL GENERATOR (VARIOUS HR & DUAL PEAKS) ---
def generate_synthetic_ppg(t, hr_bpm):
    """Generates a realistic PPG waveform with systolic and diastolic peaks."""
    freq = hr_bpm / 60.0
    # Primary pulse + secondary dichrotic notch wave
    ppg = np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(4 * np.pi * freq * t + 0.5)
    return (ppg - np.min(ppg)) / (np.max(ppg) - np.min(ppg) + 1e-8)

# --- FIX 3: STANDALONE SIGNAL QUALITY INDEX (SQI) ---
def calculate_sqi(sig, fs=100.0):
    """Calculates Signal Quality Index using spectral energy in cardiac band (0.5 - 4 Hz)."""
    freqs, psd = signal.welch(sig, fs=fs, nperseg=len(sig))
    cardiac_band_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 4.0)])
    total_power = np.sum(psd) + 1e-8
    sqi = (cardiac_band_power / total_power) * 100.0
    return np.clip(sqi, 0.0, 100.0)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Pipeline Configurations")
FS = 100
duration = 2.0
t = np.linspace(0, duration, int(FS * duration))

target_hr = st.sidebar.slider("Simulated Heart Rate (BPM)", 50, 140, 75)
hum_freq = st.sidebar.slider("Grid Noise (Hz)", 10, 60, 50)
drift_level = st.sidebar.slider("Baseline Drift Severity", 0.0, 2.0, 0.8)
noise_amp = st.sidebar.slider("Sensor Noise Level", 0.0, 1.5, 0.4)

# Generate baseline dynamic PPG
true_clean = generate_synthetic_ppg(t, target_hr)

# Corrupt signal
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

# Training loop using randomized PPG parameters
for epoch in range(40):
    optimizer.zero_grad()
    # Randomize sample HR and shifts during training loop
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

# --- FIX 3: STANDALONE CONFIDENCE SCORE ---
confidence_score = calculate_sqi(reconstructed, fs=FS)

# --- FIX 4: REAL CLINICAL METRICS (BPM & HRV / RMSSD) ---
min_distance = max(1, int(FS * 0.4))
peaks, _ = signal.find_peaks(reconstructed, distance=min_distance, prominence=0.1)

if len(peaks) >= 2:
    peak_times = t[peaks]
    rr_intervals = np.diff(peak_times)  # Interval in seconds
    bpm = 60.0 / np.mean(rr_intervals)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals)))) * 1000.0 if len(rr_intervals) > 1 else 0.0

    if 60 <= bpm <= 100:
        triage_msg = f" HEALTHY: Normal Sinus Rhythm ({bpm:.1f} BPM | RMSSD: {rmssd:.1f} ms)"
    else:
        triage_msg = f" WARNING: Abnormal Pulse Frequency Detected ({bpm:.1f} BPM)!"
else:
    bpm = 0.0
    rmssd = 0.0
    triage_msg = " WARNING: Insufficient Peaks Detected for Reliable Calculation."

# --- DASHBOARD DISPLAY ---
col1, col2, col3 = st.columns(3)
col1.metric("Est. Heart Rate (BPM)", f"{bpm:.1f} BPM" if bpm > 0 else "N/A")
col2.metric("Signal Quality Index (SQI)", f"{confidence_score:.1f}%")
col3.metric("HRV RMSSD", f"{rmssd:.1f} ms" if rmssd > 0 else "N/A")

st.info(f"**Diagnostic Triage:** {triage_msg}")

# --- PLOTLY INTERACTIVE CHARTS ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=raw_noisy, mode='lines', name='Raw Sensor Stream', line=dict(color='crimson')))
fig.add_trace(go.Scatter(x=t, y=filtered_signal, mode='lines', name='Butterworth Filter', line=dict(color='orange')))
fig.add_trace(go.Scatter(x=t, y=reconstructed, mode='lines', name='AI Reconstructed Signal', line=dict(color='royalblue', width=2)))

if len(peaks) > 0:
    fig.add_trace(go.Scatter(x=t[peaks], y=reconstructed[peaks], mode='markers', name='Systolic Peaks',
                             marker=dict(color='purple', size=10, symbol='diamond')))

fig.update_layout(title="Multi-Stage Biosensor Signal Processing Pipeline",
                   xaxis_title="Time (seconds)", yaxis_title="Normalized Amplitude",
                   template="plotly_white", height=500)

st.plotly_chart(fig, use_container_width=True)

# --- BATCH EVALUATION SECTION ---
st.markdown("---")
st.subheader("Automated Batch Stress Test")
if st.button("Run Evaluation Batch"):
    eval_sqis = []
    eval_bpms = []
    
    for _ in range(20):
        rand_hr = np.random.randint(55, 120)
        s_clean = generate_synthetic_ppg(t, rand_hr)
        s_noisy = s_clean + noise_amp * np.sin(2 * np.pi * hum_freq * t) + drift_level * np.sin(2 * np.pi * 0.2 * t)
        s_noisy = (s_noisy - np.min(s_noisy)) / (np.max(s_noisy) - np.min(s_noisy) + 1e-8)
        
        inp = torch.FloatTensor(s_noisy).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            rec = model(inp).squeeze().numpy()
        
        eval_sqis.append(calculate_sqi(rec, fs=FS))
        pks, _ = signal.find_peaks(rec, distance=min_distance, prominence=0.1)
        if len(pks) >= 2:
            eval_bpms.append(60.0 / np.mean(np.diff(t[pks])))

    eval_col1, eval_col2 = st.columns(2)
    eval_col1.metric("Mean Quality SQI", f"{np.mean(eval_sqis):.1f}%")
    eval_col2.metric("Mean Estimated BPM", f"{np.mean(eval_bpms):.1f} BPM" if len(eval_bpms) > 0 else "N/A")