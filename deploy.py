import streamlit as st
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import scipy.datasets as datasets
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Remote PPG Signal De-Noiser", layout="wide")

st.title("Remote Biosensor Signal De-Noiser")
st.markdown("""
**An Edge-AI & Signal Processing Pipeline for Rural Healthcare**
""")

# --- 1. U-NET 1D MODEL DEFINITION (WITH SKIP CONNECTIONS) ---
class UNet1D(nn.Module):
    def __init__(self):
        super(UNet1D, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        # Decoder with Skip Connections
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)  # [B, 16, L/2]
        e2 = self.enc2(e1) # [B, 32, L/4]
        
        d2 = self.dec2(e2) # [B, 16, L/2]
        d2_cat = torch.cat([d2, e1], dim=1) # Skip connection concat [B, 32, L/2]
        
        out = self.dec1(d2_cat) # [B, 1, L]
        return out

# --- 2. SIGNAL PROCESSING & UTILITIES ---
def butter_bandpass_filter(data, lowcut=0.5, highcut=4.0, fs=50.0, order=2):
    """Bandpass filter to eliminate baseline wander (<0.5Hz) and high frequency noise (>4Hz)."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def calculate_sqi(signal_data):
    """Calculates Signal Quality Index based on Skewness and Kurtosis."""
    skew = stats.skew(signal_data)
    kurt = stats.kurtosis(signal_data)
    # Good PPG signals typically exhibit positive skewness and defined kurtosis
    sqi_score = max(0.0, min(100.0, (skew * 20.0) + (kurt * 10.0) + 50.0))
    return sqi_score

@st.cache_data
def load_real_ppg_data(length=200):
    """Loads realistic physiological signal segment from SciPy datasets."""
    try:
        ecg_data = datasets.electrocardiogram()
        segment = ecg_data[2000:2000+length]
        return (segment - np.min(segment)) / (np.max(segment) - np.min(segment))
    except Exception:
        t = np.linspace(0, 4 * np.pi, length)
        wave = np.sin(t) + 0.5 * np.sin(2 * t)
        return (wave - np.min(wave)) / (np.max(wave) - np.min(wave))

@st.cache_resource
def train_model():
    """Trains the 1D U-Net PyTorch model once and caches it."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    length = 200
    base_signal = load_real_ppg_data(length)
    clean_dataset, noisy_dataset = [], []
    
    for _ in range(500):
        shift = np.random.randint(-10, 10)
        clean_wave = np.roll(base_signal, shift)
        
        t = np.linspace(0, 4 * np.pi, length)
        noisy_wave = clean_wave + 0.2 * np.sin(50 * t) + 0.15 * np.sin(0.2 * t) + np.random.normal(0, 0.1, length)
        noisy_wave = (noisy_wave - np.min(noisy_wave)) / (np.max(noisy_wave) - np.min(noisy_wave) + 1e-8)
        
        clean_dataset.append(clean_wave)
        noisy_dataset.append(noisy_wave)
        
    clean_tensor = torch.FloatTensor(np.array(clean_dataset)).unsqueeze(1)
    noisy_tensor = torch.FloatTensor(np.array(noisy_dataset)).unsqueeze(1)
    
    model = UNet1D()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    for epoch in range(45):
        permutation = torch.randperm(noisy_tensor.size(0))
        for i in range(0, noisy_tensor.size(0), 32):
            indices = permutation[i:i+32]
            optimizer.zero_grad()
            outputs = model(noisy_tensor[indices])
            loss = criterion(outputs, clean_tensor[indices])
            loss.backward()
            optimizer.step()
            
    return model

model = train_model()

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("Data Input & Noise Simulation")

uploaded_file = st.sidebar.file_uploader("Upload Raw PPG CSV (Optional)", type=["csv", "txt"])

noise_amp = st.sidebar.slider("Noise Amplitude", 0.0, 0.8, 0.3, 0.05)
hum_freq = st.sidebar.slider("Grid Hum Frequency (Hz)", 10, 100, 50, 10)
drift_level = st.sidebar.slider("Baseline Drift (Breathing)", 0.0, 0.5, 0.2, 0.05)

# --- 4. PIPELINE EXECUTION ---
FS = 50.0  # Sampling frequency in Hz
length = 200
t = np.linspace(0, length / FS, length)

if uploaded_file is not None:
    try:
        raw_input = np.loadtxt(uploaded_file, delimiter=',')[:length]
        if len(raw_input) < length:
            raw_input = np.pad(raw_input, (0, length - len(raw_input)), 'edge')
        true_clean = (raw_input - np.min(raw_input)) / (np.max(raw_input) - np.min(raw_input) + 1e-8)
    except Exception:
        st.sidebar.error("Error reading file. Using benchmark PPG.")
        true_clean = load_real_ppg_data(length)
else:
    true_clean = load_real_ppg_data(length)

# Interference simulation
high_freq_noise = noise_amp * np.sin(2 * np.pi * hum_freq * t)
baseline_drift = drift_level * np.sin(2 * np.pi * 0.2 * t)
random_noise = np.random.normal(0, noise_amp * 0.3, length)

raw_noisy = true_clean + high_freq_noise + baseline_drift + random_noise
raw_noisy = (raw_noisy - np.min(raw_noisy)) / (np.max(raw_noisy) - np.min(raw_noisy) + 1e-8)

# Filtering & Denoising
filtered_signal = butter_bandpass_filter(raw_noisy, lowcut=0.5, highcut=4.0, fs=FS)
filtered_signal = (filtered_signal - np.min(filtered_signal)) / (np.max(filtered_signal) - np.min(filtered_signal) + 1e-8)

model.eval()
with torch.no_grad():
    input_sample = torch.FloatTensor(raw_noisy).unsqueeze(0).unsqueeze(0)
    reconstructed = model(input_sample).squeeze().numpy()

# Metrics
mse_score = np.mean((reconstructed - true_clean) ** 2)
raw_ncc = np.corrcoef(reconstructed, true_clean)[0, 1]
ncc_score = 0.0 if np.isnan(raw_ncc) else float(raw_ncc)
confidence_score = max(0.0, ncc_score * np.exp(-2.0 * mse_score)) * 100.0
sqi_score = calculate_sqi(reconstructed)

# Heart Rate Calculation
peaks, _ = signal.find_peaks(reconstructed, distance=int(FS * 0.4), prominence=0.15)
if len(peaks) > 1:
    peak_intervals_sec = np.diff(peaks) / FS
    mean_interval = np.mean(peak_intervals_sec)
    heart_rate_bpm = 60.0 / mean_interval if mean_interval > 0 else 0
    interval_variance = np.var(peak_intervals_sec)
else:
    heart_rate_bpm = 0
    interval_variance = 0.0

# --- 5. DASHBOARD LAYOUT ---
tabs = st.tabs(["Interactive Signal Analysis", "Batch Performance Suite"])

with tabs[0]:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("AI Confidence", f"{confidence_score:.1f}%")
    col2.metric("Signal Quality (SQI)", f"{sqi_score:.1f}")
    col3.metric("Reconstruction MSE", f"{mse_score:.4f}")
    col4.metric("Heart Rate", f"{heart_rate_bpm:.1f} BPM" if heart_rate_bpm > 0 else "N/A")
    col5.metric("Detected Peaks", len(peaks))

    st.markdown("---")

    if confidence_score < 60.0:
        st.error(f" **Conclusion: UNRELIABLE DATA** — Low AI confidence ({confidence_score:.1f}%). High interference detected.")
    elif interval_variance > 0.05 and heart_rate_bpm > 0:
        st.warning(f" **Conclusion: ALERT** — High Heart Rate Variability / Arrhythmia pattern detected.")
    else:
        st.success(f" **Conclusion: HEALTHY** — Stable physiological rhythm reconstructed.")

    # Interactive Plotly Visualization
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Stage 1: Raw Telemetry Stream (Rural Clinic)",
                                        "Stage 2: SciPy Bandpass Filtered (0.5 - 4.0 Hz)",
                                        "Stage 3: 1D U-Net AI Reconstruction & Peak Detection",
                                        "Stage 4: Ground Truth Reference"))

    fig.add_trace(go.Scatter(y=raw_noisy, mode='lines', name='Raw Noisy', line=dict(color='crimson')), row=1, col=1)
    fig.add_trace(go.Scatter(y=filtered_signal, mode='lines', name='Bandpass Filtered', line=dict(color='darkorange')), row=2, col=1)
    fig.add_trace(go.Scatter(y=reconstructed, mode='lines', name='U-Net Denoised', line=dict(color='royalblue', width=2)), row=3, col=1)
    
    if len(peaks) > 0:
        fig.add_trace(go.Scatter(x=peaks, y=reconstructed[peaks], mode='markers', name='Systolic Peaks',
                                 marker=dict(color='darkmagenta', size=10, symbol='diamond')), row=3, col=1)
        
    fig.add_trace(go.Scatter(y=true_clean, mode='lines', name='Ground Truth', line=dict(color='forestgreen', dash='dash')), row=4, col=1)

    fig.update_layout(height=850, title_text="End-to-End Signal Processing Telemetry Pipeline", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("Model Evaluation Across Multiple Test Samples")
    if st.button("Run Batch Evaluation"):
        eval_mses, eval_confidences, bpm_list = [], [], []
        
        for _ in range(50):
            shift = np.random.randint(-15, 15)
            sample_clean = np.roll(true_clean, shift)
            sample_noise = noise_amp * np.sin(2 * np.pi * hum_freq * t) + drift_level * np.sin(2 * np.pi * 0.2 * t) + np.random.normal(0, noise_amp * 0.3, length)
            sample_noisy = (sample_clean + sample_noise)
            sample_noisy = (sample_noisy - np.min(sample_noisy)) / (np.max(sample_noisy) - np.min(sample_noisy) + 1e-8)
            
            inp = torch.FloatTensor(sample_noisy).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                rec = model(inp).squeeze().numpy()
            
            mse = np.mean((rec - sample_clean) ** 2)
            raw_c = np.corrcoef(rec, sample_clean)[0, 1]
            ncc = 0.0 if np.isnan(raw_c) else float(raw_c)
            conf = max(0.0, ncc * np.exp(-2.0 * mse)) * 100.0
            
            eval_mses.append(mse)
            eval_confidences.append(conf)
            
            pks, _ = signal.find_peaks(rec, distance=int(FS * 0.4), prominence=0.15)
            if len(pks) > 1:
                bpm_list.append(60.0 / (np.mean(np.diff(pks)) / FS))

        eval_col1, eval_col2, eval_col3 = st.columns(3)
        eval_col1.metric("Mean AI Confidence", f"{np.mean(eval_confidences):.1f}%")
        eval_col2.metric("Average Test MSE", f"{np.mean(eval_mses):.4f}")
        eval_col3.metric("Mean Estimated BPM", f"{np.mean(bpm_list):.1f}" if len(bpm_list) > 0 else "N/A")