import streamlit as st
import numpy as np
import scipy.signal as signal
import scipy.datasets as datasets
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

st.set_page_config(page_title="Remote PPG Signal De-Noiser", layout="wide")

st.title("Remote Biosensor Signal De-Noiser")
st.markdown("""
**An Edge-AI & Signal Processing Pipeline for Rural Healthcare**
""")

# --- 1. MODEL DEFINITION ---
class SignalDenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(SignalDenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# --- 2. DATA LOADERS & FILTERING ---
def butter_bandpass_filter(data, lowcut=0.5, highcut=4.0, fs=50.0, order=2):
    """Bandpass filter to eliminate baseline wander (<0.5Hz) and high frequency noise (>4Hz)."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    return y

@st.cache_data
def load_real_ppg_data(length=200):
    """Loads realistic physiological signal segment from SciPy datasets."""
    try:
        ecg_data = datasets.electrocardiogram()
        segment = ecg_data[2000:2000+length]
        # Normalize to [0, 1]
        segment = (segment - np.min(segment)) / (np.max(segment) - np.min(segment))
        return segment
    except Exception:
        # Fallback physiological waveform if offline
        t = np.linspace(0, 4 * np.pi, length)
        wave = np.sin(t) + 0.5 * np.sin(2 * t)
        return (wave - np.min(wave)) / (np.max(wave) - np.min(wave))

@st.cache_resource
def train_model():
    """Trains the base PyTorch autoencoder model once and caches it."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    length = 200
    base_signal = load_real_ppg_data(length)
    clean_dataset, noisy_dataset = [], []
    
    for _ in range(500):
        # Shift/scale variations
        shift = np.random.randint(-10, 10)
        clean_wave = np.roll(base_signal, shift)
        
        t = np.linspace(0, 4 * np.pi, length)
        noisy_wave = clean_wave + 0.2 * np.sin(50 * t) + 0.15 * np.sin(0.2 * t) + np.random.normal(0, 0.1, length)
        noisy_wave = (noisy_wave - np.min(noisy_wave)) / (np.max(noisy_wave) - np.min(noisy_wave) + 1e-8)
        
        clean_dataset.append(clean_wave)
        noisy_dataset.append(noisy_wave)
        
    clean_tensor = torch.FloatTensor(np.array(clean_dataset)).unsqueeze(1)
    noisy_tensor = torch.FloatTensor(np.array(noisy_dataset)).unsqueeze(1)
    
    model = SignalDenoisingAutoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    for epoch in range(40):
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
st.sidebar.header("Signal Noise Parameters")
st.sidebar.markdown("Simulate environmental and device interference:")

noise_amp = st.sidebar.slider("Noise Amplitude", 0.0, 0.8, 0.3, 0.05)
hum_freq = st.sidebar.slider("Grid Hum Frequency (Hz)", 10, 100, 50, 10)
drift_level = st.sidebar.slider("Baseline Drift (Breathing)", 0.0, 0.5, 0.2, 0.05)

# --- 4. DATA PROCESSING PIPELINE ---
FS = 50.0  # Sampling frequency in Hz
length = 200
t = np.linspace(0, length / FS, length)

# Real ground truth reference PPG
true_clean = load_real_ppg_data(length)

# Generate synthetic interference
high_freq_noise = noise_amp * np.sin(2 * np.pi * hum_freq * t)
baseline_drift = drift_level * np.sin(2 * np.pi * 0.2 * t)
random_noise = np.random.normal(0, noise_amp * 0.3, length)

raw_noisy = true_clean + high_freq_noise + baseline_drift + random_noise
raw_noisy = (raw_noisy - np.min(raw_noisy)) / (np.max(raw_noisy) - np.min(raw_noisy) + 1e-8)

# Bandpass Filter Stage
filtered_signal = butter_bandpass_filter(raw_noisy, lowcut=0.5, highcut=4.0, fs=FS)
filtered_signal = (filtered_signal - np.min(filtered_signal)) / (np.max(filtered_signal) - np.min(filtered_signal) + 1e-8)

# AI Autoencoder Denoising Stage
model.eval()
with torch.no_grad():
    input_sample = torch.FloatTensor(raw_noisy).unsqueeze(0).unsqueeze(0)
    reconstructed = model(input_sample).squeeze().numpy()

# Real Evaluation Metrics (MSE & Correlation)
mse_score = np.mean((reconstructed - true_clean) ** 2)
ncc_score = np.corrcoef(reconstructed, true_clean)[0, 1]

# SNR Calculation
signal_power = np.mean(true_clean ** 2)
noise_power = np.mean((raw_noisy - true_clean) ** 2)
snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 100.0

# Real Heart Rate Calculation (BPM)
peaks, _ = signal.find_peaks(reconstructed, distance=int(FS * 0.4), prominence=0.15)
if len(peaks) > 1:
    peak_intervals_sec = np.diff(peaks) / FS
    mean_interval = np.mean(peak_intervals_sec)
    heart_rate_bpm = 60.0 / mean_interval if mean_interval > 0 else 0
    interval_variance = np.var(peak_intervals_sec)
else:
    heart_rate_bpm = 0
    interval_variance = 0.0

# --- 5. DASHBOARD DISPLAY ---
tabs = st.tabs(["Single Signal Analysis", "Multi-Sample Evaluation Suite"])

with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Input SNR", f"{snr_db:.2f} dB")
    col2.metric("Reconstruction MSE", f"{mse_score:.4f}")
    col3.metric("Heart Rate", f"{heart_rate_bpm:.1f} BPM" if heart_rate_bpm > 0 else "N/A")
    col4.metric("Cross-Correlation", f"{ncc_score:.2f}")

    st.markdown("---")

    if ncc_score < 0.5 or mse_score > 0.08:
        st.error(" **Conclusion: UNRELIABLE DATA** — Environmental noise too severe for diagnostic assessment.")
    elif interval_variance > 0.05 and heart_rate_bpm > 0:
        st.warning(" **Conclusion: ALERT** — High Heart Rate Variability / Potential Arrhythmia detected.")
    else:
        st.success(" **Conclusion: HEALTHY** — Stable sinus rhythm detected.")

    fig, axs = plt.subplots(4, 1, figsize=(12, 10))

    axs[0].plot(raw_noisy, color='crimson', label='Raw Noisy Signal')
    axs[0].set_title("Stage 1: Raw Telemetry Stream (Rural Clinic)")
    axs[0].legend(loc="upper right")

    axs[1].plot(filtered_signal, color='darkorange', label='SciPy Bandpass Filtered (0.5 - 4.0 Hz)')
    axs[1].set_title("Stage 2: Classical Baseline & High-Pass Filter")
    axs[1].legend(loc="upper right")

    axs[2].plot(reconstructed, color='royalblue', linewidth=2, label='AI Denoised Signal')
    if len(peaks) > 0:
        axs[2].scatter(peaks, reconstructed[peaks], color='darkmagenta', s=80, zorder=5, label=f'Systolic Peaks ({len(peaks)})')
    axs[2].set_title(f"Stage 3: AI Reconstructed Wave & Peak Detection (MSE: {mse_score:.4f})")
    axs[2].legend(loc="upper right")

    axs[3].plot(true_clean, color='forestgreen', linestyle='--', label='Ground Truth Reference')
    axs[3].set_title("Ground Truth Signal Reference")
    axs[3].legend(loc="upper right")

    plt.tight_layout()
    st.pyplot(fig)

with tabs[1]:
    st.subheader("Model Evaluation Across Multiple Test Samples")
    st.markdown("Evaluates the denoising pipeline performance across 50 test iterations under current noise settings.")
    
    if st.button("Run Batch Evaluation"):
        eval_mses = []
        snr_improvements = []
        bpm_list = []
        
        for _ in range(50):
            shift = np.random.randint(-15, 15)
            sample_clean = np.roll(true_clean, shift)
            
            sample_noise = noise_amp * np.sin(2 * np.pi * hum_freq * t) + drift_level * np.sin(2 * np.pi * 0.2 * t) + np.random.normal(0, noise_amp * 0.3, length)
            sample_noisy = sample_clean + sample_noise
            sample_noisy = (sample_noisy - np.min(sample_noisy)) / (np.max(sample_noisy) - np.min(sample_noisy) + 1e-8)
            
            inp = torch.FloatTensor(sample_noisy).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                rec = model(inp).squeeze().numpy()
            
            mse = np.mean((rec - sample_clean) ** 2)
            eval_mses.append(mse)
            
            # Heart rate
            pks, _ = signal.find_peaks(rec, distance=int(FS * 0.4), prominence=0.15)
            if len(pks) > 1:
                bpm_list.append(60.0 / (np.mean(np.diff(pks)) / FS))

        eval_col1, eval_col2, eval_col3 = st.columns(3)
        eval_col1.metric("Average Test MSE", f"{np.mean(eval_mses):.4f}")
        eval_col2.metric("MSE Std Dev", f"{np.std(eval_mses):.4f}")
        eval_col3.metric("Mean Estimated BPM", f"{np.mean(bpm_list):.1f}" if len(bpm_list) > 0 else "N/A")
        
        fig_eval, ax_eval = plt.subplots(figsize=(10, 3))
        ax_eval.hist(eval_mses, bins=15, color='royalblue', edgecolor='black')
        ax_eval.set_title("Distribution of Reconstruction Errors (MSE)")
        ax_eval.set_xlabel("Mean Squared Error")
        ax_eval.set_ylabel("Sample Count")
        st.pyplot(fig_eval)