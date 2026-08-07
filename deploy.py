import streamlit as st
import numpy as np
import scipy.signal as signal
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
    return signal.filtfilt(b, a, data)

@st.cache_data
def load_default_ppg_data(length=200):
    """Generates a benchmark physiological PPG waveform."""
    t = np.linspace(0, 4 * np.pi, length)
    clean_wave = np.sin(t) + 0.5 * np.sin(2 * t) + 0.2 * np.sin(3 * t)
    return (clean_wave - np.min(clean_wave)) / (np.max(clean_wave) - np.min(clean_wave))

@st.cache_resource
def train_model():
    """Trains the PyTorch autoencoder model once and caches it."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    length = 200
    t = np.linspace(0, 4 * np.pi, length)
    clean_dataset, noisy_dataset = [], []
    
    for _ in range(500):
        clean_wave = np.sin(t) + 0.5 * np.sin(2 * t) + 0.2 * np.sin(3 * t)
        clean_wave = (clean_wave - np.min(clean_wave)) / (np.max(clean_wave) - np.min(clean_wave))
        
        noisy_wave = clean_wave + 0.3 * np.sin(50 * t) + 0.2 * np.sin(0.2 * t) + np.random.normal(0, 0.15, length)
        noisy_wave = (noisy_wave - np.min(noisy_wave)) / (np.max(noisy_wave) - np.min(noisy_wave))
        
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

# --- 3. SIDEBAR CONTROLS & FILE UPLOADER ---
st.sidebar.header("1. Upload Custom PPG Signal")
uploaded_file = st.sidebar.file_uploader(
    "Upload 1D CSV/TXT File", 
    type=["csv", "txt"],
    help="Upload a file containing single-column numeric PPG sensor readings."
)

st.sidebar.markdown("---")
st.sidebar.header("2. Synthetic Noise Parameters")
st.sidebar.markdown("Simulate environmental and device interference:")

noise_amp = st.sidebar.slider("Noise Amplitude", 0.0, 0.8, 0.3, 0.05)
hum_freq = st.sidebar.slider("Grid Hum Frequency (Hz)", 10, 100, 50, 10)
drift_level = st.sidebar.slider("Baseline Drift (Breathing)", 0.0, 0.5, 0.2, 0.05)

# --- 4. DATA PROCESSING PIPELINE ---
FS = 50.0  # Sampling frequency in Hz
length = 200
t = np.linspace(0, 4 * np.pi, length)

is_custom_file = False

if uploaded_file is not None:
    try:
        raw_data = np.loadtxt(uploaded_file, delimiter=',')
        if raw_data.ndim > 1:
            raw_data = raw_data[:, 0]
            
        if len(raw_data) >= length:
            true_clean = raw_data[:length]
        else:
            true_clean = np.pad(raw_data, (0, length - len(raw_data)), mode='edge')
            
        true_clean = (true_clean - np.min(true_clean)) / (np.max(true_clean) - np.min(true_clean) + 1e-8)
        is_custom_file = True
        st.sidebar.success(" Custom PPG Signal Loaded!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}. Defaulting to benchmark signal.")
        true_clean = load_default_ppg_data(length)
else:
    true_clean = load_default_ppg_data(length)

# Add interference
high_freq_noise = noise_amp * np.sin(hum_freq * t)
baseline_drift = drift_level * np.sin(0.2 * t)
random_noise = np.random.normal(0, noise_amp * 0.5, length)

raw_noisy = true_clean + high_freq_noise + baseline_drift + random_noise
raw_noisy = (raw_noisy - np.min(raw_noisy)) / (np.max(raw_noisy) - np.min(raw_noisy) + 1e-8)

# Bandpass Filter Stage
filtered_signal = butter_bandpass_filter(raw_noisy, lowcut=0.5, highcut=4.0, fs=FS)
filtered_signal = (filtered_signal - np.min(filtered_signal)) / (np.max(filtered_signal) - np.min(filtered_signal) + 1e-8)

# Signal Metrics
signal_power = np.mean(true_clean ** 2)
noise_power = np.mean((raw_noisy - true_clean) ** 2)
snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 100.0

# AI Model Inference
model.eval()
with torch.no_grad():
    input_sample = torch.FloatTensor(raw_noisy).unsqueeze(0).unsqueeze(0)
    reconstructed = model(input_sample).squeeze().numpy()

# Confidence Score (Pearson Correlation)
correlation_matrix = np.corrcoef(reconstructed, true_clean)
raw_corr = correlation_matrix[0, 1]
confidence_score = 0.0 if np.isnan(raw_corr) else float(raw_corr) * 100.0

# Peak Detection & Heart Rate
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
    if is_custom_file:
        st.info(" **Source:** Processing Custom Uploaded PPG Stream")
    else:
        st.info(" **Source:** Using Built-in Benchmark Physiological PPG Wave")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Input SNR", f"{snr_db:.2f} dB")
    col2.metric("AI Confidence", f"{confidence_score:.1f}%")
    col3.metric("Detected Peaks", f"{len(peaks)}")
    col4.metric("Heart Rate", f"{heart_rate_bpm:.1f} BPM" if heart_rate_bpm > 0 else "N/A")

    st.markdown("---")

    if confidence_score < 70.0:
        st.error(f" **Conclusion: UNRELIABLE DATA** — Environmental noise too severe for diagnostic assessment (AI Confidence: {confidence_score:.1f}%).")
    elif interval_variance > 0.05 and heart_rate_bpm > 0:
        st.warning(f" **Conclusion: ALERT** — High Heart Rate Variability / Potential Arrhythmia detected (AI Confidence: {confidence_score:.1f}%).")
    else:
        st.success(f" **Conclusion: HEALTHY** — Stable rhythm detected (AI Confidence: {confidence_score:.1f}%).")

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
    axs[2].set_title(f"Stage 3: AI Reconstructed Wave & Feature Extraction (Confidence: {confidence_score:.1f}%)")
    axs[2].legend(loc="upper right")

    axs[3].plot(true_clean, color='forestgreen', linestyle='--', label='Ground Truth Reference')
    axs[3].set_title("Stage 4: Ground Truth Signal Reference")
    axs[3].legend(loc="upper right")

    plt.tight_layout()
    st.pyplot(fig)

with tabs[1]:
    st.subheader("Model Evaluation Across Multiple Test Samples")
    st.markdown("Evaluates the pipeline across 50 test iterations under current noise settings.")
    
    if st.button("Run Batch Evaluation"):
        eval_confidences, bpm_list = [], []
        
        for _ in range(50):
            shift = np.random.randint(-15, 15)
            sample_clean = np.roll(true_clean, shift)
            sample_noise = noise_amp * np.sin(hum_freq * t) + drift_level * np.sin(0.2 * t) + np.random.normal(0, noise_amp * 0.5, length)
            sample_noisy = (sample_clean + sample_noise)
            sample_noisy = (sample_noisy - np.min(sample_noisy)) / (np.max(sample_noisy) - np.min(sample_noisy) + 1e-8)
            
            inp = torch.FloatTensor(sample_noisy).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                rec = model(inp).squeeze().numpy()
            
            raw_c = np.corrcoef(rec, sample_clean)[0, 1]
            conf = 0.0 if np.isnan(raw_c) else float(raw_c) * 100.0
            eval_confidences.append(conf)
            
            pks, _ = signal.find_peaks(rec, distance=int(FS * 0.4), prominence=0.15)
            if len(pks) > 1:
                bpm_list.append(60.0 / (np.mean(np.diff(pks)) / FS))

        eval_col1, eval_col2 = st.columns(2)
        eval_col1.metric("Mean AI Confidence", f"{np.mean(eval_confidences):.1f}%")
        eval_col2.metric("Mean Estimated BPM", f"{np.mean(bpm_list):.1f}" if len(bpm_list) > 0 else "N/A")
        
        fig_eval, ax_eval = plt.subplots(figsize=(10, 3))
        ax_eval.hist(eval_confidences, bins=15, color='royalblue', edgecolor='black')
        ax_eval.set_title("Distribution of AI Confidence Scores (%)")
        ax_eval.set_xlabel("Confidence Score (%)")
        ax_eval.set_ylabel("Sample Count")
        st.pyplot(fig_eval)