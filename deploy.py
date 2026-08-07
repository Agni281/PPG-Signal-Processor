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
    """Bandpass filter to eliminate baseline wander and high-frequency noise."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

@st.cache_data
def load_default_signal(length=200):
    """Generates a benchmark reference wave."""
    t = np.linspace(0, 4 * np.pi, length)
    clean_wave = np.sin(t) + 0.5 * np.sin(2 * t) + 0.2 * np.sin(3 * t)
    return (clean_wave - np.min(clean_wave)) / (np.max(clean_wave) - np.min(clean_wave) + 1e-8)

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
        clean_wave = (clean_wave - np.min(clean_wave)) / (np.max(clean_wave) - np.min(clean_wave) + 1e-8)
        
        noisy_wave = clean_wave + 0.3 * np.sin(50 * t) + 0.2 * np.sin(0.2 * t) + np.random.normal(0, 0.15, length)
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

# --- 3. SIDEBAR CONTROLS & FILE UPLOADER ---
st.sidebar.header("1. Upload Custom Sensor Signal")
uploaded_file = st.sidebar.file_uploader(
    "Upload 1D CSV/TXT File", 
    type=["csv", "txt"],
    help="Upload a file containing single-column numeric sensor readings."
)

st.sidebar.markdown("---")
st.sidebar.header("2. Noise Simulation Parameters")

noise_amp = st.sidebar.slider("Noise Amplitude", 0.0, 0.8, 0.3, 0.05)
hum_freq = st.sidebar.slider("Power Line Hum (Hz)", 10, 100, 50, 10)
drift_level = st.sidebar.slider("Baseline Drift", 0.0, 0.5, 0.2, 0.05)

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
        st.sidebar.success("Custom Signal Loaded!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}. Defaulting to benchmark signal.")
        true_clean = load_default_signal(length)
else:
    true_clean = load_default_signal(length)

# Add synthetic noise
high_freq_noise = noise_amp * np.sin(hum_freq * t)
baseline_drift = drift_level * np.sin(0.2 * t)
random_noise = np.random.normal(0, noise_amp * 0.5, length)

raw_noisy = true_clean + high_freq_noise + baseline_drift + random_noise
raw_noisy = (raw_noisy - np.min(raw_noisy)) / (np.max(raw_noisy) - np.min(raw_noisy) + 1e-8)

# Stage 2: Bandpass Filter
filtered_signal = butter_bandpass_filter(raw_noisy, lowcut=0.5, highcut=4.0, fs=FS)
filtered_signal = (filtered_signal - np.min(filtered_signal)) / (np.max(filtered_signal) - np.min(filtered_signal) + 1e-8)

# Signal Metrics
signal_power = np.mean(true_clean ** 2)
noise_power = np.mean((raw_noisy - true_clean) ** 2)
snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 100.0

# AI Inference
model.eval()
with torch.no_grad():
    input_sample = torch.FloatTensor(raw_noisy).unsqueeze(0).unsqueeze(0)
    reconstructed = model(input_sample).squeeze().numpy()

# Reconstruction Correlation Score
correlation_matrix = np.corrcoef(reconstructed, true_clean)
raw_corr = correlation_matrix[0, 1]
confidence_score = 0.0 if np.isnan(raw_corr) else float(raw_corr) * 100.0

# Peak Detection
peaks, _ = signal.find_peaks(reconstructed, distance=int(FS * 0.4), prominence=0.15)

# Output Signal Quality Metrics
denoised_noise_power = np.mean((reconstructed - true_clean) ** 2)
output_snr_db = 10 * np.log10(signal_power / denoised_noise_power) if denoised_noise_power > 0 else 100.0
snr_improvement = output_snr_db - snr_db

# --- 5. DASHBOARD DISPLAY ---
tabs = st.tabs(["Single Signal Analysis", "Batch Performance Suite"])

with tabs[0]:
    if is_custom_file:
        st.info(" **Source:** Processing Custom Uploaded Signal Stream")
    else:
        st.info(" **Source:** Using Built-in Benchmark Synthetic Reference Wave")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Input SNR", f"{snr_db:.2f} dB")
    col2.metric("AI Quality Score", f"{confidence_score:.1f}%")
    col3.metric("SNR Gain", f"+{snr_improvement:.2f} dB")
    col4.metric("Detected Peaks", len(peaks))

    st.markdown("---")

    # RHYTHM & SIGNAL QUALITY ASSESSMENT
    peak_intervals = np.diff(peaks)
    interval_variance = np.var(peak_intervals) if len(peak_intervals) > 0 else 0

    if confidence_score < 70.0:
        st.error(" **Conclusion: UNRELIABLE DATA** — Environmental noise too severe for diagnostic assessment.")
    elif interval_variance > 2.0:
        st.warning(" **Conclusion: ALERT** — High Heart Rate Variability / Potential Arrhythmia detected.")
    else:
        st.success(" **Conclusion: HEALTHY** — Stable sinus rhythm detected.")


    fig, axs = plt.subplots(4, 1, figsize=(12, 10))

    axs[0].plot(raw_noisy, color='crimson', label='Raw Noisy Input')
    axs[0].set_title("Stage 1: Raw Telemetry Stream")
    axs[0].legend(loc="upper right")

    axs[1].plot(filtered_signal, color='darkorange', label='SciPy Bandpass Filtered (0.5 - 4.0 Hz)')
    axs[1].set_title("Stage 2: Classical Bandpass Filter")
    axs[1].legend(loc="upper right")

    axs[2].plot(reconstructed, color='royalblue', linewidth=2, label='AI Reconstructed Signal')
    if len(peaks) > 0:
        axs[2].scatter(peaks, reconstructed[peaks], color='darkmagenta', s=80, zorder=5, label=f'Detected Peaks ({len(peaks)})')
    axs[2].set_title(f"Stage 3: AI Reconstructed Wave (Quality Score: {confidence_score:.1f}%)")
    axs[2].legend(loc="upper right")

    axs[3].plot(true_clean, color='forestgreen', linestyle='--', label='Ground Truth Reference')
    axs[3].set_title("Stage 4: Ground Truth Reference Signal")
    axs[3].legend(loc="upper right")

    plt.tight_layout()
    st.pyplot(fig)

with tabs[1]:
    st.subheader("Pipeline Performance Across Multiple Iterations")
    st.markdown("Runs 50 automated tests under current noise configurations.")
    
    if st.button("Run Batch Evaluation"):
        eval_confidences = []
        peaks_found = []
        
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
            peaks_found.append(len(pks))

        eval_col1, eval_col2 = st.columns(2)
        eval_col1.metric("Mean Quality Score", f"{np.mean(eval_confidences):.1f}%")
        eval_col2.metric("Mean Peaks Counted", f"{np.mean(peaks_found):.1f}")
        
        fig_eval, ax_eval = plt.subplots(figsize=(10, 3))
        ax_eval.hist(eval_confidences, bins=15, color='royalblue', edgecolor='black')
        ax_eval.set_title("Distribution of Signal Quality Scores (%)")
        ax_eval.set_xlabel("Reconstruction Accuracy (%)")
        ax_eval.set_ylabel("Sample Count")
        st.pyplot(fig_eval)