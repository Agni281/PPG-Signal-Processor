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

@st.cache_resource
def train_model():
    """Trains the base PyTorch autoencoder model once and caches it."""
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

st.sidebar.header("⚙️ Signal Noise Parameters")
st.sidebar.markdown("Simulate environmental and device interference:")

noise_amp = st.sidebar.slider("Noise Amplitude", 0.0, 0.8, 0.3, 0.05)
hum_freq = st.sidebar.slider("Grid Hum Frequency (Hz)", 10, 100, 50, 10)
drift_level = st.sidebar.slider("Baseline Drift (Breathing)", 0.0, 0.5, 0.2, 0.05)


length = 200
t = np.linspace(0, 4 * np.pi, length)


true_clean = np.sin(t) + 0.5 * np.sin(2 * t) + 0.2 * np.sin(3 * t)
true_clean = (true_clean - np.min(true_clean)) / (np.max(true_clean) - np.min(true_clean))


high_freq_noise = noise_amp * np.sin(hum_freq * t)
baseline_drift = drift_level * np.sin(0.2 * t)
random_noise = np.random.normal(0, noise_amp * 0.5, length)

raw_noisy = true_clean + high_freq_noise + baseline_drift + random_noise
raw_noisy = (raw_noisy - np.min(raw_noisy)) / (np.max(raw_noisy) - np.min(raw_noisy))


signal_power = np.mean(true_clean ** 2)
noise_power = np.mean((raw_noisy - true_clean) ** 2)
snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 100


model.eval()
with torch.no_grad():
    input_sample = torch.FloatTensor(raw_noisy).unsqueeze(0).unsqueeze(0)
    reconstructed = model(input_sample).squeeze().numpy()


correlation_matrix = np.corrcoef(reconstructed, true_clean)
confidence_score = correlation_matrix[0, 1] * 100


peaks, _ = signal.find_peaks(reconstructed, distance=30, prominence=0.2)
peak_intervals = np.diff(peaks)
interval_variance = np.var(peak_intervals) if len(peak_intervals) > 0 else 0




col1, col2, col3 = st.columns(3)
col1.metric("Input SNR", f"{snr_db:.2f} dB")
col2.metric("AI Confidence", f"{confidence_score:.1f}%")
col3.metric("Detected Peaks", len(peaks))

st.markdown("---")


if confidence_score < 70.0:
    st.error(" **Conclusion: UNRELIABLE DATA** — Environmental noise too severe for diagnostic assessment.")
elif interval_variance > 2.0:
    st.warning(" **Conclusion: ALERT** — High Heart Rate Variability / Potential Arrhythmia detected.")
else:
    st.success(" **Conclusion: HEALTHY** — Stable sinus rhythm detected.")




fig, axs = plt.subplots(3, 1, figsize=(12, 8))

axs[0].plot(raw_noisy, color='crimson', label='Raw Noisy Signal')
axs[0].set_title("Stage 1: Raw Telemetry Stream (Rural Clinic)")
axs[0].legend(loc="upper right")

axs[1].plot(reconstructed, color='royalblue', linewidth=2, label='AI Denoised Signal')
axs[1].scatter(peaks, reconstructed[peaks], color='darkmagenta', s=80, zorder=5, label='Systolic Peaks')
axs[1].set_title(f"Stage 2-4: AI Reconstructed Wave & Feature Extraction (Confidence: {confidence_score:.1f}%)")
axs[1].legend(loc="upper right")

axs[2].plot(true_clean, color='forestgreen', linestyle='--', label='Ground Truth Reference')
axs[2].set_title("Ground Truth Signal Reference")
axs[2].legend(loc="upper right")

plt.tight_layout()
st.pyplot(fig)
