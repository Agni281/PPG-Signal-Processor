import streamlit as st
import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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
        cat1 = torch.cat((d2, e1), dim=1)
        return self.dec1(cat1)

# --- 2. TRIAGE & FILTERS ---
def clinical_triage_engine(reconstructed, peaks, fs=50.0):
    if len(peaks) < 2:
        return {
            "status": "CRITICAL / UNREADABLE DATA",
            "color": "error",
            "bpm": 0,
            "sdnn": 0.0,
            "category": "Artifact",
            "advice": [" **Insufficient Peak Resolution:** Waveform amplitude is suppressed or distorted."]
        }
    
    rr_intervals_sec = np.diff(peaks) / fs
    rr_intervals_ms = rr_intervals_sec * 1000.0
    
    mean_rr_sec = np.mean(rr_intervals_sec)
    bpm = int(60.0 / mean_rr_sec) if mean_rr_sec > 0 else 0
    sdnn = np.std(rr_intervals_ms)
    
    peak_amps = reconstructed[peaks]
    amp_cv = (np.std(peak_amps) / (np.mean(peak_amps) + 1e-8)) * 100.0
    
    advice_notes = []
    if bpm < 50:
        status, color, category = "ALERT — SEVERE BRADYCARDIA", "warning", "Bradycardia"
        advice_notes.append(f" **Low Heart Rate ({bpm} BPM):** Pulse rate below normal resting threshold.")
    elif bpm > 110:
        status, color, category = "ALERT — TACHYCARDIA DETECTED", "warning", "Tachycardia"
        advice_notes.append(f" **Elevated Heart Rate ({bpm} BPM):** Pulse rate exceeds physiological target.")
    elif sdnn > 120.0:
        status, color, category = "ALERT — HIGH RHYTHM VARIABILITY", "warning", "Arrhythmia Risk"
        advice_notes.append(f" **Irregular Pulse Cadence (SDNN: {sdnn:.1f} ms):** Beat-to-beat timing variance.")
    else:
        status, color, category = "NOMINAL — STABLE SINUS RHYTHM", "success", "Normal"
        advice_notes.append(f" **Normal Parameters:** Heart rate ({bpm} BPM) and pulse cadence (SDNN: {sdnn:.1f} ms) stable.")

    return {"status": status, "color": color, "bpm": bpm, "sdnn": sdnn, "category": category, "advice": advice_notes}

def butter_bandpass_filter(data, lowcut=0.5, highcut=4.0, fs=50.0, order=2):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

@st.cache_data
def load_default_signal(length=200):
    # Updated: 10*pi produces ~5 heartbeats over 200 samples at 50Hz (= 75 BPM)
    t = np.linspace(0, 10 * np.pi, length)
    clean_wave = np.sin(t) + 0.5 * np.sin(2 * t) + 0.2 * np.sin(3 * t)
    return (clean_wave - np.min(clean_wave)) / (np.max(clean_wave) - np.min(clean_wave) + 1e-8)

@st.cache_resource
def train_model():
    np.random.seed(42)
    torch.manual_seed(42)
    length = 200
    t = np.linspace(0, 10 * np.pi, length)
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
    
    model = SignalDenoisingUNet1D()
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

# --- 3. SIDEBAR CONFIGURATION ---
st.sidebar.header("Data & Noise Settings")

data_source = st.sidebar.radio("Data Mode", ["Synthetic Generator", "Upload Real PPG Stream"])

if data_source == "Upload Real PPG Stream":
    uploaded_file = st.sidebar.file_uploader("Upload CSV/TXT Signal", type=["csv", "txt"])
    FS = st.sidebar.number_input("Sensor Sampling Rate (Hz)", min_value=1.0, max_value=1000.0, value=50.0, step=1.0)
else:
    uploaded_file = None
    FS = 50.0

noise_amp = st.sidebar.slider("Noise Amplitude", 0.0, 0.8, 0.3, 0.05)
hum_freq = st.sidebar.slider("Grid Hum Frequency (Hz)", 10, 100, 50, 10)
drift_level = st.sidebar.slider("Baseline Drift", 0.0, 0.5, 0.2, 0.05)

# --- 4. PROCESSING PIPELINE ---
TARGET_LENGTH = 200
t = np.linspace(0, 10 * np.pi, TARGET_LENGTH)
is_custom_file = False

if data_source == "Upload Real PPG Stream" and uploaded_file is not None:
    try:
        raw_data = np.loadtxt(uploaded_file, delimiter=',')
        if raw_data.ndim > 1:
            raw_data = raw_data[:, 0]
        
        # Resample any uploaded length to exactly 200 points for the 1D U-Net model
        if len(raw_data) != TARGET_LENGTH:
            raw_data = signal.resample(raw_data, TARGET_LENGTH)
            
        true_clean = (raw_data - np.min(raw_data)) / (np.max(raw_data) - np.min(raw_data) + 1e-8)
        is_custom_file = True
        st.sidebar.success("Custom PPG Stream Processed!")
    except Exception as e:
        st.sidebar.error(f"Error parsing file: {e}. Reverting to Synthetic.")
        true_clean = load_default_signal(TARGET_LENGTH)
else:
    true_clean = load_default_signal(TARGET_LENGTH)

# Apply noise (optional testing overlay on both synthetic or real signals)
high_freq_noise = noise_amp * np.sin(hum_freq * t)
baseline_drift = drift_level * np.sin(0.2 * t)
random_noise = np.random.normal(0, noise_amp * 0.5, TARGET_LENGTH)

raw_noisy = true_clean + high_freq_noise + baseline_drift + random_noise
raw_noisy = (raw_noisy - np.min(raw_noisy)) / (np.max(raw_noisy) - np.min(raw_noisy) + 1e-8)

# Bandpass filtering
filtered_signal = butter_bandpass_filter(raw_noisy, lowcut=0.5, highcut=4.0, fs=FS)
filtered_signal = (filtered_signal - np.min(filtered_signal)) / (np.max(filtered_signal) - np.min(filtered_signal) + 1e-8)

# Inference
model.eval()
with torch.no_grad():
    input_sample = torch.FloatTensor(raw_noisy).unsqueeze(0).unsqueeze(0)
    reconstructed = model(input_sample).squeeze().numpy()

# Peak Detection scaled according to target sampling frequency
min_distance = max(1, int(FS * 0.35)) # ~170 BPM upper bound threshold
peaks, _ = signal.find_peaks(reconstructed, distance=min_distance, prominence=0.15)

# Metrics calculation
signal_power = np.mean(true_clean ** 2)
noise_power = np.mean((raw_noisy - true_clean) ** 2)
snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 100.0

correlation_matrix = np.corrcoef(reconstructed, true_clean)
confidence_score = 0.0 if np.isnan(correlation_matrix[0, 1]) else float(correlation_matrix[0, 1]) * 100.0

# --- 5. UI DISPLAY ---
triage = clinical_triage_engine(reconstructed, peaks, fs=FS)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Input SNR", f"{snr_db:.2f} dB")
col2.metric("AI Confidence", f"{confidence_score:.1f}%")
col3.metric("Est. Heart Rate", f"{triage['bpm']} BPM")
col4.metric("Peaks Counted", len(peaks))

if triage["color"] == "error":
    st.error(f"**Status:** {triage['status']}")
elif triage["color"] == "warning":
    st.warning(f"**Status:** {triage['status']}")
else:
    st.success(f"**Status:** {triage['status']}")

fig, axs = plt.subplots(3, 1, figsize=(10, 6))
axs[0].plot(raw_noisy, color='crimson', label='Noisy Input Telemetry')
axs[0].legend()
axs[1].plot(reconstructed, color='royalblue', label='AI Denoised (1D U-Net)')
if len(peaks) > 0:
    axs[1].scatter(peaks, reconstructed[peaks], color='purple', s=60, label=f'Systolic Peaks ({len(peaks)})')
axs[1].legend()
axs[2].plot(true_clean, color='forestgreen', linestyle='--', label='Baseline Wave')
axs[2].legend()
plt.tight_layout()
st.pyplot(fig)