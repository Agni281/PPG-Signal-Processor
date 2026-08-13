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

# --- 1. MODEL DEFINITION (1D U-NET WITH SKIP CONNECTIONS) ---
class SignalDenoisingUNet1D(nn.Module):
    def __init__(self):
        super(SignalDenoisingUNet1D, self).__init__()
        
        # Encoder Path
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        
        # Decoder Path with Skip Connections
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)  # Shape: (batch, 16, 100)
        e2 = self.enc2(e1) # Shape: (batch, 32, 50)
        
        d2 = self.dec2(e2) # Shape: (batch, 16, 100)
        cat1 = torch.cat((d2, e1), dim=1) # Skip connection from e1
        
        out = self.dec1(cat1) # Shape: (batch, 1, 200)
        return out

# --- 2. DATA LOADERS & FILTERING ---
def clinical_triage_engine(reconstructed, peaks, fs=50.0):
    """
    Evaluates physiological biomarkers from PPG telemetry and generates 
    actionable, non-diagnostic clinical advice and risk-stratified guidance.
    """
    # 1. Fallback: Signal Quality Check
    if len(peaks) < 2:
        return {
            "status": "CRITICAL / UNREADABLE DATA",
            "color": "error",
            "bpm": 0,
            "sdnn": 0.0,
            "category": "Artifact",
            "advice": [
                " **Insufficient Peak Resolution:** Waveform amplitude is suppressed or distorted by extreme noise.",
                " **Troubleshooting:** Check sensor contact, ensure the LED diode is flush against the skin, and ask the patient to minimize limb movement.",
                " **Next Step:** Re-run baseline measurement after 30 seconds of rest."
            ]
        }
    
    # 2. Derive Physiological Biomarkers
    rr_intervals_sec = np.diff(peaks) / fs
    rr_intervals_ms = rr_intervals_sec * 1000.0
    
    # Estimated Heart Rate (BPM)
    mean_rr_sec = np.mean(rr_intervals_sec)
    bpm = int(60.0 / mean_rr_sec) if mean_rr_sec > 0 else 0
    
    # Standard Deviation of NN intervals (SDNN) - HRV Metric
    sdnn = np.std(rr_intervals_ms)
    
    # Amplitude Variability (Perfusion Index Proxy)
    peak_amps = reconstructed[peaks]
    amp_cv = (np.std(peak_amps) / np.mean(peak_amps)) * 100.0  # Coefficient of Variation (%)
    
    # 3. Actionable Clinical Matrix
    advice_notes = []
    
    # --- Category A: Rate Anomalies (Bradycardia / Tachycardia) ---
    if bpm < 50:
        status = "ALERT — SEVERE BRADYCARDIA"
        color = "warning"
        category = "Bradycardia"
        advice_notes.append(f" **Low Heart Rate ({bpm} BPM):** Heart rate falls below normal resting threshold (60 BPM).")
        advice_notes.append(" **Clinical Action:** Assess patient for symptoms of dizziness, fatigue, or syncope. Check current medication log for beta-blockers or calcium channel blockers.")
        advice_notes.append(" **Patient Guidance:** Avoid sudden standing to prevent orthostatic lightheadedness. Seek immediate care if chest tightness occurs.")
        
    elif bpm > 110:
        status = "ALERT — TACHYCARDIA DETECTED"
        color = "warning"
        category = "Tachycardia"
        advice_notes.append(f" **Elevated Heart Rate ({bpm} BPM):** Resting pulse exceeds normal physiological limits.")
        advice_notes.append(" **Clinical Action:** Screen for underlying drivers including fever, dehydration, acute stress, pain, or excessive stimulant intake.")
        advice_notes.append(" **Patient Guidance:** Sit comfortably, hydrate with fluids, and perform guided deep-breathing exercises. Re-measure in 15 minutes.")

    # --- Category B: Rhythm Anomalies (Arrhythmia / HRV Fluctuations) ---
    elif sdnn > 120.0:
        status = "ALERT — HIGH RHYTHM VARIABILITY"
        color = "warning"
        category = "Arrhythmia Risk"
        advice_notes.append(f" **Irregular Pulse Cadence (SDNN: {sdnn:.1f} ms):** High beat-to-beat timing variance detected.")
        advice_notes.append(" **Clinical Action:** Schedule a full 12-lead ECG trace to evaluate for Atrial Fibrillation (AFib), Premature Ventricular Contractions (PVCs), or sinus arrhythmia.")
        advice_notes.append(" **Patient Guidance:** Limit caffeine and nicotine intake. Note any sensations of heart fluttering or skipped beats in a symptom journal.")

    elif sdnn < 15.0 and len(peaks) >= 4:
        status = "ALERT — LOW HRV (AUTONOMIC STRESS)"
        color = "warning"
        category = "Autonomic Depression"
        advice_notes.append(f" **Suppressed HRV (SDNN: {sdnn:.1f} ms):** Abnormally rigid beat-to-beat timing.")
        advice_notes.append(" **Clinical Action:** Indicates high sympathetic overdrive or fatigue. Evaluate autonomic nervous system stress, metabolic health, or sleep apnea risk.")
        advice_notes.append(" **Patient Guidance:** Focus on recovery, prioritize sleep hygiene, and avoid intense physical exertion until baseline HRV recovers.")

    # --- Category C: Normal Rhythm ---
    else:
        status = "NOMINAL — STABLE SINUS RHYTHM"
        color = "success"
        category = "Normal"
        advice_notes.append(f" **Normal Physiological Parameters:** Heart rate ({bpm} BPM) and pulse cadence (SDNN: {sdnn:.1f} ms) are within target ranges.")
        advice_notes.append(" **Clinical Action:** Telemetry signal is clean. Continue standard passive continuous monitoring protocol.")
        advice_notes.append(" **Patient Guidance:** No immediate health interventions required. Maintain regular daily routines.")

    # --- Category D: Signal Perfusion Warning ---
    if amp_cv > 15.0:
        advice_notes.append(f" **Peripheral Perfusion Instability (Amp CV: {amp_cv:.1f}%):** Significant pulse wave height variation.")
        advice_notes.append(" **Clinical Action:** May indicate peripheral vasoconstriction, cold extremities, or mild sensor displacement.")

    return {
        "status": status,
        "color": color,
        "bpm": bpm,
        "sdnn": sdnn,
        "category": category,
        "advice": advice_notes
    }

def butter_bandpass_filter(data, lowcut=0.5, highcut=4.0, fs=50.0, order=2):
    """Bandpass filter to eliminate baseline wander and high-frequency noise."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

@st.cache_data
def load_default_signal(length=200):
    """Generates a benchmark PPG wave."""
    t = np.linspace(0, 4 * np.pi, length)
    clean_wave = np.sin(t) + 0.5 * np.sin(2 * t) + 0.2 * np.sin(3 * t)
    return (clean_wave - np.min(clean_wave)) / (np.max(clean_wave) - np.min(clean_wave) + 1e-8)

@st.cache_resource
def train_model():
    """Trains the 1D U-Net PyTorch model once and caches it."""
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

# --- 3. SIDEBAR CONTROLS & FILE UPLOADER ---
st.sidebar.header("Signal Noise Parameters")
st.sidebar.markdown("Simulate environmental and device interference:")

uploaded_file = st.sidebar.file_uploader(
    "Upload Custom Biosensor Signal (CSV/TXT)", 
    type=["csv", "txt"],
    help="Upload a 1D CSV/TXT file with raw sensor telemetry."
)

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
        st.sidebar.success("Custom Signal Loaded!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}. Defaulting to PPG reference signal.")
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

# AI U-Net Inference
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
        st.info(" **Source:** Using Built-in Benchmark PPG Signal")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Input SNR", f"{snr_db:.2f} dB")
    col2.metric("AI Quality Score", f"{confidence_score:.1f}%")
    col3.metric("SNR Gain", f"+{snr_improvement:.2f} dB")
    col4.metric("Detected Peaks", len(peaks))

    st.markdown("---")

    # RHYTHM ASSESSMENT
    peak_intervals = np.diff(peaks)
    interval_variance = np.var(peak_intervals) if len(peak_intervals) > 0 else 0

   # Run Deep Clinical Triage Engine
    triage = clinical_triage_engine(reconstructed, peaks, fs=FS)

    # Display Triage Banner
    if triage["color"] == "error":
        st.error(f"**Clinical Status:** {triage['status']}")
    elif triage["color"] == "warning":
        st.warning(f"**Clinical Status:** {triage['status']}")
    else:
        st.success(f"**Clinical Status:** {triage['status']}")

    # Clinical Biomarkers & Guidance Box
    with st.expander("🩺 **Clinical Telemetry Guidance & Biomarkers**", expanded=True):
        bio_col1, bio_col2 = st.columns(2)
        bio_col1.metric("Est. HRV (SDNN)", f"{triage.get('sdnn', 0.0):.1f} ms", help="Standard Deviation of Normal-to-Normal Peak Intervals")
        bio_col2.metric("Morphology Stability", "High" if triage.get('sdnn', 0) < 50 else "Moderate/Low")
        
        st.markdown("##### **Actionable Clinical Insights:**")
        for note in triage["advice"]:
            st.markdown(f"- {note}")

    fig, axs = plt.subplots(4, 1, figsize=(12, 10))

    axs[0].plot(raw_noisy, color='crimson', label='Raw Noisy Signal')
    axs[0].set_title("Stage 1: Raw Telemetry Stream (Rural Clinic)")
    axs[0].legend(loc="upper right")

    axs[1].plot(filtered_signal, color='darkorange', label='SciPy Bandpass Filtered (0.5 - 4.0 Hz)')
    axs[1].set_title("Stage 2: Classical Bandpass Filter")
    axs[1].legend(loc="upper right")

    axs[2].plot(reconstructed, color='royalblue', linewidth=2, label='AI Denoised Signal (1D U-Net)')
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