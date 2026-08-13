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

# --- 1. MODEL DEFINITION (CORRECTED CHANNEL PROGRESSION) ---
class SignalDenoisingUNet1D(nn.Module):
    def __init__(self):
        super(SignalDenoisingUNet1D, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose1d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)       # (B, 16, L/2)
        e2 = self.enc2(e1)      # (B, 32, L/4)
        d2 = self.dec2(e2)      # (B, 16, L/2)
        cat1 = torch.cat((d2, e1), dim=1) # (B, 32, L/2)
        return self.dec1(cat1)  # (B, 1, L)

# --- 2. SIGNAL HELPERS & METRICS ---
def generate_synthetic_ppg(t, hr_bpm):
    """Generates a dynamic cardiac wave for given BPM."""
    freq = hr_bpm / 60.0
    ppg = np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(4 * np.pi * freq * t + 0.5)
    return (ppg - np.min(ppg)) / (np.max(ppg) - np.min(ppg) + 1e-8)

def calculate_sqi(sig, fs=50.0):
    """Calculates Signal Quality Index using spectral energy in cardiac band (0.5 - 4 Hz)."""
    freqs, psd = signal.welch(sig, fs=fs, nperseg=len(sig))
    cardiac_band_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 4.0)])
    total_power = np.sum(psd) + 1e-8
    sqi = (cardiac_band_power / total_power) * 100.0
    return np.clip(sqi, 0.0, 100.0)

def clinical_triage_engine(reconstructed, peaks, fs=50.0):
    if len(peaks) < 2:
        return {
            "status": "CRITICAL / UNREADABLE DATA",
            "color": "error",
            "bpm": 0,
            "sdnn": 0.0,
            "category": "Artifact",
            "advice": ["⚠️ **Insufficient Peak Resolution:** Waveform amplitude is suppressed or distorted."]
        }
    
    rr_intervals_sec = np.diff(peaks) / fs
    rr_intervals_ms = rr_intervals_sec * 1000.0
    
    mean_rr_sec = np.mean(rr_intervals_sec)
    bpm = int(60.0 / mean_rr_sec) if mean_rr_sec > 0 else 0
    sdnn = np.std(rr_intervals_ms)
    
    advice_notes = []
    if bpm < 50:
        status, color, category = "ALERT — SEVERE BRADYCARDIA", "warning", "Bradycardia"
        advice_notes.append(f"📉 **Low Heart Rate ({bpm} BPM):** Pulse rate below normal resting threshold.")
    elif bpm > 110:
        status, color, category = "ALERT — TACHYCARDIA DETECTED", "warning", "Tachycardia"
        advice_notes.append(f"📈 **Elevated Heart Rate ({bpm} BPM):** Pulse rate exceeds physiological target.")
    elif sdnn > 120.0:
        status, color, category = "ALERT — HIGH RHYTHM VARIABILITY", "warning", "Arrhythmia Risk"
        advice_notes.append(f"⚡ **Irregular Pulse Cadence (SDNN: {sdnn:.1f} ms):** High beat-to-beat timing variance.")
    else:
        status, color, category = "NOMINAL — STABLE SINUS RHYTHM", "success", "Normal"
        advice_notes.append(f"✅ **Normal Parameters:** Heart rate ({bpm} BPM) and pulse cadence (SDNN: {sdnn:.1f} ms) are stable.")

    return {"status": status, "color": color, "bpm": bpm, "sdnn": sdnn, "category": category, "advice": advice_notes}

def butter_bandpass_filter(data, lowcut=0.5, highcut=4.0, fs=50.0, order=2):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

@st.cache_resource
def train_model():
    np.random.seed(42)
    torch.manual_seed(42)
    length = 200
    FS = 50.0
    duration = 4.0
    t = np.linspace(0, duration, length)
    
    clean_dataset, noisy_dataset = [], []
    for _ in range(500):
        rand_hr = np.random.randint(50, 130)
        clean_wave = generate_synthetic_ppg(t, rand_hr)
        noisy_wave = clean_wave + 0.3 * np.sin(2 * np.pi * 50 * t) + 0.2 * np.sin(2 * np.pi * 0.2 * t) + np.random.normal(0, 0.15, length)
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

# --- 3. RESTORED SIDEBAR CONFIGURATION & FILE UPLOADER ---
st.sidebar.header("Data & Noise Settings")

data_source = st.sidebar.radio("Data Mode", ["Synthetic Generator", "Upload Real PPG Stream"])

if data_source == "Upload Real PPG Stream":
    uploaded_file = st.sidebar.file_uploader("Upload CSV/TXT Signal", type=["csv", "txt"])
    FS = st.sidebar.number_input("Sensor Sampling Rate (Hz)", min_value=1.0, max_value=1000.0, value=50.0, step=1.0)
    target_hr = 75 # Default fallback
else:
    uploaded_file = None
    FS = 50.0
    target_hr = st.sidebar.slider("Simulated Heart Rate (BPM)", 50, 140, 75)

noise_amp = st.sidebar.slider("Noise Amplitude", 0.0, 0.8, 0.3, 0.05)
hum_freq = st.sidebar.slider("Grid Hum Frequency (Hz)", 10, 100, 50, 10)
drift_level = st.sidebar.slider("Baseline Drift", 0.0, 0.5, 0.2, 0.05)

# --- 4. DATA PROCESSING PIPELINE ---
TARGET_LENGTH = 200
duration = 4.0
t = np.linspace(0, duration, TARGET_LENGTH)
is_custom_file = False

if data_source == "Upload Real PPG Stream" and uploaded_file is not None:
    try:
        raw_data = np.loadtxt(uploaded_file, delimiter=',')
        if raw_data.ndim > 1:
            raw_data = raw_data[:, 0]
        
        # Resample any uploaded file length to 200 points for U-Net compatibility
        if len(raw_data) != TARGET_LENGTH:
            raw_data = signal.resample(raw_data, TARGET_LENGTH)
            
        true_clean = (raw_data - np.min(raw_data)) / (np.max(raw_data) - np.min(raw_data) + 1e-8)
        is_custom_file = True
        st.sidebar.success("Custom PPG Stream Processed!")
    except Exception as e:
        st.sidebar.error(f"Error parsing file: {e}. Reverting to Synthetic.")
        true_clean = generate_synthetic_ppg(t, target_hr)
else:
    true_clean = generate_synthetic_ppg(t, target_hr)

# Corrupt signal with noise controls
high_freq_noise = noise_amp * np.sin(2 * np.pi * hum_freq * t)
baseline_drift = drift_level * np.sin(2 * np.pi * 0.2 * t)
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

# Peak Detection & Metrics
min_distance = max(1, int(FS * 0.35))
peaks, _ = signal.find_peaks(reconstructed, distance=min_distance, prominence=0.15)
confidence_score = calculate_sqi(reconstructed, fs=FS)

# --- 5. DASHBOARD TABS ---
tabs = st.tabs(["Single Signal Analysis", "Batch Performance Suite"])
triage = clinical_triage_engine(reconstructed, peaks, fs=FS)

with tabs[0]:
    if is_custom_file:
        st.info("ℹ️ **Source:** Processing Custom Uploaded Signal Stream")
    else:
        st.info("ℹ️ **Source:** Using Built-in Dynamic PPG Generator")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Signal Quality (SQI)", f"{confidence_score:.1f}%")
    col2.metric("Est. Heart Rate", f"{triage['bpm']} BPM" if triage['bpm'] > 0 else "N/A")
    col3.metric("HRV (SDNN)", f"{triage['sdnn']:.1f} ms" if triage['sdnn'] > 0 else "N/A")
    col4.metric("Peaks Counted", len(peaks))

    st.markdown("---")

    # Display Triage Banner
    if triage["color"] == "error":
        st.error(f"**Clinical Status:** {triage['status']}")
    elif triage["color"] == "warning":
        st.warning(f"**Clinical Status:** {triage['status']}")
    else:
        st.success(f"**Clinical Status:** {triage['status']}")

    # Clinical Guidance Box
    with st.expander("🩺 **Clinical Telemetry Guidance & Biomarkers**", expanded=True):
        st.markdown("##### **Actionable Insights:**")
        for note in triage["advice"]:
            st.markdown(f"- {note}")

    # Interactive Subplots
    fig = make_subplots(rows=4, cols=1, subplot_titles=(
        "Stage 1: Raw Telemetry Stream (With Noise)",
        "Stage 2: Classical Bandpass Filter (0.5 - 4.0 Hz)",
        f"Stage 3: AI Reconstructed Wave & Peaks (SQI: {confidence_score:.1f}%)",
        "Stage 4: Ground Truth Reference Signal"
    ))

    fig.add_trace(go.Scatter(x=t, y=raw_noisy, mode='lines', name='Raw Noisy', line=dict(color='crimson')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=filtered_signal, mode='lines', name='Bandpass Filtered', line=dict(color='darkorange')), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=reconstructed, mode='lines', name='AI Reconstructed', line=dict(color='royalblue', width=2)), row=3, col=1)
    
    if len(peaks) > 0:
        fig.add_trace(go.Scatter(x=t[peaks], y=reconstructed[peaks], mode='markers', name='Systolic Peaks',
                                 marker=dict(color='purple', size=8, symbol='diamond')), row=3, col=1)

    fig.add_trace(go.Scatter(x=t, y=true_clean, mode='lines', name='Ground Truth', line=dict(color='forestgreen', dash='dash')), row=4, col=1)

    fig.update_layout(height=800, template="plotly_white", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# --- RESTORED BATCH EVALUATION SUITE WITH MEAN CALCULATOR ---
with tabs[1]:
    st.subheader("Pipeline Performance Across Multiple Iterations")
    st.markdown("Runs **50 automated test iterations** under current noise configurations to evaluate stability.")
    
    if st.button("Run Batch Evaluation"):
        eval_confidences = []
        peaks_found = []
        eval_bpms = []
        
        for _ in range(50):
            # Introduce phase shift / noise variation per iteration
            shift = np.random.randint(-15, 15)
            sample_clean = np.roll(true_clean, shift)
            sample_noise = noise_amp * np.sin(2 * np.pi * hum_freq * t) + drift_level * np.sin(2 * np.pi * 0.2 * t) + np.random.normal(0, noise_amp * 0.5, TARGET_LENGTH)
            sample_noisy = (sample_clean + sample_noise)
            sample_noisy = (sample_noisy - np.min(sample_noisy)) / (np.max(sample_noisy) - np.min(sample_noisy) + 1e-8)
            
            inp = torch.FloatTensor(sample_noisy).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                rec = model(inp).squeeze().numpy()
            
            sqi = calculate_sqi(rec, fs=FS)
            eval_confidences.append(sqi)
            
            pks, _ = signal.find_peaks(rec, distance=min_distance, prominence=0.15)
            peaks_found.append(len(pks))
            
            if len(pks) >= 2:
                eval_bpms.append(60.0 / np.mean(np.diff(t[pks])))

        # Batch Mean Summary Metrics
        eval_col1, eval_col2, eval_col3 = st.columns(3)
        eval_col1.metric("Mean Quality Score (SQI)", f"{np.mean(eval_confidences):.1f}%")
        eval_col2.metric("Mean Peaks Counted", f"{np.mean(peaks_found):.1f}")
        eval_col3.metric("Mean Estimated BPM", f"{np.mean(eval_bpms):.1f} BPM" if len(eval_bpms) > 0 else "N/A")
        
        # Plot distribution histogram using Plotly
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(x=eval_confidences, nbinsx=15, marker_color='royalblue', opacity=0.75))
        hist_fig.update_layout(
            title="Distribution of Signal Quality Scores (SQI % across 50 runs)",
            xaxis_title="Quality Score (%)",
            yaxis_title="Sample Count",
            template="plotly_white",
            height=350
        )
        st.plotly_chart(hist_fig, use_container_width=True)