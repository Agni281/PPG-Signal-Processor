import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, FloatSlider, IntSlider

np.random.seed(42)
torch.manual_seed(42)

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

def clinical_triage_engine(reconstructed, peaks, fs=50.0):
    """
    Evaluates clinical biomarkers, signal quality, and rhythm metrics 
    to provide non-diagnostic clinical advice and risk stratification.
    """
    if len(peaks) < 2:
        return {
            "status": "CRITICAL / UNRELIABLE",
            "color": "error",
            "sqi_score": 0.0,
            "hrv_sdnn": 0.0,
            "advice": [
                "🚨 **Insufficient Peak Detection:** Signal attenuation or extreme artifact detected.",
                "👉 **Action:** Check sensor contact, reposition photodiode, and verify optical alignment."
            ]
        }
    
    # 1. Heart Rate Variability (HRV) - SDNN (Standard Deviation of NN intervals in ms)
    rr_intervals_ms = (np.diff(peaks) / fs) * 1000.0
    sdnn = np.std(rr_intervals_ms)
    
    # 2. Perfusion / Morphological Quality (Perfusion Index Proxy)
    peak_amps = reconstructed[peaks]
    amplitude_variance = np.var(peak_amps)
    
    # 3. Clinical Advice Matrix
    advice_notes = []
    
    # Assess Signal Quality & Artifacts
    if amplitude_variance > 0.02:
        advice_notes.append("⚠️ **Peripheral Perfusion Instability:** High beat-to-beat amplitude variation detected. May indicate peripheral vasoconstriction or motion interference.")
    
    # Assess Rhythm & HRV
    if sdnn > 100.0:
        status = "ALERT — HIGH HRV / POSSIBLE ARRHYTHMIA"
        color = "warning"
        advice_notes.append("⚡ **Elevated RR Interval Variance (SDNN > 100ms):** Significant beat-to-beat inconsistency observed.")
        advice_notes.append("👉 **Clinical Recommendation:** Perform 12-lead ECG trace to rule out Atrial Fibrillation or Frequent PVCs.")
    elif sdnn < 10.0 and len(peaks) > 3:
        status = "ALERT — MONOTONOUS RHYTHM (LOW HRV)"
        color = "warning"
        advice_notes.append("📉 **Depressed Heart Rate Variability (SDNN < 10ms):** Reduced autonomic regulation observed.")
        advice_notes.append("👉 **Clinical Recommendation:** Evaluate patient for sympathetic overactivity, severe physical stress, or autonomic neuropathy.")
    else:
        status = "NOMINAL — STABLE SINUS PATTERN"
        color = "success"
        advice_notes.append("✅ **Normal Pulse Interval Dynamics:** Beat-to-beat variability falls within standard physiological boundaries.")
        advice_notes.append("👉 **Clinical Recommendation:** Continue passive remote monitoring. No immediate sensor repositioning required.")
        
    return {
        "status": status,
        "color": color,
        "sdnn": sdnn,
        "advice": advice_notes
    }

def butter_bandpass_filter(data, lowcut=0.5, highcut=4.0, fs=50.0, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def generate_base_dataset(num_samples=500, length=200):
    t = np.linspace(0, 4 * np.pi, length)
    clean_dataset, noisy_dataset = [], []
    for _ in range(num_samples):
        clean_wave = np.sin(t) + 0.5 * np.sin(2 * t) + 0.2 * np.sin(3 * t)
        clean_wave = (clean_wave - np.min(clean_wave)) / (np.max(clean_wave) - np.min(clean_wave) + 1e-8)
        
        noisy_wave = clean_wave + 0.3 * np.sin(50 * t) + 0.2 * np.sin(0.2 * t) + np.random.normal(0, 0.15, length)
        noisy_wave = (noisy_wave - np.min(noisy_wave)) / (np.max(noisy_wave) - np.min(noisy_wave) + 1e-8)
        
        clean_dataset.append(clean_wave)
        noisy_dataset.append(noisy_wave)
    return np.array(clean_dataset), np.array(noisy_dataset)

print("Training base model for interactive dashboard...")
clean_data, noisy_data = generate_base_dataset()
clean_tensor = torch.FloatTensor(clean_data).unsqueeze(1)
noisy_tensor = torch.FloatTensor(noisy_data).unsqueeze(1)

model = SignalDenoisingUNet1D()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

epochs = 40
batch_size = 32
for epoch in range(epochs):
    permutation = torch.randperm(noisy_tensor.size(0))
    for i in range(0, noisy_tensor.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        optimizer.zero_grad()
        outputs = model(noisy_tensor[indices])
        loss = criterion(outputs, clean_tensor[indices])
        loss.backward()
        optimizer.step()
print("Model Ready!\n")

def run_interactive_pipeline(noise_level, hum_freq, drift_level):
    FS = 50.0
    length = 200
    t = np.linspace(0, 4 * np.pi, length)
    
    true_clean = np.sin(t) + 0.5 * np.sin(2 * t) + 0.2 * np.sin(3 * t)
    true_clean = (true_clean - np.min(true_clean)) / (np.max(true_clean) - np.min(true_clean) + 1e-8)
    
    high_freq_noise = noise_level * np.sin(hum_freq * t)
    baseline_drift = drift_level * np.sin(0.2 * t)
    random_noise = np.random.normal(0, noise_level * 0.5, length)
    
    raw_noisy = true_clean + high_freq_noise + baseline_drift + random_noise
    raw_noisy = (raw_noisy - np.min(raw_noisy)) / (np.max(raw_noisy) - np.min(raw_noisy) + 1e-8)
    
    filtered_signal = butter_bandpass_filter(raw_noisy, lowcut=0.5, highcut=4.0, fs=FS)
    filtered_signal = (filtered_signal - np.min(filtered_signal)) / (np.max(filtered_signal) - np.min(filtered_signal) + 1e-8)
    
    signal_power = np.mean(true_clean ** 2)
    noise_power = np.mean((raw_noisy - true_clean) ** 2)
    snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 100
    
    model.eval()
    with torch.no_grad():
        input_sample = torch.FloatTensor(raw_noisy).unsqueeze(0).unsqueeze(0)
        reconstructed = model(input_sample).squeeze().numpy()
        
    correlation_matrix = np.corrcoef(reconstructed, true_clean)
    raw_corr = correlation_matrix[0, 1]
    confidence_score = 0.0 if np.isnan(raw_corr) else float(raw_corr) * 100.0
    
    peaks, _ = signal.find_peaks(reconstructed, distance=int(FS * 0.4), prominence=0.15)
    
    denoised_noise_power = np.mean((reconstructed - true_clean) ** 2)
    output_snr_db = 10 * np.log10(signal_power / denoised_noise_power) if denoised_noise_power > 0 else 100.0
    snr_improvement = output_snr_db - snr_db

    print(f"SNR: {snr_db:.2f} dB | Confidence: {confidence_score:.2f}% | SNR Improvement: +{snr_improvement:.2f} dB | Peaks Detected: {len(peaks)}")
    
    # RHYTHM ASSESSMENT
    peak_intervals = np.diff(peaks)
    interval_variance = np.var(peak_intervals) if len(peak_intervals) > 0 else 0

    if confidence_score < 70.0:
        triage_msg = "Conclusion: UNRELIABLE DATA - Noise too high for diagnostic safety."
    elif interval_variance > 0.05:
        triage_msg = "Conclusion: ALERT - Potential Arrhythmia / Irregular Rhythm Detected."
    else:
        triage_msg = "Conclusion: HEALTHY - Normal Sinus Rhythm."

    print(triage_msg)
    
    plt.figure(figsize=(12, 9))
    
    plt.subplot(4, 1, 1)
    plt.plot(raw_noisy, color='crimson', label='Raw Sensor Stream')
    plt.title(f'Raw Input Signal (Grid Hum: {hum_freq}Hz | Drift: {drift_level:.2f})')
    plt.legend(loc='upper right')
    
    plt.subplot(4, 1, 2)
    plt.plot(filtered_signal, color='darkorange', label='SciPy Bandpass Filtered')
    plt.title('Stage 2: Classical Bandpass Filter (0.5 - 4.0 Hz)')
    plt.legend(loc='upper right')
    
    plt.subplot(4, 1, 3)
    plt.plot(reconstructed, color='royalblue', linewidth=2, label='AI Denoised Signal (1D U-Net)')
    if len(peaks) > 0:
        plt.scatter(peaks, reconstructed[peaks], color='darkmagenta', s=90, zorder=5, label=f'Systolic Peaks ({len(peaks)})')
    plt.title(f'AI Reconstructed Signal (Confidence: {confidence_score:.1f}%)')
    plt.legend(loc='upper right')
    
    plt.subplot(4, 1, 4)
    plt.plot(true_clean, color='forestgreen', linestyle='--', label='Target Reference Wave')
    plt.title('Ground Truth Baseline')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

print("--- PPG SIGNAL DASHBOARD ---")
interact(
    run_interactive_pipeline,
    noise_level=FloatSlider(min=0.0, max=0.8, step=0.05, value=0.3, description='Noise Amp'),
    hum_freq=IntSlider(min=10, max=100, step=10, value=50, description='Grid Hum (Hz)'),
    drift_level=FloatSlider(min=0.0, max=0.5, step=0.05, value=0.2, description='Baseline Drift')
);