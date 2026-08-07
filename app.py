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
        clean_wave = (clean_wave - np.min(clean_wave)) / (np.max(clean_wave) - np.min(clean_wave))
        
        noisy_wave = clean_wave + 0.3 * np.sin(50 * t) + 0.2 * np.sin(0.2 * t) + np.random.normal(0, 0.15, length)
        noisy_wave = (noisy_wave - np.min(noisy_wave)) / (np.max(noisy_wave) - np.min(noisy_wave))
        
        clean_dataset.append(clean_wave)
        noisy_dataset.append(noisy_wave)
    return np.array(clean_dataset), np.array(noisy_dataset)

print("Training base model for interactive dashboard...")
clean_data, noisy_data = generate_base_dataset()
clean_tensor = torch.FloatTensor(clean_data).unsqueeze(1)
noisy_tensor = torch.FloatTensor(noisy_data).unsqueeze(1)

model = SignalDenoisingAutoencoder()
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
    true_clean = (true_clean - np.min(true_clean)) / (np.max(true_clean) - np.min(true_clean))
    
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
    if len(peaks) > 1:
        peak_intervals_sec = np.diff(peaks) / FS
        heart_rate_bpm = 60.0 / np.mean(peak_intervals_sec)
        interval_variance = np.var(peak_intervals_sec)
    else:
        heart_rate_bpm = 0
        interval_variance = 0.0
    
    print(f"SNR: {snr_db:.2f} dB | AI Confidence: {confidence_score:.2f}% | Peaks Detected: {len(peaks)} | Heart Rate: {heart_rate_bpm:.1f} BPM")
    if confidence_score < 70.0:
        triage_msg = "Conclusion: UNRELIABLE DATA - Noise too high for diagnostic safety."
    elif interval_variance > 0.05 and heart_rate_bpm > 0:
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
    plt.title('Stage 2: Classical Bandpass Filter')
    plt.legend(loc='upper right')
    
    plt.subplot(4, 1, 3)
    plt.plot(reconstructed, color='royalblue', linewidth=2, label='AI Denoised Signal')
    if len(peaks) > 0:
        plt.scatter(peaks, reconstructed[peaks], color='darkmagenta', s=90, zorder=5, label='Systolic Peaks')
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