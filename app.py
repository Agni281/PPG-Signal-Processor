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

# --- 1. MODEL DEFINITION (FIXED CHANNEL PROGRESSION) ---
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

# --- 2. SIGNAL GENERATOR & SQI HELPERS ---
def generate_synthetic_ppg(t, hr_bpm):
    freq = hr_bpm / 60.0
    ppg = np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(4 * np.pi * freq * t + 0.5)
    return (ppg - np.min(ppg)) / (np.max(ppg) - np.min(ppg) + 1e-8)

def calculate_sqi(sig, fs=50.0):
    freqs, psd = signal.welch(sig, fs=fs, nperseg=len(sig))
    cardiac_band_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 4.0)])
    total_power = np.sum(psd) + 1e-8
    sqi = (cardiac_band_power / total_power) * 100.0
    return np.clip(sqi, 0.0, 100.0)

def butter_bandpass_filter(data, lowcut=0.5, highcut=4.0, fs=50.0, order=2):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

# --- 3. TRAIN BASE MODEL ---
print("Training base model for notebook dashboard...")
FS = 50.0
length = 200
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

# --- 4. INTERACTIVE PIPELINE FUNCTION ---
def run_interactive_pipeline(target_hr, noise_level, hum_freq, drift_level):
    true_clean = generate_synthetic_ppg(t, target_hr)
    
    high_freq_noise = noise_level * np.sin(2 * np.pi * hum_freq * t)
    baseline_drift = drift_level * np.sin(2 * np.pi * 0.2 * t)
    random_noise = np.random.normal(0, noise_level * 0.5, length)
    
    raw_noisy = true_clean + high_freq_noise + baseline_drift + random_noise
    raw_noisy = (raw_noisy - np.min(raw_noisy)) / (np.max(raw_noisy) - np.min(raw_noisy) + 1e-8)
    
    filtered_signal = butter_bandpass_filter(raw_noisy, lowcut=0.5, highcut=4.0, fs=FS)
    filtered_signal = (filtered_signal - np.min(filtered_signal)) / (np.max(filtered_signal) - np.min(filtered_signal) + 1e-8)
    
    model.eval()
    with torch.no_grad():
        input_sample = torch.FloatTensor(raw_noisy).unsqueeze(0).unsqueeze(0)
        reconstructed = model(input_sample).squeeze().numpy()
        
    sqi_score = calculate_sqi(reconstructed, fs=FS)
    
    min_distance = max(1, int(FS * 0.35))
    peaks, _ = signal.find_peaks(reconstructed, distance=min_distance, prominence=0.15)
    
    if len(peaks) >= 2:
        rr_intervals = np.diff(t[peaks])
        est_bpm = 60.0 / np.mean(rr_intervals)
        bpm_str = f"{est_bpm:.1f} BPM"
    else:
        bpm_str = "N/A"

    print(f"SQI Score: {sqi_score:.1f}% | Est. Heart Rate: {bpm_str} | Peaks Counted: {len(peaks)}")
    
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
    plt.title(f'AI Reconstructed Signal (SQI Score: {sqi_score:.1f}%)')
    plt.legend(loc='upper right')
    
    plt.subplot(4, 1, 4)
    plt.plot(true_clean, color='forestgreen', linestyle='--', label='Target Reference Wave')
    plt.title(f'Ground Truth Baseline (Simulated HR: {target_hr} BPM)')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

print("--- PPG SIGNAL DASHBOARD ---")
interact(
    run_interactive_pipeline,
    target_hr=IntSlider(min=50, max=140, step=1, value=75, description='Simulated BPM'),
    noise_level=FloatSlider(min=0.0, max=0.8, step=0.05, value=0.3, description='Noise Amp'),
    hum_freq=IntSlider(min=10, max=100, step=10, value=50, description='Grid Hum (Hz)'),
    drift_level=FloatSlider(min=0.0, max=0.5, step=0.05, value=0.2, description='Baseline Drift')
);