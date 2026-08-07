import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

# --- 1. UPGRADED 1D U-NET MODEL ---
class UNet1DSignalDenoiser(nn.Module):
    def __init__(self):
        super(UNet1DSignalDenoiser, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        d2 = self.dec2(e2)
        d2_cat = torch.cat([d2, e1], dim=1)
        out = self.dec1(d2_cat)
        return out

def butter_bandpass_filter(data, lowcut=0.5, highcut=4.0, fs=50.0, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def generate_base_dataset(num_samples=800, length=200):
    t = np.linspace(0, 4 * np.pi, length)
    base_wave = np.sin(t) + 0.35 * np.sin(2 * t) + 0.15 * np.cos(3 * t)
    base_wave = (base_wave - np.min(base_wave)) / (np.max(base_wave) - np.min(base_wave))

    clean_dataset, noisy_dataset = [], []

    for _ in range(num_samples):
        shift = np.random.randint(-15, 15)
        clean_wave = np.roll(base_wave, shift)

        n_amp = np.random.uniform(0.1, 0.5)
        grid_freq = np.random.choice([50, 60, 20, 80])
        drift_amp = np.random.uniform(0.05, 0.3)

        hf_noise = n_amp * np.sin(grid_freq * t)
        drift = drift_amp * np.sin(0.3 * t)
        gauss = np.random.normal(0, n_amp * 0.25, length)

        noisy_wave = clean_wave + hf_noise + drift + gauss
        noisy_wave = (noisy_wave - np.min(noisy_wave)) / (np.max(noisy_wave) - np.min(noisy_wave) + 1e-8)

        clean_dataset.append(clean_wave)
        noisy_dataset.append(noisy_wave)

    clean_tensor = torch.FloatTensor(np.array(clean_dataset)).unsqueeze(1)
    noisy_tensor = torch.FloatTensor(np.array(noisy_dataset)).unsqueeze(1)

    return clean_tensor, noisy_tensor, base_wave

clean_tensor, noisy_tensor, base_signal = generate_base_dataset()

model = UNet1DSignalDenoiser()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

print("Training Upgraded U-Net Signal Denoiser...")
for epoch in range(50):
    permutation = torch.randperm(noisy_tensor.size(0))
    for i in range(0, noisy_tensor.size(0), 32):
        indices = permutation[i:i+32]
        optimizer.zero_grad()
        outputs = model(noisy_tensor[indices])

        mse_loss = criterion(outputs, clean_tensor[indices])
        diff_pred = outputs[:, :, 1:] - outputs[:, :, :-1]
        diff_true = clean_tensor[indices][:, :, 1:] - clean_tensor[indices][:, :, :-1]
        grad_loss = criterion(diff_pred, diff_true)

        total_loss = mse_loss + 0.5 * grad_loss
        total_loss.backward()
        optimizer.step()

print("Training Complete.")

def process_and_plot(hum_freq=50, drift_level=0.2, noise_amp=0.3):
    FS = 50.0
    length = 200
    t = np.linspace(0, length / FS, length)

    true_clean = base_signal
    high_freq_noise = noise_amp * np.sin(2 * np.pi * hum_freq * t)
    baseline_drift = drift_level * np.sin(2 * np.pi * 0.2 * t)
    random_noise = np.random.normal(0, noise_amp * 0.3, length)

    raw_noisy = true_clean + high_freq_noise + baseline_drift + random_noise
    raw_noisy = (raw_noisy - np.min(raw_noisy)) / (np.max(raw_noisy) - np.min(raw_noisy) + 1e-8)

    filtered_signal = butter_bandpass_filter(raw_noisy, lowcut=0.5, highcut=4.0, fs=FS)
    filtered_signal = (filtered_signal - np.min(filtered_signal)) / (np.max(filtered_signal) - np.min(filtered_signal) + 1e-8)

    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(raw_noisy).unsqueeze(0).unsqueeze(0)
        reconstructed = model(input_tensor).squeeze().numpy()

    mse_score = np.mean((reconstructed - true_clean) ** 2)
    raw_ncc = np.corrcoef(reconstructed, true_clean)[0, 1]
    ncc_score = 0.0 if np.isnan(raw_ncc) else float(raw_ncc)
    confidence_score = max(0.0, ncc_score * np.exp(-1.0 * mse_score)) * 100.0

    peaks, _ = signal.find_peaks(reconstructed, distance=int(FS * 0.4), prominence=0.15)
    if len(peaks) > 1:
        peak_intervals_sec = np.diff(peaks) / FS
        heart_rate_bpm = 60.0 / np.mean(peak_intervals_sec)
        interval_variance = np.var(peak_intervals_sec)
    else:
        heart_rate_bpm = 0
        interval_variance = 0.0

    print(f"--- Pipeline Execution Summary ---")
    print(f"AI Confidence: {confidence_score:.1f}% | Reconstruction MSE: {mse_score:.4f}")
    print(f"Calculated Heart Rate: {heart_rate_bpm:.1f} BPM")

    if confidence_score < 60.0:
        print("Conclusion: UNRELIABLE DATA - High noise level.")
    elif interval_variance > 0.05 and heart_rate_bpm > 0:
        print("Conclusion: ALERT - Potential Arrhythmia / High HRV Detected.")
    else:
        print("Conclusion: HEALTHY - Normal Rhythm Reconstructed.")

    plt.figure(figsize=(12, 8))
    
    plt.subplot(4, 1, 1)
    plt.plot(raw_noisy, color='crimson', label='Raw Noisy Signal')
    plt.title("Stage 1: Raw Telemetry Stream")
    plt.legend(loc='upper right')

    plt.subplot(4, 1, 2)
    plt.plot(filtered_signal, color='darkorange', label='SciPy Bandpass Filtered')
    plt.title("Stage 2: Classical Bandpass Filter")
    plt.legend(loc='upper right')

    plt.subplot(4, 1, 3)
    plt.plot(reconstructed, color='royalblue', linewidth=2, label='AI Denoised Signal')
    if len(peaks) > 0:
        plt.scatter(peaks, reconstructed[peaks], color='darkmagenta', s=80, zorder=5, label='Systolic Peaks')
    plt.title(f"Stage 3: AI Reconstructed Wave (Confidence: {confidence_score:.1f}%)")
    plt.legend(loc='upper right')

    plt.subplot(4, 1, 4)
    plt.plot(true_clean, color='forestgreen', linestyle='--', label='Ground Truth Reference')
    plt.title("Stage 4: Reference Signal")
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_and_plot()