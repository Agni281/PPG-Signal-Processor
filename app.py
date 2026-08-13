import os
import pandas as pd
import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import ipywidgets as widgets
from ipywidgets import interact, FloatSlider, IntSlider

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ==========================================
# 1. 1D U-NET MODEL ARCHITECTURE
# ==========================================
class SignalDenoisingUNet1D(nn.Module):
    def __init__(self):
        super(SignalDenoisingUNet1D, self).__init__()
        # Encoder (Downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        # Decoder (Upsampling)
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)       # Shape: [Batch, 16, 100]
        e2 = self.enc2(e1)      # Shape: [Batch, 32, 50]
        d2 = self.dec2(e2)      # Shape: [Batch, 16, 100]
        
        # Skip Connection: Concatenate decoder output with encoder output
        cat1 = torch.cat((d2, e1), dim=1)  # Shape: [Batch, 32, 100]
        out = self.dec1(cat1)              # Shape: [Batch, 1, 200]
        return out


# ==========================================
# 2. REAL-WORLD DATASET LOADER
# ==========================================
class RealWorldPPGDataset(Dataset):
    def __init__(self, csv_filepaths, window_length=200, raw_fs=500.0, target_fs=50.0):
        self.samples = []
        self.window_length = window_length
        
        for file in csv_filepaths:
            if not os.path.exists(file):
                print(f"Warning: File {file} not found. Skipping...")
                continue
                
            df = pd.read_csv(file)
            
            # Select pleth_1 or fallback to any pleth column
            if 'pleth_1' in df.columns:
                raw_sig = df['pleth_1'].values.astype(np.float32)
            else:
                pleth_cols = [c for c in df.columns if 'pleth' in c.lower()]
                if pleth_cols:
                    raw_sig = df[pleth_cols[0]].values.astype(np.float32)
                else:
                    continue
                
            raw_sig = raw_sig[~np.isnan(raw_sig)]
            
            # Resample native 500 Hz data down to 50 Hz target
            num_samples_target = int(len(raw_sig) * (target_fs / raw_fs))
            resampled_sig = signal.resample(raw_sig, num_samples_target)
            
            # Slice into 200-sample windows (50% overlap)
            step_size = window_length // 2
            for start in range(0, len(resampled_sig) - window_length, step_size):
                chunk = resampled_sig[start : start + window_length]
                norm_chunk = (chunk - np.min(chunk)) / (np.max(chunk) - np.min(chunk) + 1e-8)
                self.samples.append(norm_chunk)
                
        self.samples = np.array(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clean_target = self.samples[idx]
        
        # Add motion & drift noise to train robust denoising
        t = np.linspace(0, 4, self.window_length)
        motion_noise = 0.25 * np.sin(2 * np.pi * 0.3 * t) + np.random.normal(0, 0.08, self.window_length)
        noisy_input = clean_target + motion_noise
        noisy_input = (noisy_input - np.min(noisy_input)) / (np.max(noisy_input) - np.min(noisy_input) + 1e-8)
        
        return torch.FloatTensor(noisy_input).unsqueeze(0), torch.FloatTensor(clean_target).unsqueeze(0)


# ==========================================
# 3. HELPER & METRIC FUNCTIONS
# ==========================================
def butter_bandpass_filter(data, lowcut=0.5, highcut=4.0, fs=50.0, order=2):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def calculate_sqi(sig, fs=50.0):
    freqs, psd = signal.welch(sig, fs=fs, nperseg=len(sig))
    cardiac_band_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 4.0)])
    total_power = np.sum(psd) + 1e-8
    sqi = (cardiac_band_power / total_power) * 100.0
    return np.clip(sqi, 0.0, 100.0)


# ==========================================
# 4. TRAINING THE U-NET ON REAL DATA
# ==========================================
# List your local training CSV files
training_files = ['s20_walk.csv']

dataset = RealWorldPPGDataset(csv_filepaths=training_files, window_length=200, raw_fs=500.0, target_fs=50.0)

if len(dataset) > 0:
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = SignalDenoisingUNet1D()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training U-Net on {len(dataset)} windows extracted from real CSV data...")
    model.train()
    num_epochs = 30

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for noisy_batch, clean_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(noisy_batch)
            loss = criterion(outputs, clean_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] — Loss: {epoch_loss / len(dataloader):.6f}")

    # Save weights to load in Streamlit
    torch.save(model.state_dict(), 'unet_realdata_weights.pth')
    print("✅ Model trained and weights saved to unet_realdata_weights.pth!")
else:
    print("⚠️ No valid CSV data loaded. Please make sure 's20_walk.csv' is in your directory.")


# ==========================================
# 5. INTERACTIVE TESTING DASHBOARD
# ==========================================
FS = 50.0
length = 200
t = np.linspace(0, 4.0, length)

def test_interactive_pipeline(window_index):
    if len(dataset) == 0:
        print("Cannot run test visualization without loaded dataset samples.")
        return
        
    window_idx = min(window_index, len(dataset) - 1)
    noisy_tensor, clean_tensor = dataset[window_idx]
    
    raw_noisy = noisy_tensor.squeeze().numpy()
    true_clean = clean_tensor.squeeze().numpy()
    
    # Classical Filter
    filtered_signal = butter_bandpass_filter(raw_noisy, lowcut=0.5, highcut=4.0, fs=FS)
    filtered_signal = (filtered_signal - np.min(filtered_signal)) / (np.max(filtered_signal) - np.min(filtered_signal) + 1e-8)
    
    # U-Net Model Inference
    model.eval()
    with torch.no_grad():
        input_sample = noisy_tensor.unsqueeze(0)
        reconstructed = model(input_sample).squeeze().numpy()
        
    sqi_score = calculate_sqi(reconstructed, fs=FS)
    min_distance = max(1, int(FS * 0.35))
    peaks, _ = signal.find_peaks(reconstructed, distance=min_distance, prominence=0.15)
    
    if len(peaks) >= 2:
        est_bpm = 60.0 / np.mean(np.diff(t[peaks]))
        bpm_str = f"{est_bpm:.1f} BPM"
    else:
        bpm_str = "N/A"

    print(f"Window: {window_idx} | SQI: {sqi_score:.1f}% | Est. Heart Rate: {bpm_str} | Peaks: {len(peaks)}")
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(t, raw_noisy, color='crimson', label='Raw Input Window')
    plt.title(f'Stage 1: Raw Real Data Stream (Window #{window_idx})')
    plt.legend(loc='upper right')
    
    plt.subplot(3, 1, 2)
    plt.plot(t, filtered_signal, color='darkorange', label='SciPy Bandpass Filtered')
    plt.title('Stage 2: Classical Bandpass Filter (0.5 - 4.0 Hz)')
    plt.legend(loc='upper right')
    
    plt.subplot(3, 1, 3)
    plt.plot(t, reconstructed, color='royalblue', linewidth=2, label='1D U-Net Output')
    if len(peaks) > 0:
        plt.scatter(t[peaks], reconstructed[peaks], color='purple', s=80, zorder=5, label=f'Detected Peaks ({len(peaks)})')
    plt.title(f'Stage 3: 1D U-Net Reconstructed Signal (SQI Score: {sqi_score:.1f}%)')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

# Run interactive slider across dataset windows
if len(dataset) > 0:
    interact(
        test_interactive_pipeline,
        window_index=IntSlider(min=0, max=max(0, len(dataset) - 1), step=1, value=0, description='Window Index')
    );