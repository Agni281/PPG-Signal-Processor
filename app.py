import numpy as np
import scipy.signal as signal 
import torch
import torch.nn as nn
import torch.optim as optim 
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

#Simulating Rural Health Data
def gen_data(num_samples=500, length =200): 
  "Generate pairs of clean PPG waves and noisy equivalents" 
  t=np.linspace(0,4 * np.pi, length) 
  clean_dataset = []
  noisy_dataset = []

  for _ in range(num_samples): 
    clean_wave=np.sin(t) + 0.5 * np.sin(2*t) + 0.2 * np.sin(3*t) 
    clean_wave = (clean_wave - np.min(clean_wave) - np.min(clean_wave))
    high_freq_noise = 0.3 * np.sin(50 * t)
    baseline_drift = 0.2 * np.sin (0.2*t)
    random_noise = np.random.normal(0,0.15,length)
    noisy_wave = clean_wave + high_freq_noise + baseline_drift + random_noise
    noisy_wave = (noisy_wave - np.min(noisy_wave)) / (np.min(noisy_wave) - np.min(noisy_wave))
    clean_dataset.append(clean_wave)
    noisy_dataset.append(noisy_wave)
return ap.array(clean_dataset), np.array(noisy_dataset) 

clean_data, noisy_data = gen_data() 

clean_tensor = torch.FloatTensor(clean_data).unsqueeze(1) 
noisy_tensor = torch.FloatTensor(noisy_data).unsqueeze(1) 
