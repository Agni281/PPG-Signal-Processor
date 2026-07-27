import numpy as np
import scipy.signal as signal 
import torch
import torch.nn as nn
import torch.optim as optim 
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

def gen_data(num_samples=500, length =200): 
  "Generate pairs of clean PPG waves and noisy equivalents" 
