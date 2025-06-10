import torch, os 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time

from torch.utils.cpp_extension import load

from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.observer import MinMaxObserver, default_observer

# Check if CUDA is available and set the device
device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Convert MNIST Image Files into a Tensor of 4-Dimensions (# of images, Height, Width, Color Channels)
transform = transforms.ToTensor()

# Train Data
train_data = datasets.MNIST(root='./cnn_data', train=True, download=True, transform=transform)

# Test Data
test_data = datasets.MNIST(root='./cnn_data', train=False, download=True, transform=transform)

# Create a small batch size for images...let's say 10
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

axx_mult = 'mul8s_acc'
axx_conv2d_kernel = load(name='PyInit_conv2d_'+axx_mult,
                         sources=["/scratch-local/khan/low_bit_quantization/nvidia_quant/adapt/adapt/cpu-kernels/axx_conv2d.cpp"],
                         extra_cflags = ['-DAXX_MULT=' + axx_mult + ' -march=native -fopenmp -O3' ],
                         extra_ldflags=['-lgomp'],
                         verbose=True)

def calculate_scale_and_zero_point_from_weights(weights, quant_min=-128, quant_max=127):
    """Calculates scale and zero-point given weights."""
    min_val = weights.min().item()
    max_val = weights.max().item()
    # Calculate scale
    scale = (max_val - min_val) / (quant_max - quant_min)
    # Calculate zero-point
    zero_point = quant_min - min_val / scale
    zero_point = int(round(zero_point))
    # Ensure zero-point is within quantization range
    zero_point = min(max(quant_min, zero_point), quant_max)
    return scale, zero_point

def quantize_weights_to_int8(weights, scale, zero_point, quant_min=-128, quant_max=127):
    """Quantizes weights to int8 using calculated scale and zero-point."""
    # Fake quantize to int8
    weights_int = torch.round(weights / scale + zero_point)
    weights_int = torch.clamp(weights_int, quant_min, quant_max)
    return weights_int.to(torch.int8)

def approx_multiplier_hook(module, input, output):
    # Simulating the effect of the custom approximate multiplier
    quint8_acts = input[0] # activations arrive as uint8
    # dequantized_acts = quint8_acts.dequantize()  # This returns a float tensor

    # adjusted_int8_acts = (dequantized_acts - quint8_acts.q_zero_point()) * quint8_acts.q_scale()
    # int8_acts = adjusted_int8_acts.round().to(dtype=torch.int8)
    
    ss, zz = calculate_scale_and_zero_point_from_weights(module.weight)
    int8_weights = quantize_weights_to_int8(module.weight, ss, zz)
    int8_acts = quint8_acts.to(dtype=torch.int8)
    output = axx_conv2d_kernel.forward(int8_acts, int8_weights, module.kernel_size, module.stride, module.padding)

    return output


def calculatescaleandzeropoint(min_val, max_val):
    quantmin = 0
    quantmax = 255
    scale = (max_val - min_val) / (quantmax - quantmin)
    zeropoint = torch.round(quantmin - min_val / scale)
    zeropoint = min(max(zeropoint, quantmin), quantmax)
    return scale, zeropoint


# Model Class
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Define custom QConfigs
        self.qconfig1 = QConfig(
            activation=MinMaxObserver.with_args(dtype=torch.qint8),
            weight=default_observer.with_args(dtype=torch.qint8))
        
        self.qconfig2 = QConfig(
            activation=MinMaxObserver.with_args(dtype=torch.int32),
            weight=default_observer.with_args(dtype=torch.qint8))
        
        self.quant = torch.ao.quantization.QuantStub(self.qconfig1)
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        # self.dequant = torch.ao.quantization.DeQuantStub(self.qconfig2)
        self.dequant = torch.quantization.DeQuantStub()
        
        # Fully Connected Layer
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, X):
        X = self.quant(X)
        min_val = self.quant.activation_post_process.min_val
        max_val = self.quant.activation_post_process.max_val
        scale, zero_point = calculatescaleandzeropoint(min_val, max_val)
        # print(min_val, max_val)
        
        X = self.conv1(X)
        X  = self.dequant(X)
        # print(torch.equal(X, y))
        # X = y
        X = (X.to(torch.float32) - zero_point) * scale
        X = F.max_pool2d(X, 2, 2)
        
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        
        X = X.reshape(-1, 16*5*5)
        
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        # X = self.dequant(X)
        return F.log_softmax(X, dim=1)

# Create an Instance of our Model
torch.manual_seed(41)
model = ConvolutionalNetwork()
print(model)

# Set the model to training mode
model.train()

# Prepare for QAT with custom QConfigs
model_prepared = torch.ao.quantization.prepare_qat(model)


# Register hooks (if you actually want to use them)
model_prepared.conv1.register_forward_hook(approx_multiplier_hook)
#model_prepared.conv2.register_forward_hook(approx_multiplier_hook)
#model_prepared.fc1.register_forward_hook(approx_multiplier_hook)
#model_prepared.fc2.register_forward_hook(approx_multiplier_hook)
#model_prepared.fc3.register_forward_hook(approx_multiplier_hook)

# Loss Function Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_prepared.parameters(), lr=0.001)

start_time = time.time()

# Create Variables To Track Things
epochs = 1
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0

    # Train
    model_prepared.train()
    for b, (X_train, y_train) in enumerate(train_loader):
        b += 1
        y_pred = model_prepared(X_train)
        loss = criterion(y_pred, y_train)
        
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b % 600 == 0:
            print(f'Epoch: {i}  Batch: {b}  Loss: {loss.item()}')

    train_losses.append(loss.item())
    train_correct.append(trn_corr.item())

    # Test
    model_prepared.eval()
    model_int8 = torch.ao.quantization.convert(model_prepared, inplace=True)
    
    with torch.no_grad():
        tst_corr = 0
        total_samples = 0
        
        for b, (X_test, y_test) in enumerate(test_loader):
            y_val = model_int8(X_test)
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum().item()
            total_samples += y_test.size(0)
        
        test_accuracy = 100 * tst_corr / total_samples
        test_loss = criterion(y_val, y_test)
        
        test_losses.append(test_loss.item())
        test_correct.append(test_accuracy)
        
        current_time = time.time()
        total = current_time - start_time
        print(f'Training Took: {total/60} minutes!')
        print(f'Epoch: {i}  Test Loss: {test_loss.item():.4f}  Test Accuracy: {test_accuracy:.2f}%')