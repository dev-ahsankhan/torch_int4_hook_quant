import torch, os 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.cpp_extension import load
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer

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
quant_desc = QuantDescriptor(num_bits=8, fake_quant=False, unsigned=False, calib_method='histogram')
quantizer = TensorQuantizer(quant_desc)
quantizer_w = TensorQuantizer(quant_desc)

axx_mult = 'mul8s_acc'
axx_conv2d_kernel = load(name='PyInit_conv2d_'+axx_mult,
                         sources=["/scratch-local/khan/low_bit_quantization/nvidia_quant/adapt/adapt/cpu-kernels/axx_conv2d.cpp"],
                         extra_cflags = ['-DAXX_MULT=' + axx_mult + ' -march=native -fopenmp -O3' ],
                         extra_ldflags=['-lgomp'],
                         verbose=True)
# Define the approximate multiplier hook
def approx_multiplier_hook2(module, input, output):
    # Simulating the effect of the custom approximate multiplier

    # print(f"Post-forward hook in {module.__class__.__name8__}")
    # Example of modifying the output in some way to simulate approximation
    # Dummy multiplication factor
    #multiplier = 0.9  # This represents the custom multiplier effect
    
    breakpoint()
    weight = module.weight().int_repr().to(dtype=torch.int8)
    input = input[0].int_repr().to(dtype=torch.int8)
    output = axx_conv2d_kernel.forward(input, weight, module.kernel_size, module.stride, module.padding)
    # print(output.dtype)
    return output

def approx_multiplier_hook(module, input, output):
    # Simulating the effect of the custom approximate multiplier

    # print(f"Post-forward hook in {module.__class__.__name8__}")
    # Example of modifying the output in some way to simulate approximation
    # Dummy multiplication factor
    #multiplier = 0.9  # This represents the custom multiplier effect

    
    quint8_weights = input[0] # activations arrive as uint8     #scale & zero-point comes from data
    dequantized_weights = quint8_weights.dequantize()  # This returns a float tensor
    
    adjusted_int8_weights = (dequantized_weights - quint8_weights.q_zero_point()) * quint8_weights.q_scale()
    int8_acts = adjusted_int8_weights.round().to(dtype=torch.int8)
    weight = module.weight().int_repr().to(dtype=torch.int8)
    output = axx_conv2d_kernel.forward(int8_acts, weight, module.kernel_size, module.stride, module.padding)
    return output

# Model Class
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # QuantStub converts tensors from floating point to quantized
        qconfig1 = QConfig(
            activation=MinMaxObserver.with_args(dtype=torch.quint8),
            weight=default_observer.with_args(dtype=torch.qint8))
        
        qconfig2 = QConfig(
            activation=MinMaxObserver.with_args(dtype=torch.int32),
            weight=default_observer.with_args(dtype=torch.qint8))
        
        self.quant = torch.ao.quantization.QuantStub(qconfig1)
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.dequant = torch.ao.quantization.DeQuantStub(qconfig2)
        self.bn = torch.nn.BatchNorm2d(1)
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.quant = torch.ao.quantization.QuantStub(qconfig1
        self.fc_layer = nn.Linear(28*28, 10)
        self.dequant = torch.ao.quantization.DeQuantStub(qconfig2)


    def forward(self, x):
        x = self.quant(x)
        print(x.dtype)
        x = self.conv(x)
        
        x = self.relu(x)
        
        x = self.flatten(x)
        
        x = self.dequant(x)
        x = self.quant(x)
        x = self.fc_layer(x)
        return x

# Create an Instance of our Model
torch.manual_seed(41)
model = ConvolutionalNetwork().to(device)  # Move model to the device
model_fp32 = model.eval()
model_fp32.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
model_fp32_prepared = torch.ao.quantization.prepare_qat(model_fp32.train())
model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

model = model_int8

int8_tensor = model.conv.weight()
int8_values = int8_tensor.int_repr()
print(int8_values)

# Register the approximate multiplier post-forward hook
model.conv.register_forward_hook(approx_multiplier_hook)
model.fc_layer.register_forward_hook(approx_multiplier_hook)
# model.fc1.register_forward_hook(approx_multiplier_hook)
# model.fc2.register_forward_hook(approx_multiplier_hook)
# model.fc3.register_forward_hook(approx_multiplier_hook)

# Loss Function Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

import time
start_time = time.time()

# Create Variables To Track Things
epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []

# For Loop of Epochs
for i in range(epochs):
    trn_corr = 0
    tst_corr = 0

    # Train
    for b, (X_train, y_train) in enumerate(train_loader):
        b += 1  # start our batches at 1
        # X_train, y_train = X_train.to(device), y_train.to(device)  # Move data to device
        y_pred = model(X_train) 
        breakpoint()
        loss = criterion(y_pred, y_train) 

        predicted = torch.max(y_pred.data, 1)[1] 
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr

        # Update our parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print out some results
        if b % 600 == 0:
            print(f'Epoch: {i}  Batch: {b}  Loss: {loss.item()}')

    train_losses.append(loss.item())
    train_correct.append(trn_corr.item())

    # Test
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            X_test, y_test = X_test.to(device), y_test.to(device)  # Move data to device
            y_val = model(X_test)
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss.item())
    test_correct.append(tst_corr.item())

current_time = time.time()
total = current_time - start_time
print(f'Training Took: {total / 60} minutes!')