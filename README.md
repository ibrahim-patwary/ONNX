## Overview

This repository contains machine learning models in **ONNX (Open Neural Network Exchange)** format. ONNX allows you to train models in one framework and use them for inference in others, enabling interoperability across different platforms and hardware.

ONNX models can be run on a variety of devices, ensuring **cross-platform** support for CPUs, GPUs, and specialized hardware accelerators.

## Usage
### Convert a PyTorch Model to ONNX and Run an ONNX Model for Inference

You can easily export a pre-trained PyTorch model to ONNX format using the following code:

```python
import torch
import torchvision.models as models
import onnx
import onnxruntime as ort
import numpy as np

# Load a pretrained model
model = models.resnet18(pretrained=True)
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model to ONNX
torch.onnx.export(model, dummy_input, "resnet18.onnx") 


# Load ONNX model
onnx_model = onnx.load("resnet18.onnx")

# Create an ONNX Runtime session
session = ort.InferenceSession("resnet18.onnx")

# Run the model with example input
inputs = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = session.run(None, {"input": inputs})
