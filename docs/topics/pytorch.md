# PyTorch

PyTorch is an open-source deep learning framework widely used for research and production.

## Key Features

1. **Dynamic Computation Graphs**:
   - Build and modify computation graphs on the fly.

2. **Autograd**:
   - Automatic differentiation for computing gradients.

3. **TorchScript**:
   - Convert PyTorch models to a production-ready format.

4. **Extensive Library Support**:
   - Includes `torchvision`, `torchtext`, and `torchaudio` for handling images, text, and audio.

5. **Community and Ecosystem**:
   - Large community and extensive resources.

## Getting Started

1. **Installation**:
   - Install PyTorch using pip or conda:
     ```bash
     pip install torch torchvision torchaudio
     ```

2. **Basic Workflow**:
   - Define a model using `torch.nn.Module`.
   - Use `torch.optim` for optimization.
   - Use `torch.utils.data` for data loading and preprocessing.

3. **Example**:
   ```python
   import torch
   import torch.nn as nn
   import torch.optim as optim

   # Define a simple model
   class SimpleModel(nn.Module):
       def __init__(self):
           super(SimpleModel, self).__init__()
           self.fc = nn.Linear(10, 1)

       def forward(self, x):
           return self.fc(x)

   model = SimpleModel()
   optimizer = optim.SGD(model.parameters(), lr=0.01)
   criterion = nn.MSELoss()

   # Dummy data
   inputs = torch.randn(5, 10)
   targets = torch.randn(5, 1)

   # Training step
   optimizer.zero_grad()
   outputs = model(inputs)
   loss = criterion(outputs, targets)
   loss.backward()
   optimizer.step()
   ```

## Tools and Resources

- [PyTorch Official Documentation](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Examples](https://github.com/pytorch/examples)
- [DeepLearning.AI PyTorch Specialization](https://www.deeplearning.ai/)
