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

   This example demonstrates a basic workflow in PyTorch, a popular deep learning framework:

   1. **Model Definition**:
      - The `SimpleModel` class inherits from `torch.nn.Module`, which is the base class for all neural networks in PyTorch.
      - Inside the model, a single fully connected (linear) layer is defined using `nn.Linear(10, 1)`. This means the model takes an input of size 10 and outputs a single value.
      - The `forward` method defines how the input data flows through the model. In this case, it simply passes the input through the linear layer.

   2. **Loss Function**:
      - The loss function measures how far the model's predictions are from the actual target values. Here, `nn.MSELoss` is used, which calculates the Mean Squared Error (MSE) between predictions and targets. MSE is commonly used for regression tasks.

   3. **Optimizer**:
      - The optimizer updates the model's parameters to minimize the loss. `optim.SGD` is used here, which implements the Stochastic Gradient Descent algorithm. The learning rate (`lr=0.01`) controls the step size for each update.

   4. **Dummy Data**:
      - Random input data (`inputs`) and target values (`targets`) are generated using `torch.randn`. This is just for demonstration purposes; in a real scenario, you would use actual data.

   5. **Training Step**:
      - The training process involves the following steps:
        - **Zero Gradients**: `optimizer.zero_grad()` clears the gradients from the previous step. This is necessary because PyTorch accumulates gradients by default.
        - **Forward Pass**: `model(inputs)` computes the model's predictions for the input data.
        - **Loss Computation**: `criterion(outputs, targets)` calculates the loss between the predictions and the actual targets.
        - **Backward Pass**: `loss.backward()` computes the gradients of the loss with respect to the model's parameters.
        - **Parameter Update**: `optimizer.step()` updates the model's parameters using the computed gradients.

   This workflow is the foundation of training neural networks in PyTorch. By iterating over these steps with real data, the model learns to make better predictions over time.
