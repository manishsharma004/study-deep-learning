# Backpropagation

Backpropagation is a fundamental algorithm used to train neural networks by minimizing the error between predicted and actual outputs.

## Key Concepts

1. **Forward Pass**:
   - Compute the output of the network for a given input.
   - Calculate the loss using a loss function (e.g., Mean Squared Error, Cross-Entropy Loss).

2. **Backward Pass**:
   - Compute gradients of the loss with respect to each weight using the chain rule of calculus.
   - Propagate the error backward through the network.

3. **Weight Update**:
   - Update weights using an optimization algorithm like Stochastic Gradient Descent (SGD) or Adam:
     \[ w_{\text{new}} = w_{\text{old}} - \eta \frac{\partial L}{\partial w} \]
     where \( \eta \) is the learning rate and \( \frac{\partial L}{\partial w} \) is the gradient of the loss with respect to the weight.

## Explanation of Symbols in Equations

To ensure clarity and alignment with scientific standards, here is a detailed explanation of the symbols used in the equations:

1. **Weight Update Equation**:
   \[ w_{\text{new}} = w_{\text{old}} - \eta \frac{\partial L}{\partial w} \]
   - **\( w_{\text{new}} \)**: The updated weight after applying the gradient descent step.
   - **\( w_{\text{old}} \)**: The current weight before the update.
   - **\( \eta \)**: The learning rate, a hyperparameter that controls the step size during optimization.
   - **\( \frac{\partial L}{\partial w} \)**: The gradient of the loss function \( L \) with respect to the weight \( w \). This measures how much the loss changes with a small change in \( w \).

2. **Gradient Calculation Using Chain Rule**:
   \[ \frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} * \frac{\partial z}{\partial w} \]
   - **\( \frac{\partial L}{\partial w} \)**: The gradient of the loss with respect to the weight.
   - **\( \frac{\partial L}{\partial z} \)**: The gradient of the loss with respect to an intermediate variable \( z \).
   - **\( \frac{\partial z}{\partial w} \)**: The gradient of the intermediate variable \( z \) with respect to the weight \( w \).
   - **\( z \)**: An intermediate variable, often representing the weighted sum of inputs in a neural network layer.

These symbols and equations are standard in scientific literature and align with conventions used in papers and textbooks on neural networks and optimization.

## Mathematical Foundation

- **Chain Rule**: Backpropagation relies on the chain rule to compute gradients efficiently.
  \[ \frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} * \frac{\partial z}{\partial w} \]
  where \( z \) is an intermediate variable.

- **Gradient Descent**: Gradients are used to adjust weights to minimize the loss function.

## Example: Backpropagation in a Simple Neural Network

### Problem Setup
We will train a simple neural network with one hidden layer to classify flowers in the Iris dataset based on their features.

### Code Example
```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target.reshape(-1, 1)  # Labels: 0, 1, 2 (Setosa, Versicolor, Virginica)

# One-hot encode the labels
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize weights and biases
np.random.seed(42)
weights_input_hidden = np.random.rand(4, 5)  # 4 inputs, 5 hidden neurons
weights_hidden_output = np.random.rand(5, 3)  # 5 hidden neurons, 3 outputs
bias_hidden = np.random.rand(1, 5)
bias_output = np.random.rand(1, 3)

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Learning rate
learning_rate = 0.1

# Training loop
for epoch in range(10000):
    # Forward pass
    hidden_layer_input = np.dot(X_train, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Compute loss (Mean Squared Error)
    loss = np.mean((y_train - predicted_output) ** 2)

    # Backward pass
    error_output_layer = y_train - predicted_output
    delta_output_layer = error_output_layer * sigmoid_derivative(predicted_output)

    error_hidden_layer = delta_output_layer.dot(weights_hidden_output.T)
    delta_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(delta_output_layer) * learning_rate
    weights_input_hidden += X_train.T.dot(delta_hidden_layer) * learning_rate
    bias_output += np.sum(delta_output_layer, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(delta_hidden_layer, axis=0, keepdims=True) * learning_rate

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Evaluate on the test set
hidden_layer_input = np.dot(X_test, weights_input_hidden) + bias_hidden
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
predicted_output = sigmoid(output_layer_input)

# Convert predictions to class labels
predicted_labels = np.argmax(predicted_output, axis=1)
true_labels = np.argmax(y_test, axis=1)

# Print accuracy
accuracy = np.mean(predicted_labels == true_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

### Explanation
1. **Dataset**:
   - The Iris dataset contains 150 samples of flowers, each with 4 features and 3 classes.
2. **Forward Pass**:
   - Compute activations for the hidden and output layers.
3. **Backward Pass**:
   - Calculate errors and gradients for the output and hidden layers.
4. **Weight Update**:
   - Adjust weights and biases using the gradients and learning rate.

### Visualization
You can visualize the dataset using libraries like Matplotlib or Seaborn to better understand the feature distributions and class separations.

## Practical Tips

- Use proper initialization (e.g., Xavier, He) to avoid vanishing/exploding gradients.
- Normalize inputs and use techniques like Batch Normalization.
- Monitor gradients during training to ensure they are not too small or too large.

## Tools and Resources

- [PyTorch Autograd Documentation](https://pytorch.org/docs/stable/autograd.html)
- [DeepLearning.AI Neural Networks and Deep Learning](https://www.deeplearning.ai/)
