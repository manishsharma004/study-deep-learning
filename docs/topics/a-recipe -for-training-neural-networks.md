# A Recipe for Training Neural Networks

This document summarizes practical heuristics for training neural networks, inspired by Andrej Karpathy's blog post "A Recipe for Training Neural Networks."

## Key Heuristics

1. **Initialization**:
   - Use Xavier or He initialization for weights.
   - Biases can be initialized to zero.

2. **Learning Rate**:
   - Start with a learning rate of 0.01 or 0.001.
   - Use learning rate schedules like StepLR, CosineAnnealingLR, or OneCycleLR.

3. **Regularization**:
   - Apply weight decay (e.g., 1e-4 or 1e-3).
   - Use dropout (p=0.2–0.5) to prevent overfitting.

4. **Batch Size**:
   - Start with a batch size of 32 or 64.
   - Experiment with larger batch sizes if using GPUs.

5. **Optimization**:
   - Use Adam or SGD with momentum.
   - Tune hyperparameters like learning rate and momentum.

6. **Monitoring**:
   - Track training and validation loss/accuracy.
   - Use tools like TensorBoard, Weights & Biases, or MLflow.

7. **Debugging**:
   - Check gradients and activations for vanishing/exploding values.
   - Visualize loss curves to identify issues like overfitting or underfitting.

## Common Pitfalls

- **Overfitting**: Use regularization techniques and monitor validation performance.
- **Underfitting**: Increase model capacity or train for more epochs.
- **Vanishing/Exploding Gradients**: Use proper initialization and normalization techniques.

## Tools and Resources

- [Karpathy's Blog Post](https://karpathy.github.io/2019/04/25/recipe/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [DeepLearning.AI Courses](https://www.deeplearning.ai/)
