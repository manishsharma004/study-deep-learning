# Optimization Techniques

Optimization is the process of minimizing the loss function to improve the performance of a neural network.

## Common Optimization Algorithms

1. **Stochastic Gradient Descent (SGD)**:
   - Updates weights using a small batch of data.
   - Formula: \[ w_{new} = w_{old} - \eta \frac{\partial L}{\partial w} \]
   - Pros: Simple and effective.
   - Cons: Can be slow to converge.

2. **Momentum**:
   - Accelerates SGD by adding a fraction of the previous update to the current update.
   - Formula: \[ v = \gamma v_{prev} + \eta \frac{\partial L}{\partial w} \]
     \[ w_{new} = w_{old} - v \]

3. **Adam**:
   - Combines momentum and adaptive learning rates.
   - Formula: \[ m_t = \beta_1 m_{t-1} + (1-\beta_1) \frac{\partial L}{\partial w} \]
     \[ v_t = \beta_2 v_{t-1} + (1-\beta_2) (\frac{\partial L}{\partial w})^2 \]
     \[ w_{new} = w_{old} - \frac{\eta}{\sqrt{v_t} + \epsilon} m_t \]

## Learning Rate Schedules

1. **Step Decay**:
   - Reduces the learning rate by a factor after a fixed number of epochs.

2. **Cosine Annealing**:
   - Gradually reduces the learning rate following a cosine curve.

3. **OneCycleLR**:
   - Increases the learning rate initially, then decreases it.

## Practical Tips

- Start with Adam for most tasks.
- Use learning rate schedules to improve convergence.
- Monitor training and validation loss to adjust hyperparameters.

## Tools and Resources

- [PyTorch Optimizers](https://pytorch.org/docs/stable/optim.html)
- [DeepLearning.AI Courses](https://www.deeplearning.ai/)
