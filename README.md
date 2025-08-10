# Deep Learning Study

A personal study project for understanding deep learning concepts through practical implementation and documentation.

## Project Structure

- `docs/` - Documentation and study notes
  - `topics/` - Topic-specific explanations and examples
- `notebooks/` - Jupyter notebooks with hands-on implementations
- `src/` - Source code (if any)
- `main.py` - Main entry point

## Topics Covered

- **Backpropagation** - Manual implementation and practical examples using the Iris dataset
- **Optimization** - Gradient descent and other optimization algorithms
- **PyTorch** - Deep learning framework fundamentals

## Getting Started

### Prerequisites

- Python 3.8+
- Recommended: Virtual environment

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd study-deep-learning
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or if using uv:
   ```bash
   uv sync
   ```

### Running the Examples

- **Jupyter Notebooks**: Open and run notebooks in the `notebooks/` directory
- **Main Script**: Run `python main.py` for basic examples

## Learning Path

1. Start with the documentation in `docs/topics/`
2. Work through the Jupyter notebooks for hands-on practice
3. Experiment with the code examples

## System Requirements

This project runs well on:
- **CPU**: Ryzen 5 5600H or equivalent (6 cores, 12 threads recommended)
- **GPU**: Optional for basic examples, RX5500M or better for advanced topics
- **RAM**: 8GB+ recommended
- **Storage**: Minimal requirements

## Notes

This is a study project focusing on understanding fundamentals through manual implementation before transitioning to frameworks like PyTorch.

## License

This project is for educational purposes.
