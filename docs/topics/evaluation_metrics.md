# Evaluation Metrics for Classification

Important metrics
- Accuracy: overall percent correct (can be misleading for imbalanced classes).
- Precision / Recall / F1: per-class measures; report macro‑F1 to treat classes equally.
- Confusion Matrix: inspect common errors and class-specific mistakes.

Implementation
- Use `sklearn.metrics` for computation (accuracy_score, f1_score with `average='macro'`).

Resources
- `scikit-learn` metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
