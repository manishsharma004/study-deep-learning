# Week 3 — Text Preprocessing & Classical Representations

Time commitment: 5–8 hours

Overview
- Focus: Build a robust preprocessing pipeline and a TF‑IDF / BoW baseline for text classification.

Objectives
- Learn normalization, tokenization, stopword handling, stemming vs lemmatization, and n‑grams.
- Build TF‑IDF features and train a classical classifier using a PyTorch training loop (PyTorch-first).

Tasks
1. Environment: install Python 3.10+, scikit‑learn, pandas, NLTK, spaCy, Jupyter, and `torch`.
2. Data: choose a dataset (IMDB / SST‑2 / AG News). Create a small dev subset for quick iteration.
3. Preprocessing pipeline:
   - Normalize: lowercase, punctuation removal, whitespace, basic `unicode` cleanup.
   - Tokenize: compare whitespace split, NLTK tokenizers, and spaCy tokenization.
   - Optional: stemming (Porter) vs lemmatization (spaCy), stopword removal.
4. Feature extraction:
   - Build `CountVectorizer` and `TfidfVectorizer` (unigram and unigram+bigram).
   - Use `max_features` to cap vocabulary size for faster PyTorch models (e.g., 10k–50k).
   - TF‑IDF output is typically a sparse matrix (CSR). For small dev experiments you can convert to dense with `.toarray()`; for larger data keep it sparse and sample mini-batches.
5. Model & evaluation (PyTorch-first approach):
   - Goal: train a simple PyTorch classifier on TF‑IDF features so the baseline uses the same training loop you'll use for embeddings in Week 4.
   - Conversion steps (small dev set):
     1. Compute TF‑IDF features: `X = tfidf.transform(docs)` (CSR matrix).
     2. Convert to dense for dev runs: `X_dense = X.toarray()` and then `inputs = torch.from_numpy(X_dense).float()`; `labels = torch.from_numpy(y).long()`.
     3. Wrap into `TensorDataset(inputs, labels)` and `DataLoader` for batching.
   - Model: a single-layer linear classifier is sufficient:
     - `model = nn.Linear(input_dim, num_classes)`
     - Loss: `nn.CrossEntropyLoss()`
     - Optimizer: `torch.optim.Adam(model.parameters(), lr=1e-3)` or SGD.
   - Training loop (beginner-friendly):
     - For each epoch: set model.train(), iterate batches, `optimizer.zero_grad()`, `outputs = model(batch_inputs)`, `loss = criterion(outputs, batch_labels)`, `loss.backward()`, `optimizer.step()`.
     - Evaluate on a validation set each epoch: set model.eval(), compute predictions, and use `sklearn.metrics` (accuracy, `f1_score(average='macro')`) for reporting.
   - Memory note: converting large sparse TF‑IDF matrices to dense can blow up RAM; restrict `max_features`, use a dev subset, or consider sparse-aware approaches for production.
   - Focus: use PyTorch exclusively for model training to establish a consistent training loop pattern for Week 4.
6. Document: record preprocessing choices, feature shapes, and per‑fold metrics. Save hyperparameters (learning rate, batch size, seed) in a small JSON or CSV.

Deliverables
- `notebooks/week3_tfidf_baseline.ipynb` or `scripts/week3_tfidf_baseline.py` implementing the PyTorch training loop for TF‑IDF features.
- Small results file (CSV/JSON) with hyperparameters and per-fold metrics; saved model checkpoint (optional) for the PyTorch baseline.

Resources
- NLTK: https://www.nltk.org/
- spaCy: https://spacy.io/
- scikit-learn text features: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
- PyTorch: https://pytorch.org/
- Helpful metrics: `sklearn.metrics.accuracy_score`, `sklearn.metrics.f1_score(average='macro')`

Notes
- Keep runs short: use a dev subset and fixed random seeds for reproducibility.
- Change only one preprocessing choice per experiment to isolate effects.
- Use `torch.manual_seed(seed)` and set deterministic flags if needed for reproducibility.
- Device: for TF‑IDF linear models CPU is often sufficient; for embedding-based Week 4 models use GPU if available and call `.to(device)` on model and batches.

PyTorch training loop notes
- This PyTorch-first approach establishes the same training pattern you'll use for embedding-based models in Week 4.
- TF‑IDF features are computed with scikit-learn (for convenience) but all model training uses PyTorch exclusively.
- Benefits: consistent evaluation, device handling, and training loop patterns across classical and neural approaches.

Resources (add PyTorch)
- PyTorch: https://pytorch.org/
- TorchText (useful utilities): https://pytorch.org/text/
