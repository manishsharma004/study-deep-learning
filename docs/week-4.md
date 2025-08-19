# Week 4 — Pretrained Embeddings & Small Neural Baselines

Time commitment: 5–8 hours

Overview
- Focus: Load pretrained embeddings (GloVe/fastText), build a small PyTorch model, and compare to the TF‑IDF baseline.

Objectives
- Learn to create an embedding matrix from pretrained vectors and handle OOV tokens.
- Implement Dataset/DataLoader with a collate function and a small model (average-pooled embeddings → MLP).

Tasks
1. Choose embeddings: GloVe (50/100/200d) or fastText (better OOV handling).
2. Build vocabulary and embedding matrix; initialize OOV embeddings.
3. Implement Dataset + DataLoader with padding and masks.
4. Implement two small models: average-pooled embeddings→MLP and optional 1D‑CNN.
5. Train using CrossEntropyLoss and Adam/AdamW; apply early stopping and track val metrics.
6. Compare performance and runtime to TF‑IDF baseline.

Deliverables
- `notebooks/week4_embedding_baseline.ipynb` or `scripts/week4_embedding_baseline.py`.
- Saved best checkpoint and short comparison write-up.

Resources
- GloVe: https://nlp.stanford.edu/projects/glove/
- fastText: https://fasttext.cc/
- Practical PyTorch NLP: https://github.com/bentrevett/pytorch-sentiment-analysis

Notes
- Use small embedding sizes and smaller datasets for CPU experiments to keep iterations fast.
