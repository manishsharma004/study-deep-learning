# Week 3–4: NLP Fundamentals and Text Representations

Time commitment: 5–8 hours/week (same cadence as `month-1.md`)

## Overview
These two weeks cover practical NLP preprocessing and classical + embedding-based representations. The aim is to move from raw text to features you can train on, then build two reproducible baselines (classical TF‑IDF + logistic regression and a small PyTorch model using pretrained embeddings). The plan below mirrors the structure and tone used in `month-1.md` so it fits the rest of the docs.

## Checklist (requirements)
- Read and follow the existing repo style in `docs/month-1.md` and `docs/plan.md`.
- Produce a concise, complete Week 3–4 plan with tasks, experiments and deliverables.
- Add links to important resources.
- Save the plan as `docs/week-3-4.md` (updated here).

---

## Objectives (clear, measurable)
- Convert raw text into cleaned, tokenized inputs suitable for ML models.
- Implement a classical BoW/TF‑IDF baseline and a small neural baseline using pretrained embeddings.
- Run basic experiments that isolate preprocessing and model choices (one variable at a time).
- Produce reproducible artifacts: notebook/script, saved metrics, short write-up.

---

## Week-by-week breakdown

### Week 3 — Text preprocessing & classical features
- Goal: Build a robust preprocessing pipeline and a TF‑IDF baseline.
- Tasks:
  1. Setup environment (Python 3.10+, scikit‑learn, NLTK/spaCy, pandas, Jupyter).
  2. Implement preprocessing functions: normalize (lowercase), strip punctuation, normalize whitespace, basic unicode cleanup.
  3. Tokenization experiments: whitespace, NLTK word tokenizer, spaCy tokenization. Compare outputs on a small sample.
  4. Optional: stemming (Porter) vs lemmatization (spaCy) and stopword removal; decide which to use for the baseline.
  5. Construct TF‑IDF features (unigrams ± bigrams) with `sklearn.feature_extraction.text.TfidfVectorizer`.
  6. Train a logistic regression or linear SVM baseline with 5-fold CV. Track accuracy and macro‑F1.
- Deliverables:
  - `notebooks/week3_tfidf_baseline.ipynb` or `scripts/tfidf_baseline.py` (reproducible run).
  - A CSV or JSON with per-fold metrics and a short results section in the notebook describing preprocessing choices.

### Week 4 — Pretrained embeddings & small neural baseline
- Goal: Load pretrained embeddings, build a small PyTorch model, and compare to TF‑IDF baseline.
- Tasks:
  1. Choose embeddings: GloVe (common sizes: 50/100/200d) or fastText (advantage: OOV handling).
  2. Build vocabulary from training data; initialize embedding matrix using pretrained vectors; set OOV token to random normal or zero.
  3. Implement Dataset + DataLoader with a collate function (pad sequences, return lengths or mask).
  4. Implement two small models:
     - Average-pooled embeddings -> dropout -> MLP -> softmax (fast to train, strong baseline).
     - Optional: 1D-CNN over embeddings (small receptive field, test as ablation).
  5. Training loop: CrossEntropyLoss, Adam/AdamW, small LR (e.g., 1e-3), early stopping on val set.
  6. Evaluate against TF‑IDF baseline on the same splits; track accuracy, macro‑F1, and training speed.
- Deliverables:
  - `notebooks/week4_embedding_baseline.ipynb` or `scripts/embedding_baseline.py`.
  - Saved model checkpoint (best val) and a short write-up comparing results and runtime.

---

## Experiments & Ablations (keep them small and focused)
- Preprocessing ablations (Week 3): stemming vs lemmatization; lowercasing only vs lowercase + punctuation removal; unigrams vs unigrams+bigrams.
- Feature/model ablations (Week 4): static vs fine‑tuned embeddings; embedding size (50 vs 100 vs 200); average-pool vs max-pool; MLP width/depth; with/without dropout and weight decay.
- Report: change only one variable per run and record hyperparameters in a simple table (CSV or small YAML). Use a fixed random seed for reproducibility.

---

## Targets (practical, dataset-dependent)
- TF‑IDF + logistic baseline: aim for a reasonable baseline (e.g., IMDB accuracy ~85–90 on CPU depending on preprocessing and hyperparams).
- Embedding-based neural baseline: expect improvement over TF‑IDF (>1–3% absolute) on many sentiment/dataset tasks; if not, inspect preprocessing/vocabulary/OOV handling.

---

## Suggested minimal file layout (add to repo)
- `notebooks/week3_tfidf_baseline.ipynb` — preprocessing + TF‑IDF baseline.
- `notebooks/week4_embedding_baseline.ipynb` — embedding model and comparisons.
- `data/README.md` (optional) — where to download datasets (IMDB, SST-2, AG News) and how to prepare them.
- `docs/week-3-4.md` — this file (updated).

---

## Quick commands / try-it (copyable)
```bash
# create virtual env and install minimal deps
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch scikit-learn pandas nltk spacy jupyter
# for spaCy English model (optional)
python -m spacy download en_core_web_sm
```

---

## Recommended resources (concise + important)
- Tokenization & preprocessing
  - NLTK: https://www.nltk.org/
  - spaCy usage: https://spacy.io/
- Classical features and scikit-learn
  - scikit-learn text features: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
  - Logistic regression / SVM examples: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
- Pretrained embeddings
  - GloVe: https://nlp.stanford.edu/projects/glove/
  - fastText: https://fasttext.cc/docs/en/english-vectors.html
- Practical tutorials
  - Hugging Face course (tokenization/preprocessing sections): https://huggingface.co/course
  - Practical PyTorch NLP tutorial: https://github.com/bentrevett/pytorch-sentiment-analysis

---

## Notes and style alignment
- This document follows the concise actionable format used in `docs/month-1.md` and `docs/plan.md` (objectives, tasks, deliverables, resources).
- If you want, I can also create the starter notebooks (`notebooks/week3_tfidf_baseline.ipynb` and `notebooks/week4_embedding_baseline.ipynb`) with a small runnable skeleton and tests.

---

Last updated: 2025-08-17
