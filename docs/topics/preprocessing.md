# Text Preprocessing

Key steps
- Normalization: lowercase, `unicode` normalization (NFKC/NFKD optionally), remove or standardize punctuation.
- Cleaning: remove HTML tags, URLs, excessive whitespace, or other dataset-specific noise.
- `Stopwords`: remove common words when using bag-of-words or TF‑IDF if helpful.
- Stemming vs Lemmatization: Porter stemmer (NLTK) is fast; spaCy lemmatization preserves real words.

Tips
- Save raw and cleaned versions to allow easy ablations.
- Keep preprocessing minimal for neural baselines where embeddings or tokenizers handle morphology.

Libraries
- NLTK: tokenizers and stemmers.
- spaCy: fast tokenization and lemmatization (download `en_core_web_sm` or larger models as needed).
