# Bag-of-Words and TF‑IDF

Overview
- CountVectorizer (BoW) and `TfidfVectorizer` convert tokenized text into sparse numeric features suitable for classical ML algorithms.

Practical tips
- Use `max_features` or `min_df` to control vocabulary size.
- Try unigram and unigram+bigram representations; bigrams add local context at the expense of higher dimensionality.
- Normalize features (L2) before feeding into linear models when appropriate.

Resources
- `scikit-learn` documentation: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
