# Tokenization

What it is
- Tokenization splits raw text into tokens (words, punctuation, or subword units).

Common approaches
- Whitespace/token split (fast, brittle).
- Rule-based/tokenizer libraries: NLTK tokenizers, spaCy tokenizer.
- Subword tokenizers (BPE, WordPiece, Unigram) used for modern transformers — see `docs/topics/embeddings.md`.

Practical notes
- For classical baselines, word-level tokenization is usually sufficient.
- For pretrained Transformer models, use the tokenizer provided by the model (Hugging Face Tokenizers).

Resources
- Hugging Face Tokenizers: https://github.com/huggingface/tokenizers
- spaCy tokenization: https://spacy.io/usage/spacy-101#tokenization
