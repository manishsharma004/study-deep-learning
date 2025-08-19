# Pretrained Embeddings

Overview
- Pretrained word vectors (GloVe, fastText) provide dense representations for words trained on large corpora.
- fastText includes subword information which helps with OOV words.

How to use
1. Download pretrained vectors (GloVe or fastText).
2. Build a vocabulary from your dataset and map tokens to indices.
3. Initialize an embedding matrix using pretrained vectors where available; initialize OOV tokens randomly (normal distribution) or with zeros.
4. In PyTorch, use `nn.Embedding.from_pretrained` to load the matrix and optionally set `freeze=False` to fine‑tune.

Resources
- GloVe: https://nlp.stanford.edu/projects/glove/
- fastText: https://fasttext.cc/
- Gensim for loading: https://radimrehurek.com/gensim/
