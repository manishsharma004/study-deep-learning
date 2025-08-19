# Small PyTorch Models for Text

1) Average‑pooled embedding model
- Embed tokens -> mean over non-pad tokens -> dropout -> linear -> `softmax`.
- Fast to train and strong baseline.

2) 1D‑CNN
- Multiple kernel sizes (e.g., 3,4,5), apply conv -> ReLU -> max-pool -> concatenate -> dropout -> linear.
- Good for local phrase patterns.

Training notes
- Use small batches (16–64), Adam/AdamW, and monitor validation macro‑F1.
- Use gradient clipping (e.g., 1.0) to stabilize training when necessary.

Resources
- Example CNN for text (Kim, 2014): https://arxiv.org/abs/1408.5882
