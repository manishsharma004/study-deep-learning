# Dataset, Collate Function, and DataLoader

Overview
- For variable-length text, implement a `collate_fn` to pad sequences and return masks or lengths.
- Use PyTorch `Dataset` to yield token index sequences and labels; `DataLoader` with `collate_fn` batches them.

Implementation notes
- Pad token sequences to the batch max length with a PAD token id (e.g., 0).
- Return attention masks or lengths for models that need them.
- For reproducible experiments, set `worker_init_fn` and `torch.manual_seed` in DataLoader loops.

Resources
- PyTorch DataLoader docs: https://pytorch.org/docs/stable/data.html
