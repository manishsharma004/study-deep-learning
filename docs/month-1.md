# Month 1: Deep Learning Essentials and NLP Fundamentals

Time commitment: 5–8 hours/week

Outcome by end of Month 1:
- Solid grasp of neural networks, backpropagation, optimizers, activations, losses, and regularization.
- Practical experience training an MLP classifier in PyTorch with experiments and tracking.
- Working knowledge of text preprocessing, tokenization, TF‑IDF, and pretrained word embeddings.
- A baseline NLP classifier built with classical features and a simple neural model using embeddings.

Prerequisites and setup:
- Python 3.10+, PyTorch (CPU or CUDA), Jupyter/VS Code, NumPy, scikit‑learn, matplotlib/seaborn, NLTK, spaCy.
- Optional: Weights & Biases or MLflow for experiment tracking.

Suggested weekly cadence (adapt as needed):
- Day 1: Reading/videos and notes.
- Day 2–3: Core implementation.
- Day 4: Experiments and evaluation.
- Day 5: Reflection, write-up, and backlog cleanup.

---

## Week 1–2: Deep Learning Essentials and Neural Networks

Objectives:
- Review perceptrons, MLPs, forward/backward pass, initialization, and vanishing/exploding gradients.
- Understand gradient descent variants (SGD, momentum, Adam, AdamW) and learning rate schedules.
- Apply regularization (weight decay, dropout, early stopping) and normalization (batch norm, layer norm where applicable).

Study material:
- DeepLearning.AI Neural Networks and Deep Learning (overview of backprop and optimization).
- PyTorch docs: nn.Module, autograd, optim (SGD, Adam/AdamW), lr_scheduler (StepLR, CosineAnnealingLR, OneCycleLR).
- Recommended blog: “A Recipe for Training Neural Networks” by Andrej Karpathy (for practical heuristics).

Open-source GitHub resources (Week 1–2):
- PyTorch Examples: https://github.com/pytorch/examples (MNIST, word language model, etc.)
- PyTorch Tutorials (code): https://github.com/pytorch/tutorials
- Beginner-friendly PyTorch tutorial: https://github.com/yunjey/pytorch-tutorial
- Tiny autograd engine + MLP from scratch: https://github.com/karpathy/micrograd
- Awesome PyTorch (curated list): https://github.com/bharathgs/Awesome-pytorch-list
- Dive into Deep Learning (interactive book + notebooks): https://github.com/d2l-ai/d2l-en
- Karpathy’s NN Zero to Hero (code): https://github.com/karpathy/nn-zero-to-hero
- fastai practical deep learning (book + code): https://github.com/fastai/fastbook
- PyTorch Lightning (training loops, logging, checkpoints): https://github.com/Lightning-AI/pytorch-lightning

Hands-on project: MLP classifier
- Dataset: MNIST or Fashion‑MNIST (or a small tabular dataset like UCI Adult for variety).
- Baseline: Single hidden‑layer MLP with ReLU, CrossEntropyLoss, SGD.
- Training loop: Manual loop using nn.Module, DataLoader, zero_grad/backward/step, and evaluation on a validation split.
- Experiments (change one variable at a time):
  - Activations: ReLU vs Tanh vs GELU.
  - Optimizers: SGD (with/without momentum) vs Adam vs AdamW.
  - LR schedules: none vs StepLR vs CosineAnnealingLR vs OneCycleLR.
  - Regularization: weight decay grid (e.g., 0, 1e‑4, 1e‑3), dropout (p=0.2–0.5), early stopping.
  - Initialization: default vs Xavier/He.
- Tracking: Record train/val loss and accuracy each epoch; save best model checkpoint; log key hyperparameters and results.
- Targets: MNIST accuracy ≥ 97% or Fashion‑MNIST ≥ 88% within 10–30 epochs on CPU/GPU.

Deliverables:
- Reproducible training script or notebook, saved metrics/plots, and a brief summary (what worked, what did not, next steps).

Stretch goals:
- Implement the backward pass for a 2‑layer MLP in NumPy to verify gradients against PyTorch autograd.
- Try mixed‑precision (if using GPU) and measure speed/accuracy trade‑offs.
- Add unit tests for data pipeline and loss/accuracy computation.

---

## Week 3–4: NLP Fundamentals and Text Representations

Objectives:
- Learn text normalization, tokenization, stopword handling, stemming vs lemmatization, n‑grams.
- Build classical representations: bag‑of‑words and TF‑IDF, and train a simple baseline classifier.
- Use pretrained word embeddings (GloVe or fastText) and build a simple neural text classifier in PyTorch.

Study material:
- Hugging Face NLP Course: sections on tokenization and preprocessing.
- NLTK and spaCy documentation for preprocessing pipelines.
- Background on word embeddings: Word2Vec/GloVe/fastText (conceptual understanding and practical loading).

Open-source GitHub resources (Week 3–4):
- scikit-learn examples (text features/classification): https://github.com/scikit-learn/scikit-learn/tree/main/examples
- NLTK (library + examples): https://github.com/nltk/nltk
- spaCy (library + usage examples): https://github.com/explosion/spaCy
- Hugging Face Datasets: https://github.com/huggingface/datasets
- GloVe (pretrained embeddings + scripts): https://github.com/stanfordnlp/GloVe
- fastText (embeddings + text classification): https://github.com/facebookresearch/fastText
- PyTorch text utilities (torchtext): https://github.com/pytorch/text
- Practical PyTorch NLP tutorial series: https://github.com/bentrevett/pytorch-sentiment-analysis
- Awesome NLP (curated list): https://github.com/keon/awesome-nlp
- Gensim (Word2Vec/fastText implementation): https://github.com/RaRe-Technologies/gensim
- spaCy course (hands-on exercises): https://github.com/explosion/spaCy-course
- Hugging Face Tokenizers (BPE/Unigram/WordPiece): https://github.com/huggingface/tokenizers
- nlpaug (text data augmentation): https://github.com/makcedward/nlpaug
- TextAttack (adversarial attacks and augmentation): https://github.com/QData/TextAttack

Hands-on project A: Classical baseline
- Dataset: IMDB sentiment, SST‑2, or AG News (choose one for consistency through Month 1).
- Pipeline: Clean text, tokenize, optional stemming/lemmatization, TF‑IDF features.
- Model: Logistic regression or linear SVM using scikit‑learn.
- Evaluation: Accuracy and macro‑F1 on a held‑out validation split; inspect confusion matrix and most informative features.

Hands-on project B: Embedding‑based neural baseline
- Embeddings: Load pretrained GloVe or fastText vectors; build a vocabulary with OOV handling.
- Dataset pipeline: Map tokens to indices, pad sequences, create attention masks; implement a collate function for variable lengths.
- Model: Average (or max) pooled embeddings → dropout → MLP; or a small 1D‑CNN over tokens.
- Training: CrossEntropyLoss, Adam/AdamW; track metrics and early stopping on validation.
- Ablations:
  - Static vs fine‑tuned embeddings.
  - With/without dropout and weight decay.
  - Sequence length limits (e.g., 128 vs 256 tokens) and their impact on accuracy and speed.

Targets:
- Achieve a clear lift of the embedding‑based neural model over the TF‑IDF baseline on the chosen dataset.

Deliverables:
- Two baselines (classical and neural), plots/tables comparing metrics, and a short write‑up discussing preprocessing choices and trade‑offs.

Stretch goals:
- Implement subword tokenization (e.g., unigram/BPE conceptually using a library) and compare to word‑level.
- Add simple data augmentation (e.g., synonym replacement) and measure effects.
- Package the data pipeline and models into reusable modules.

---

References and quick links:
- PyTorch: https://pytorch.org/docs/stable/nn.html, https://pytorch.org/docs/stable/autograd.html, https://pytorch.org/docs/stable/optim.html, https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html, https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html, https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
- Datasets: https://pytorch.org/vision/stable/datasets.html, https://huggingface.co/datasets
- NLP tooling: https://www.nltk.org/, https://spacy.io/
- Embeddings: https://nlp.stanford.edu/projects/glove/, https://fasttext.cc/
- Baselines and metrics: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

Open-source GitHub resources (general):
- PyTorch core repos: https://github.com/pytorch/pytorch, https://github.com/pytorch/vision, https://github.com/pytorch/text, https://github.com/pytorch/examples, https://github.com/pytorch/tutorials
- NLP stacks: https://github.com/nltk/nltk, https://github.com/explosion/spaCy, https://github.com/huggingface/datasets
- Embeddings: https://github.com/stanfordnlp/GloVe, https://github.com/facebookresearch/fastText
- Curated lists: https://github.com/bharathgs/Awesome-pytorch-list, https://github.com/keon/awesome-nlp, https://github.com/huggingface/course
- Weights & Biases examples: https://github.com/wandb/examples
- MLflow examples: https://github.com/mlflow/mlflow/tree/master/examples
- TorchData (data pipelining): https://github.com/pytorch/data
- TorchMetrics (model metrics): https://github.com/Lightning-AI/torchmetrics

Checklist:
- MLP script/notebook with experiments and results archived.
- Classical TF‑IDF baseline with metrics.
- Embedding‑based neural baseline with metrics.
- One‑page reflection on findings and open questions to carry into Month 2 (Transformers).
