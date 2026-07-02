# Study Notebook

**Author:** Haozhe (Jimmy) Jia, Boston University

A collection of deep-learning study projects I built during my time in the
[Kolachalama Lab](https://vkola-lab.github.io/) at Boston University. Each
subproject reimplements a foundational architecture from scratch in PyTorch,
with notebooks that walk through every building block, verify tensor shapes at
each step, and visualize intermediate results. This repository doubles as my
portfolio of hands-on model implementations.

## Projects

### 1. [jimmy-gpt2](jimmy-gpt2/): GPT-2 from scratch

A from-scratch reimplementation of the GPT-2 transformer, weight-compatible
with HuggingFace's pretrained checkpoints.

- Core blocks built bottom-up: causal self-attention, GELU MLP, pre-LayerNorm
  transformer block, and the full GPT model (token and positional embeddings,
  block stack, LM head)
- Loads OpenAI's pretrained weights (`gpt2` through `gpt2-xl`) into the
  from-scratch model and generates text with top-k sampling
- Training forward pass with cross-entropy loss on Tiny Shakespeare, tokenized
  with `tiktoken`'s GPT-2 BPE encoder
- Walkthrough notebooks with per-module shape checks and pretrained-model
  experiments, including a look at how the token embedding shares weights with
  the LM head

See the project's own [README](jimmy-gpt2/README.md) for architecture details
and usage.

### 2. [ViT](ViT/): Vision Transformer from scratch (in progress)

Study notes on the Vision Transformer ([Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)),
built up module by module in [`ViT_notes.ipynb`](ViT/ViT_notes.ipynb).

- **Patch embedding**: implemented as a strided `Conv2d` that maps an image
  `(B, C, H, W)` to a token sequence `(B, num_patches, d_model)`, with
  visualizations of the image cut into patches and a PCA-to-RGB view of the
  resulting embeddings
- **Positional embedding**: a learnable `[CLS]` token plus sinusoidal position
  encodings (currently being implemented)
- Planned: multi-head self-attention, the transformer encoder, and a
  classification head to complete the model

## Repository structure

```
.
├── requirements.txt
├── jimmy-gpt2/                 # GPT-2 reimplementation (see its README)
│   ├── train_gpt2.py           # Full model, pretrained-weight loading, generation
│   ├── datasets/input.txt      # Tiny Shakespeare corpus
│   └── notebooks/              # Step-by-step walkthrough notebooks
└── ViT/
    ├── ViT_notes.ipynb         # ViT built module by module, with visualizations
    └── *.jpeg                  # Sample image for the patch/embedding demos
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open any notebook with `jupyter notebook` and run it top to bottom.
