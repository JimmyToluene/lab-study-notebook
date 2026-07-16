# Study Notebook

**Author:** Haozhe (Jimmy) Jia, Boston University

A collection of deep-learning study projects I built during my time in the
[Kolachalama Lab](https://vkola-lab.github.io/) at Boston University.
The subprojects focus on large-scale Transformer-based models in the NLP and
Computer Vision fields.

## Contents

- [Projects](#projects)
  - [1. jimmy-gpt2: GPT-2 from scratch](#1-jimmy-gpt2-gpt-2-from-scratch) (NLP)
  - [2. ViTransformer: Vision Transformer from scratch](#2-vitransformer-vision-transformer-from-scratch) (Computer Vision)
- [Datasets](#datasets)
- [Repository structure](#repository-structure)
- [Setup](#setup)

## Projects
### NLP Side:
#### 1. [jimmy-gpt2](Jimmy-gpt2/): GPT-2 from scratch

A from-scratch reimplementation of the GPT-2 transformer, weight-compatible
with HuggingFace's pretrained checkpoints, following Andrej Karpathy's
["Let's reproduce GPT-2 (124M)"](https://www.youtube.com/watch?v=l8pRSuU81PU)
video.

<p align="center">
  <img src="Jimmy-gpt2/assets/GPT-2.webp" alt="GPT-2 architecture" width="720">
</p>
<p align="center">
  <em>The GPT-2 architecture and next-token prediction flow. Figure from
  <a href="https://medium.com/@vipul.koti333/from-theory-to-code-step-by-step-implementation-and-code-breakdown-of-gpt-2-model-7bde8d5cecda">Vipul Koti's article</a>.</em>
</p>

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

See the project's own [README](Jimmy-gpt2/README.md) for architecture details
and usage.

### Computer Vision Side:
#### 2. [ViTransformer](ViTransformer/): Vision Transformer from scratch

Study notes on the Vision Transformer ([Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)),
built up module by module in [`ViT_notes.ipynb`](ViTransformer/notebooks/ViT_notes.ipynb)
and trained end to end on MNIST. The building blocks and the full model are
extracted into an importable [`vit/`](ViTransformer/vit/) package (patch
embedding, positional encoding, transformer encoder, and the assembled
`ViTBackbone` / `ViTClassifier`), so training scripts can simply do
`from vit import ViTBackbone, ViTClassifier`.

Everything *around* the model — YAML config parsing, dataset registry and
transforms, train/eval loops, checkpointing, Mixup/CutMix, optimizers and
schedulers, W&B logging — lives in [`engine/`](ViTransformer/engine/), so the
`vit/` package stays a clean architecture reference. Training is driven by
configs in [`configs/`](ViTransformer/configs/) rather than edited constants:

```bash
python train.py --config configs/mnist-baseline.yaml
python train.py --config configs/tiny-imagenet-baseline.yaml --wandb
python test.py  --ckpt runs/mnist-baseline/best.pt      # test-split accuracy
python infer.py --ckpt runs/mnist-baseline/best.pt a.png b.png --topk 3
```

<p align="center">
  <img src="assets/vit_figure.png" alt="ViT architecture" width="720">
</p>
<p align="center">
  <em>The ViT architecture (Figure 1 of Dosovitskiy et al., 2020): an image is
  split into fixed-size patches, linearly embedded, combined with position
  embeddings and a learnable [class] token, fed through a standard transformer
  encoder, and classified from the [class] token by an MLP head.</em>
</p>

**Patch embedding**\
A strided `Conv2d` that maps an image `(B, C, H, W)` to a token sequence
`(B, num_patches, d_model)`, with visualizations of the image cut into patches
and a PCA-to-RGB view of the resulting embeddings.

**Positional encoding**\
A learnable `[CLS]` token prepended to the sequence, plus a **learnable**
position embedding — a `nn.Parameter` of shape `(1, n_patches + 1, d_model)`
added to the tokens, which is what ViT itself uses. Without it self-attention is
permutation-invariant and the model cannot tell one patch position from another.
The `use_pe` flag turns it off to make that ablation easy to run.

The sinusoidal alternative from "Attention Is All You Need" is kept in the
module as a commented-out reference:

$$
PE_{(pos,\,2k)} = \sin\!\left(\frac{pos}{10000^{2k/d_{model}}}\right), \qquad
PE_{(pos,\,2k+1)} = \cos\!\left(\frac{pos}{10000^{2k/d_{model}}}\right)
$$

Each position $pos$ gets a fixed vector of interleaved sines and cosines at
geometrically decreasing frequencies, added to the token embedding at the
input: $\text{x'}_{pos} = x_{pos} + PE_{pos}$.

**Transformer encoder**\
Multi-head self-attention (batched QKV projection,
`scaled_dot_product_attention` without a causal mask — the one line that
separates this from the causal attention in jimmy-gpt2), a GELU feed-forward
network with an `r_ffn` expansion ratio, and pre-LayerNorm residual connections,
stacked into encoder blocks. Each residual branch is wrapped in `DropPath`
(stochastic depth), whose rate scales linearly from 0 at the first block to
`drop_path` at the last, following the DeiT/timm schedule.

**Classification**\
`ViTBackbone` runs the stack and returns the final-LayerNorm'd `[CLS]` token;
`ViTClassifier` puts a single `nn.Linear` head on top, returning raw logits for
`CrossEntropyLoss` (no `Softmax` in the model — the loss applies its own).
[`simple_train_test.py`](ViTransformer/notebooks/simple_train_test.py) is the
minimal, dependency-free version of that loop: it trains on MNIST and reports
test accuracy every epoch. The small baseline in
[`configs/mnist-baseline.yaml`](ViTransformer/configs/mnist-baseline.yaml)
(`d_model=9`, 3 layers, 3 heads, 5 epochs) reaches **92% test accuracy**.

**Training recipe**\
[`configs/tiny-imagenet-baseline.yaml`](ViTransformer/configs/tiny-imagenet-baseline.yaml)
is the full DeiT-style recipe on a harder dataset: DeiT-Tiny width (`d_model=192`,
6 layers), AdamW with decoupled weight decay, cosine LR with warmup, RandAugment,
random erasing, Mixup/CutMix with soft-target cross-entropy, label smoothing,
stochastic depth, gradient clipping, and early stopping on a held-out validation
split carved from `train/` (the test split is never touched until `test.py`).

## Datasets

**[Tiny Shakespeare](https://github.com/karpathy/char-rnn)** (jimmy-gpt2)\
A ~1 MB plain-text corpus of Shakespeare's plays, popularized by Andrej
Karpathy's char-rnn. Included in the repo at `jimmy-gpt2/datasets/input.txt`
and used for the GPT-2 training forward pass.

**[MNIST](https://en.wikipedia.org/wiki/MNIST_database)** (ViTransformer)\
The classic handwritten-digit dataset (60k train / 10k test, 28x28 grayscale)
created by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges, and
introduced in ["Gradient-Based Learning Applied to Document Recognition"](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
(LeCun et al., Proc. IEEE 1998), the LeNet-5 paper. Downloaded automatically
by `torchvision` into an untracked `datasets/` directory the first time the
ViT notebook or training script runs, and used to train the ViT classifier.

**[Tiny ImageNet](https://www.kaggle.com/c/tiny-imagenet)** (ViTransformer)\
A 200-class subset of ImageNet (100k train / 10k val, 64x64 RGB) released for
Stanford's CS231n. Used as the harder benchmark for the full DeiT-style recipe.
Run [`prepare_tiny_imagenet.py`](ViTransformer/prepare_tiny_imagenet.py) once to
download it and reorganize the official `val/` split into the per-class
subdirectories `ImageFolder` expects; the official `val/` then serves as the
*test* split, while validation is carved out of `train/`.

## Repository structure

```
.
├── requirements.txt
├── assets/                     # Shared static images
│   ├── vit_figure.png          # Architecture figure from the ViT paper
│   └── ave-mujica-resized.jpeg # Sample image for the patch/embedding demos
├── Jimmy-gpt2/                 # GPT-2 reimplementation (see its README)
│   ├── train_gpt2.py           # Full model, pretrained-weight loading, generation
│   ├── assets/GPT-2.webp       # Architecture figure (credit: Vipul Koti, Medium)
│   ├── datasets/input.txt      # Tiny Shakespeare corpus
│   └── notebooks/              # Step-by-step walkthrough notebooks
└── ViTransformer/
    ├── train.py                # Config-driven training (val monitoring, best/last ckpt)
    ├── test.py                 # Test-split accuracy from a checkpoint
    ├── infer.py                # Classify image files with a checkpoint
    ├── prepare_tiny_imagenet.py  # Download + reorganize Tiny ImageNet for ImageFolder
    ├── configs/                # YAML recipes (the only place hyperparameters live)
    │   ├── mnist-baseline.yaml
    │   └── tiny-imagenet-baseline.yaml
    ├── vit/                    # Importable package: the architecture, nothing else
    │   ├── __init__.py         # Re-exports ViTBackbone / ViTClassifier
    │   ├── ViT.py              # ViTBackbone (embeddings + encoder stack) + ViTClassifier
    │   ├── patch_embed.py      # PatchEmbedding (strided-Conv2d patchifier)
    │   ├── pos_encoding.py     # [CLS] token + learnable positional embedding
    │   └── transformer.py      # Multi-head self-attention, GELU FFN, DropPath, encoder block
    ├── engine/                 # Everything around the model (training plumbing)
    │   ├── config.py           # Typed Config dataclasses loaded from YAML
    │   ├── data.py             # Dataset registry, train/eval transforms, dataloaders
    │   ├── model.py            # Config + dataset spec -> a wired ViTClassifier
    │   ├── loops.py            # train_one_epoch / evaluate
    │   ├── checkpoint.py       # save_checkpoint / load_model (config travels with weights)
    │   ├── mixup.py            # Mixup/CutMix + SoftTargetCrossEntropy
    │   ├── optim.py            # AdamW + cosine/warmup schedulers
    │   ├── tracking.py         # W&B wrapper that no-ops when disabled
    │   └── utils.py            # Seeding, device selection
    └── notebooks/
        ├── ViT_notes.ipynb     # ViT built module by module, trained on MNIST
        └── simple_train_test.py  # Minimal MNIST training loop against vit/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open any notebook with `jupyter notebook` and run it top to bottom.

To train the ViT instead, run from inside `ViTransformer/` (the scripts import
`vit/` and `engine/` as top-level packages):

```bash
cd ViTransformer
python train.py --config configs/mnist-baseline.yaml
```

MNIST downloads itself on first run. Tiny ImageNet needs
`python prepare_tiny_imagenet.py` first.
