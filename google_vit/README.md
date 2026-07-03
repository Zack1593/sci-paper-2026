# google-vit — Vision Transformer image classification (JAX/Flax)

A compact, self-contained image-classification project built on Google's
**Vision Transformer (ViT)**. The model definition is a faithful port of the
official repository
([google-research/vision_transformer](https://github.com/google-research/vision_transformer),
`vit_jax/models_vit.py`) and is trained **from scratch on CIFAR-10**.

> ViT — *"An Image is Worth 16×16 Words: Transformers for Image Recognition at
> Scale"*, Dosovitskiy et al., ICLR 2021.

## How it works

ViT treats an image as a sequence of patches and feeds them to a standard
Transformer encoder:

1. **Patchify + embed** — a strided convolution splits the image into
   non-overlapping patches and linearly projects each to `hidden_size`
   (`models_vit.VisionTransformer`, the `embedding` conv).
2. **Class token + position embeddings** — a learned `[CLS]` token is prepended
   and learned positional embeddings are added (`AddPositionEmbs`).
3. **Transformer encoder** — `num_layers` pre-norm blocks, each with
   multi-head self-attention and an MLP with GELU (`Encoder1DBlock`).
4. **Head** — the final `[CLS]` representation is sent to a linear classifier.

## Layout

```
google-vit/
├── vit_jax/
│   ├── models_vit.py      # Flax ViT (faithful port of the official model)
│   ├── configs.py         # ViT-Ti/S/B + a small CIFAR config
│   ├── input_pipeline.py  # CIFAR-10 download + preprocessing ([-1, 1])
│   └── checkpoint.py      # download + load official pretrained .npz weights
├── train.py               # training loop (optax AdamW + warmup-cosine)
├── predict.py             # run a CIFAR-trained model on an image
├── predict_pretrained.py  # ImageNet classification with pretrained ViT-B_16
└── requirements.txt
```

## Setup

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows
pip install -r requirements.txt
```

JAX is installed CPU-only here. For GPU/TPU, install the matching `jaxlib`
build per the [JAX install guide](https://docs.jax.dev/en/latest/installation.html).

## Train

```bash
# Full CIFAR-10 from scratch with the small CPU-friendly model
python train.py --model cifar_tiny --epochs 30 --batch-size 128 --lr 1e-3

# Quick smoke test (subsampled, ~10s on CPU)
python train.py --model cifar_tiny --epochs 1 \
    --train-size 2048 --test-size 1000 --batch-size 128
```

Key flags: `--model {cifar_tiny,ti16,s16,b16}`, `--epochs`, `--batch-size`,
`--lr`, `--weight-decay`, `--warmup-steps`, `--train-size`/`--test-size`
(subsample), `--output` (checkpoint path). Parameters are saved as a Flax
msgpack file (default `checkpoints/vit_cifar10.msgpack`).

## Predict

```bash
python predict.py --image path/to/image.png \
    --model cifar_tiny --checkpoint checkpoints/vit_cifar10.msgpack
```

The image is resized to 32×32 and the top-k CIFAR-10 classes are printed.

## Pretrained ViT-B_16 weights

The official checkpoints from `gs://vit_models` load directly into this model.
`vit_jax/checkpoint.py` downloads a `.npz`, remaps the (older-Flax) parameter
names onto this project's tree, interpolates the position embeddings to the
target resolution, and re-initialises the head when the class count differs.

**ImageNet inference** with the 384px ImageNet-2012 model (downloads ~350 MB on
first run, then cached under `pretrained/`):

```bash
python predict_pretrained.py --image path/to/photo.jpg
```

```
Top-5 predictions for SIG_Pro_by_Augustas_Didzgalvis.jpg:
  1. revolver                     97.25%
  2. holster                       1.14%
  3. rifle                         0.63%
  4. assault rifle                 0.24%
  5. projectile                    0.09%
```

**Fine-tuning on CIFAR-10** from the ImageNet-21k checkpoint (drops the 21 843-way
head, resizes position embeddings to the chosen resolution):

```bash
python train.py --model b16 --init-checkpoint imagenet21k/ViT-B_16.npz \
    --resolution 224 --epochs 5 --batch-size 64 --lr 1e-3 --warmup-steps 200
```

`--init-checkpoint` accepts either a name under `gs://vit_models/` (auto
downloaded) or a local `.npz` path. Fine-tuning ViT-B is GPU territory — on CPU
it runs but is slow; use a small `--resolution` and `--train-size` to try it.

## Model configs

| name         | patch | hidden | layers | heads | mlp  | params |
|--------------|:-----:|:------:|:------:|:-----:|:----:|:------:|
| `cifar_tiny` | 4×4   | 128    | 6      | 4     | 256  | ~0.8M  |
| `ti16`       | 16×16 | 192    | 12     | 3     | 768  | ~5.5M  |
| `s16`        | 16×16 | 384    | 12     | 6     | 1536 | ~22M   |
| `b16`        | 16×16 | 768    | 12     | 12    | 3072 | ~86M   |

`cifar_tiny` uses 4×4 patches because a 16×16 patch on a 32×32 image leaves only
4 tokens. The 16×16 variants match the names in the ViT paper and are best used
with the official pretrained checkpoints (see Notes) rather than trained from
scratch on CIFAR-10.

## Notes

- **Training from scratch vs. pretraining.** ViT's headline accuracy comes from
  large-scale pretraining (ImageNet-21k / JFT) followed by fine-tuning. Trained
  from scratch on CIFAR-10 the small model here learns a reasonable classifier
  but will not approach pretrained numbers — load the official `ViT-B_16` weights
  (see *Pretrained ViT-B_16 weights* above) for real accuracy.
- **Flax parameter-name remap.** The official checkpoints were written by an
  older Flax whose `@nn.compact` auto-naming used a single shared counter
  (`LayerNorm_0`, `MultiHeadDotProductAttention_1`, `LayerNorm_2`, `MlpBlock_3`).
  Modern Flax uses a per-class counter (`LayerNorm_0/_1`,
  `MultiHeadDotProductAttention_0`, `MlpBlock_0`), so `checkpoint.py` renames
  those three submodules on load. All array shapes are identical.
- **CIFAR-10 download.** The canonical host (`cs.toronto.edu`) currently serves
  an expired TLS certificate. `input_pipeline.py` first attempts a verified
  download and only falls back to an unverified TLS context if verification
  fails, always validating the archive against the official SHA-256 so integrity
  is preserved.
- **CPU performance.** Everything runs on CPU. The `cifar_tiny` model trains at
  ~1s/epoch on a 2k subset; a full 30-epoch run on all 50k images is feasible on
  CPU but slow — use a GPU `jaxlib` build for real training.