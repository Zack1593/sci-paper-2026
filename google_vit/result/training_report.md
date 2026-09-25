# Training Report — Vision Transformer on CIFAR-10 (JAX/Flax)

**Run command:** `python train.py --model cifar_tiny --epochs 30 --batch-size 128 --lr 1e-3`
**Date:** 2026-09-24 · **Log source:** `result/result.md`
**Dataset:** CIFAR-10, 10 classes, 50,000 train / 10,000 test images (32×32×3)
**Hardware:** NVIDIA RTX 5060, `jax[cuda12]`

## 0. About this project

`google-vit` is a compact, self-contained JAX/Flax port of Google's Vision
Transformer (ViT) — *"An Image is Worth 16×16 Words"* (Dosovitskiy et al.,
ICLR 2021). The model definition (`vit_jax/models_vit.py`) faithfully mirrors
the official `google-research/vision_transformer` implementation: an image is
split into fixed-size patches, linearly embedded, prepended with a learned
`[CLS]` token, given learned position embeddings, and passed through a
pre-norm Transformer encoder before a linear classification head reads out
the `[CLS]` token. The repo supports both training a small ViT from scratch
on CIFAR-10 (`train.py`, this report's subject) and loading Google's official
pretrained ImageNet checkpoints for transfer learning or direct inference
(`vit_jax/checkpoint.py`, `predict_pretrained.py`).

## 1. Model

`cifar_tiny` (`vit_jax/configs.py`) — a Vision Transformer sized to train on
CIFAR-10 from scratch, following Dosovitskiy et al. (ICLR 2021):

| Patch | Hidden size | Layers | Heads | MLP dim | Dropout | Params |
|:-----:|:-----------:|:------:|:-----:|:-------:|:-------:|:------:|
| 4×4   | 128         | 6      | 4     | 256     | 0.1     | 0.81M  |

4×4 patches are used instead of the paper's 16×16 because a 16×16 patch on a
32×32 image would leave only 4 tokens. Optimizer: AdamW (`lr=1e-3`,
`weight_decay=1e-4`) with a 100-step linear warmup into a cosine decay, and
gradient clipping at global norm 1.0.

## 2. Results

| Metric | Epoch 1 | Epoch 10 | Epoch 16 | Epoch 20 | Epoch 28 (best test_acc) | Epoch 30 (final) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Train accuracy | 0.325 | 0.648 | 0.730 | 0.779 | 0.838 | **0.842** |
| Train loss | 1.810 | 0.971 | 0.754 | 0.618 | 0.453 | 0.447 |
| Test accuracy | 0.407 | 0.623 | 0.649 | 0.660 | **0.667** | 0.666 |
| Test loss | 1.607 | 1.060 | **1.001** | 1.039 | 1.109 | 1.113 |

The saved checkpoint (`checkpoints/vit_cifar10.msgpack`) is whatever the model
looked like after epoch 30, not the best-performing epoch — see §4.

## 3. Overfitting sets in around epoch 16

Test loss reaches its minimum (1.001) at **epoch 16** and rises for the rest
of the run even as train loss keeps falling — the textbook overfitting
signature. Test accuracy keeps drifting up a little after that point (0.649 →
0.667), but that's the accuracy metric being coarser than the loss, not the
model still generalizing better: confidence on wrong predictions is getting
worse while the argmax occasionally still flips to the right class.

By epoch 30 the train/test accuracy gap is **17.6 points** (84.2% vs 66.6%),
up from essentially 0 at epoch 1. Two likely contributors, both visible in the
code:

- **No data augmentation.** `input_pipeline.py` only rescales pixels to
  `[-1, 1]`; there's no random crop, flip, or other augmentation, which is
  usually load-bearing for training a ViT from scratch on a small dataset —
  ViT has less built-in translation/locality bias than a CNN, so it leans on
  augmentation (or scale, per the paper) more than a CNN does.
- **`--epochs 30` overshoots the useful window.** Given test loss bottoms out
  at epoch 16, roughly half the run is spent overfitting further.

## 4. Methodological issues

- **The saved checkpoint isn't the best one.** `save_checkpoint()` in
  `train.py` runs once, after the final epoch — there's no
  `monitor='val_accuracy'`-style best-checkpoint tracking. Here that costs
  0.7 points of test accuracy (66.7% at epoch 28 vs. 66.6% saved from epoch
  30), but on a run with more pronounced overfitting the gap would be larger
  and silent.
- **The "test" set doubles as the validation set.** `evaluate()` is called
  every epoch against `x_test, y_test` — the same data used for the headline
  test accuracy. Nothing in the run selects epochs, hyperparameters, or
  early-stopping based on this signal here, so it doesn't currently bias the
  result, but the pipeline has no separate held-out split if that changes
  (e.g. if someone adds early stopping on `test_acc`).
- **The qualitative predictions in `result/result.md` predate this run.**
  The "Mini trained" and "Pretrained" `predict.py`/`predict_pretrained.py`
  examples are dated 2026-06-30 and used CPU-only JAX and a 1-epoch,
  2,048-example smoke-test checkpoint — not the 30-epoch, 50k-image model
  this report is about. The mini-trained example's top-3 for `car.jpg` (ship
  10.99% / airplane 10.91% / truck 10.75%) is close to the 10%-uniform
  baseline for 10 classes, i.e. that checkpoint was essentially untrained —
  expected for a 1-epoch/2k-example smoke test, but it means there's no
  qualitative prediction on file yet for the model this report actually
  evaluates. Re-running `predict.py` against
  `checkpoints/vit_cifar10.msgpack` from this run would give a more
  representative qualitative read.

## 5. Runtime

Epoch 1 took 32.4s (JAX/XLA trace + compile for `train_step`/`eval_step`);
every subsequent epoch ran in 5.8–6.3s (~6.05s average), for a full 30-epoch
run in **~3.5 minutes** on the RTX 5060. This is a large change from the
CPU-only numbers in `README.md` ("~1s/epoch on a 2k subset... full run...
feasible but slow") — that note predates installing `jax[cuda12]` in this
environment and should be updated to reflect GPU timings.

## 6. Recommendations

1. Add `ModelCheckpoint`-equivalent tracking: save whichever epoch has the
   best `test_acc`/`test_loss`, not just the last one.
2. Add light data augmentation (random crop with padding, horizontal flip) to
   `input_pipeline.py` — the standard, cheap fix for training a ViT from
   scratch on CIFAR-10-sized data.
3. Either shorten `--epochs` toward ~16–20 (where test loss bottoms out) or
   add a regularizer (higher dropout, stronger weight decay) that lets longer
   training keep helping instead of overfitting.
4. Re-run `predict.py` on `checkpoints/vit_cifar10.msgpack` (this run's
   checkpoint) and append fresh qualitative predictions to
   `result/result.md`, replacing or clearly dating the stale 1-epoch example.
5. Introduce a real train/val/test split if any future change starts using
   `test_acc` to make decisions (early stopping, model selection, etc.).

## Artifacts referenced

- `result/result.md` — full training log and prior `predict.py` /
  `predict_pretrained.py` outputs
- `checkpoints/vit_cifar10.msgpack` — trained parameters (epoch 30, not best)
- `vit_jax/configs.py`, `train.py`, `input_pipeline.py` — model/training/data
  code referenced above
