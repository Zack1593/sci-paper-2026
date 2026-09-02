# Training Report — Traffic Sign Recognition CNN

**Run command:** `python main.py --training --data_dir data --model_dir models --epochs 20 --batch_size 64 --learning_rate 0.001`
**Date:** 2026-09-02 · **Log source:** `outputs/terminal_out.txt`
**Dataset:** GTSRB (German Traffic Sign Recognition Benchmark), 43 classes

## 1. Model

A small sequential CNN (`scripts/model_training.py`):

| Block | Layers |
|---|---|
| 1 | Conv2D(32, 5×5) → Conv2D(32, 5×5) → MaxPool(2×2) → Dropout(0.25) |
| 2 | Conv2D(64, 3×3) → Conv2D(64, 3×3) → MaxPool(2×2) → Dropout(0.25) |
| Head | Flatten → Dense(256) → Dropout(0.5) → Dense(43, softmax) |

Compiled with `categorical_crossentropy` / Adam (default learning rate), trained for 20 epochs, batch size 64, on an RTX 5060.

## 2. Results

| Metric | Epoch 1 | Epoch 10 | Epoch 20 (final) |
|---|---|---|---|
| Train accuracy | 0.445 | 0.908 | 0.941 |
| Train loss | 2.249 | 0.298 | 0.206 |
| Val accuracy | 0.809 | 0.961 | **0.980** |
| Val loss | 0.721 | 0.121 | **0.069** |

Both val accuracy and val loss reach their best values of the run at epoch 20 (0.9804 / 0.0693), with the curves still improving slightly rather than degrading — see `outputs/evaluation_plots.png`. Validation accuracy plateaus around epoch 11–14 (~0.97) with minor fluctuation before ticking up again toward epoch 20; there's no visible overfitting (val loss keeps trending down alongside train loss), so a longer run would likely have squeezed out marginal further gains rather than regressing.

One notable pattern: **validation accuracy is consistently higher than training accuracy throughout the run** (e.g. epoch 20: 0.941 train vs 0.980 val). This is expected here specifically because Dropout (0.25/0.25/0.5) is active during training but disabled during validation — it's a normal artifact of this architecture, not a sign of a data problem.

## 3. Runtime — Blackwell PTX JIT concern didn't materialize

Earlier testing on this RTX 5060 (compute capability 12.0a, Blackwell) surfaced this warning:
```
TensorFlow was not built with CUDA kernel binaries compatible with compute capability 12.0a.
CUDA kernels will be jit-compiled from PTX, which could take 30 minutes or longer.
```
In practice, epoch 1 (which absorbs the JIT/XLA warm-up cost) took only **14s** total (491 steps at ~15ms/step, vs ~3ms/step for the model's actual per-step compute) — not the 30-minute worst case. Every subsequent epoch ran in 1–2s. For this specific model (small, few distinct kernel shapes), the PTX JIT fallback is a minor one-time cost, not a practical blocker. This may not hold for larger/more architecturally diverse models, where more distinct kernels would each pay the JIT tax.

## 4. Dataset — class imbalance not accounted for

`outputs/class_distribution.png` shows substantial imbalance across the 43 classes: several classes have ~1,700–1,800 training samples (e.g. classes 1, 2, 4, 5, 10, 12, 13, 38) while others have under 200 (e.g. classes 0, 19, 24, 27, 32, 37) — roughly a **10:1 ratio** between the largest and smallest classes. The training pipeline does not apply `class_weight` or any resampling, and uses plain accuracy as its only tracked metric. With this much imbalance, overall accuracy (94–98%) can look strong while minority classes underperform — but this can't be confirmed or refuted from the current outputs, because:

- No confusion matrix or per-class precision/recall/F1 is generated anywhere in the pipeline (`scripts/evaluation.py` only plots aggregate accuracy/loss curves).
- Test set class balance isn't reported either, so it's unknown whether test accuracy is being pulled up by over-representation of easy/majority classes.

**This is the biggest gap in evaluating whether the model is actually good**, independent of the headline 98% validation accuracy.

## 5. Methodological issues worth fixing

- **The "validation" set is the test set.** `model_training.py` calls `model.fit(..., validation_data=(X_test, y_test))` — the same `X_test`/`y_test` used later in `evaluation.py`. There is no held-out split the model/training process never sees. The reported 98.04% is therefore an optimistic estimate of generalization, since the test set was monitored (and implicitly could influence early-stopping/epoch-count decisions) during training. A proper 3-way split (train/val/test) is needed before this number can be trusted as a true holdout metric.
- **`best_model.h5` is not actually the best model** — it's just the final epoch's weights. There's no `ModelCheckpoint(save_best_only=True)`; `model.save()` runs once after all 20 epochs complete. In this particular run epoch 20 happened to have the best val accuracy/loss, so it's harmless here, but it's not guaranteed by the code and will silently save a worse model on a different run.
- **`--learning_rate` is a dead argument.** `main.py`/`model_training.py` accept `--learning_rate` (set to 0.001 in the run command) but `model.compile()` uses `optimizer='adam'` — the string form, which ignores the CLI value and always uses Keras's Adam default (also 0.001, so this run happened to match, but the flag has no effect for any other value).

## 6. Recommendations

1. Split off a real validation set separate from the test set; only evaluate on test once, at the end.
2. Add `ModelCheckpoint(save_best_only=True, monitor='val_accuracy')` and load that checkpoint in `evaluation.py` instead of trusting the final epoch.
3. Wire `args.learning_rate` into an actual `Adam(learning_rate=args.learning_rate)` instance.
4. Extend `evaluation.py` to compute a confusion matrix and per-class precision/recall (e.g. via `sklearn.metrics.classification_report`) so minority-class performance is visible, not just aggregate accuracy.
5. Consider `class_weight` in `model.fit()` given the ~10:1 class imbalance.

## Artifacts referenced

- `outputs/terminal_out.txt` — full training log
- `outputs/evaluation_plots.png` — accuracy/loss curves
- `outputs/class_distribution.png` — per-class sample counts
- `models/best_model.h5`, `models/history.npy` — trained weights and epoch history
