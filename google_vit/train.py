"""Train a Vision Transformer on CIFAR-10 from scratch.

Example:
    python train.py --model cifar_tiny --epochs 10 --batch-size 128

For a quick smoke test on CPU:
    python train.py --model cifar_tiny --epochs 1 \
        --train-size 2000 --test-size 1000 --batch-size 128
"""

import argparse
import os
import time
from typing import Any, Dict

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from vit_jax import checkpoint
from vit_jax import configs
from vit_jax import input_pipeline
from vit_jax import models_vit


class TrainState(train_state.TrainState):
    pass


def maybe_resize(images: np.ndarray, resolution: int):
    """Bilinearly resizes a (N, 32, 32, 3) batch to the model resolution."""
    if resolution == input_pipeline.IMAGE_SIZE:
        return images
    return jax.image.resize(
        images, (images.shape[0], resolution, resolution, 3), method='bilinear')


def create_model(model_name: str, num_classes: int) -> models_vit.VisionTransformer:
    config = configs.get_config(model_name)
    return models_vit.VisionTransformer(
        num_classes=num_classes,
        patches=config.patches,
        transformer=config.transformer,
        hidden_size=config.hidden_size,
        representation_size=config.representation_size,
        classifier=config.classifier,
    )


def create_train_state(
        rng: Any,
        model: models_vit.VisionTransformer,
        learning_rate: float,
        total_steps: int,
        warmup_steps: int,
        weight_decay: float,
        resolution: int = input_pipeline.IMAGE_SIZE,
) -> TrainState:
    """Initializes parameters and the optimizer."""
    dummy = jnp.ones([1, resolution, resolution, 3])
    variables = model.init(rng, dummy, train=False)
    params = variables['params']

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=max(total_steps, warmup_steps + 1),
        end_value=0.0,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state: TrainState, images, labels, dropout_rng):
    """A single optimization step."""
    dropout_rng = jax.random.fold_in(dropout_rng, state.step)

    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params}, images, train=True,
            rngs={'dropout': dropout_rng})
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, labels).mean()
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return state, loss, accuracy


@jax.jit
def eval_step(state: TrainState, images, labels):
    logits = state.apply_fn({'params': state.params}, images, train=False)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).sum()
    correct = jnp.sum(jnp.argmax(logits, -1) == labels)
    return loss, correct


def evaluate(state: TrainState, x, y, batch_size: int,
             resolution: int = input_pipeline.IMAGE_SIZE) -> Dict[str, float]:
    total_loss, total_correct = 0.0, 0
    for images, labels in input_pipeline.iterate_batches(x, y, batch_size):
        loss, correct = eval_step(state, maybe_resize(images, resolution), labels)
        total_loss += float(loss)
        total_correct += int(correct)
    n = x.shape[0]
    return {'loss': total_loss / n, 'accuracy': total_correct / n}


def save_checkpoint(path: str, state: TrainState, meta: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        f.write(flax.serialization.to_bytes(state.params))
    print(f'Saved parameters to {path}  ({meta})')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='cifar_tiny',
                        choices=sorted(configs.MODEL_CONFIGS),
                        help='Model config name.')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--train-size', type=int, default=0,
                        help='Subsample the training set (0 = full 50k).')
    parser.add_argument('--test-size', type=int, default=0,
                        help='Subsample the test set (0 = full 10k).')
    parser.add_argument('--output', default='checkpoints/vit_cifar10.msgpack')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--resolution', type=int, default=input_pipeline.IMAGE_SIZE,
                        help='Resize CIFAR images to this resolution before the '
                             'model (e.g. 224 for pretrained ViT-B_16).')
    parser.add_argument('--init-checkpoint', default=None,
                        help='Official checkpoint to fine-tune from, e.g. '
                             '"imagenet21k/ViT-B_16.npz" (downloaded from '
                             'gs://vit_models) or a local .npz path. The head is '
                             're-initialised for CIFAR-10 and position '
                             'embeddings are resized as needed.')
    parser.add_argument('--cache-dir', default='pretrained',
                        help='Where to cache downloaded checkpoints.')
    args = parser.parse_args()

    print(f'JAX devices: {jax.devices()}')
    print('Loading CIFAR-10 ...')
    x_train, y_train, x_test, y_test = input_pipeline.load_cifar10(
        train_size=args.train_size, test_size=args.test_size)
    print(f'  train={x_train.shape[0]}  test={x_test.shape[0]}')

    steps_per_epoch = input_pipeline.num_batches(
        x_train.shape[0], args.batch_size, drop_remainder=True)
    total_steps = steps_per_epoch * args.epochs

    model = create_model(args.model, input_pipeline.NUM_CLASSES)
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    state = create_train_state(
        init_rng, model, args.lr, total_steps, args.warmup_steps,
        args.weight_decay, resolution=args.resolution)

    if args.init_checkpoint:
        ckpt = args.init_checkpoint
        if not os.path.exists(ckpt):
            ckpt = checkpoint.download(args.init_checkpoint, args.cache_dir)
        print(f'Fine-tuning from {args.init_checkpoint} ...')
        pretrained = checkpoint.load_pretrained(
            pretrained_path=ckpt,
            init_params=state.params,
            classifier=configs.get_config(args.model).classifier,
        )
        state = state.replace(params=pretrained)

    param_count = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(state.params))
    print(f'Model {args.model!r} @ {args.resolution}px: {param_count / 1e6:.2f}M parameters')
    print(f'Training for {args.epochs} epochs '
          f'({steps_per_epoch} steps/epoch, {total_steps} total).')

    for epoch in range(args.epochs):
        t0 = time.time()
        running_loss, running_acc, seen = 0.0, 0.0, 0
        for images, labels in input_pipeline.iterate_batches(
                x_train, y_train, args.batch_size, shuffle=True,
                seed=args.seed + epoch, drop_remainder=True):
            state, loss, acc = train_step(
                state, maybe_resize(images, args.resolution), labels, dropout_rng)
            running_loss += float(loss)
            running_acc += float(acc)
            seen += 1
            if seen % args.log_every == 0:
                print(f'  epoch {epoch + 1} step {seen}/{steps_per_epoch} '
                      f'loss={running_loss / seen:.4f} acc={running_acc / seen:.4f}')

        metrics = evaluate(state, x_test, y_test, args.batch_size, args.resolution)
        print(f'[epoch {epoch + 1}/{args.epochs}] '
              f'train_loss={running_loss / max(seen, 1):.4f} '
              f'train_acc={running_acc / max(seen, 1):.4f} '
              f'test_loss={metrics["loss"]:.4f} '
              f'test_acc={metrics["accuracy"]:.4f} '
              f'({time.time() - t0:.1f}s)')

    save_checkpoint(args.output, state, {
        'model': args.model, 'epochs': args.epochs,
        'test_accuracy': round(metrics['accuracy'], 4),
    })


if __name__ == '__main__':
    main()
