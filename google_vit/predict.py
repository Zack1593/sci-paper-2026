"""Run a trained Vision Transformer on an image and print the top predictions.

Example:
    python predict.py --image some_cat.png --checkpoint checkpoints/vit_cifar10.msgpack
"""

import argparse

import flax
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from vit_jax import configs
from vit_jax import input_pipeline
from vit_jax import models_vit


def load_image(path: str) -> np.ndarray:
  """Loads an image, resizes to 32x32 and scales to [-1, 1]."""
  img = Image.open(path).convert('RGB').resize(
      (input_pipeline.IMAGE_SIZE, input_pipeline.IMAGE_SIZE))
  arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
  return arr[None]  # add batch dim -> (1, 32, 32, 3)


def build_and_load(model_name: str, checkpoint: str):
  config = configs.get_config(model_name)
  model = models_vit.VisionTransformer(
      num_classes=input_pipeline.NUM_CLASSES,
      patches=config.patches,
      transformer=config.transformer,
      hidden_size=config.hidden_size,
      representation_size=config.representation_size,
      classifier=config.classifier,
  )
  dummy = jnp.ones([1, input_pipeline.IMAGE_SIZE, input_pipeline.IMAGE_SIZE, 3])
  params = model.init(jax.random.PRNGKey(0), dummy, train=False)['params']
  with open(checkpoint, 'rb') as f:
    params = flax.serialization.from_bytes(params, f.read())
  return model, params


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--image', required=True, help='Path to an image file.')
  parser.add_argument('--model', default='cifar_tiny',
                      choices=sorted(configs.MODEL_CONFIGS))
  parser.add_argument('--checkpoint', default='checkpoints/vit_cifar10.msgpack')
  parser.add_argument('--topk', type=int, default=3)
  args = parser.parse_args()

  model, params = build_and_load(args.model, args.checkpoint)
  x = load_image(args.image)
  logits = model.apply({'params': params}, x, train=False)
  probs = jax.nn.softmax(logits, axis=-1)[0]

  topk = np.argsort(np.asarray(probs))[::-1][:args.topk]
  print(f'Predictions for {args.image}:')
  for rank, idx in enumerate(topk, 1):
    print(f'  {rank}. {input_pipeline.CIFAR10_CLASSES[idx]:<12} '
          f'{float(probs[idx])*100:5.2f}%')


if __name__ == '__main__':
  main()