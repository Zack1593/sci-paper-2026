"""Classify an image with an official pretrained ViT, e.g. ViT-B_16.

Downloads the checkpoint (and ImageNet labels) on first use, loads the weights
into the Flax model from this project, and prints the top-k ImageNet-1k classes.

Example:
    python predict_pretrained.py --image cat.jpg

The default checkpoint ``imagenet21k+imagenet2012/ViT-B_16.npz`` was fine-tuned
at 384x384 on ImageNet-2012, so images are resized to 384 and the model outputs
1000-way logits in the standard ImageNet (WordNet-id sorted) order.
"""

import argparse
import json
import os
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from vit_jax import checkpoint
from vit_jax import configs
from vit_jax import models_vit

_LABELS_URL = ('https://storage.googleapis.com/download.tensorflow.org/'
               'data/imagenet_class_index.json')


def load_labels(cache_dir: str) -> list:
    """Returns the 1000 ImageNet class names in logit order."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, 'imagenet_class_index.json')
    if not os.path.exists(path):
        print(f'Downloading ImageNet labels from {_LABELS_URL} ...')
        urllib.request.urlretrieve(_LABELS_URL, path)
    with open(path) as f:
        index = json.load(f)
    return [index[str(i)][1].replace('_', ' ') for i in range(len(index))]


def load_image(path: str, resolution: int) -> np.ndarray:
    img = Image.open(path).convert('RGB').resize((resolution, resolution))
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return arr[None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--model', default='b16', choices=sorted(configs.MODEL_CONFIGS))
    parser.add_argument('--checkpoint',
                        default='imagenet21k+imagenet2012/ViT-B_16.npz',
                        help='Checkpoint name under gs://vit_models/.')
    parser.add_argument('--resolution', type=int, default=384)
    parser.add_argument('--num-classes', type=int, default=1000)
    parser.add_argument('--cache-dir', default='pretrained')
    parser.add_argument('--topk', type=int, default=5)
    args = parser.parse_args()

    config = configs.get_config(args.model)
    model = models_vit.VisionTransformer(
        num_classes=args.num_classes,
        patches=config.patches,
        transformer=config.transformer,
        hidden_size=config.hidden_size,
        representation_size=config.representation_size,
        classifier=config.classifier,
    )

    print(f'Initialising {args.model} at {args.resolution}px ...')
    dummy = jnp.ones([1, args.resolution, args.resolution, 3])
    init_params = model.init(jax.random.PRNGKey(0), dummy, train=False)['params']

    ckpt_path = checkpoint.download(args.checkpoint, args.cache_dir)
    print('Loading pretrained weights ...')
    params = checkpoint.load_pretrained(
        pretrained_path=ckpt_path,
        init_params=init_params,
        classifier=config.classifier,
    )

    labels = load_labels(args.cache_dir)
    x = load_image(args.image, args.resolution)
    logits = model.apply({'params': params}, x, train=False)
    probs = jax.nn.softmax(logits, axis=-1)[0]

    topk = np.argsort(np.asarray(probs))[::-1][:args.topk]
    print(f'\nTop-{args.topk} predictions for {args.image}:')
    for rank, idx in enumerate(topk, 1):
        print(f'  {rank}. {labels[idx]:<28} {float(probs[idx]) * 100:5.2f}%')


if __name__ == '__main__':
    main()
