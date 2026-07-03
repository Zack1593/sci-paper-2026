"""A compact JAX/Flax Vision Transformer for image classification.

Mirrors the model definition of google-research/vision_transformer and trains
it from scratch on CIFAR-10.
"""

from vit_jax import checkpoint
from vit_jax import configs
from vit_jax import input_pipeline
from vit_jax import models_vit

__all__ = ['checkpoint', 'configs', 'input_pipeline', 'models_vit']