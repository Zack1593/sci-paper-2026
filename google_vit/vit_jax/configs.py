"""Model configurations for the Vision Transformer.

The widths/depths follow the naming convention from the ViT paper
("An Image is Worth 16x16 Words", Dosovitskiy et al., 2021) and the official
``vit_jax/configs/models.py``. ``patch4`` variants are added for small
low-resolution datasets such as CIFAR-10, where a 16x16 patch would leave too
few tokens.
"""

import ml_collections


def _base() -> ml_collections.ConfigDict:
  config = ml_collections.ConfigDict()
  config.patches = ml_collections.ConfigDict({'size': (16, 16)})
  config.hidden_size = 768
  config.classifier = 'token'
  config.representation_size = None
  config.transformer = ml_collections.ConfigDict()
  config.transformer.num_layers = 12
  config.transformer.mlp_dim = 3072
  config.transformer.num_heads = 12
  config.transformer.dropout_rate = 0.1
  config.transformer.attention_dropout_rate = 0.0
  return config


def get_ti16_config() -> ml_collections.ConfigDict:
  """ViT-Tiny/16."""
  config = _base()
  config.name = 'ViT-Ti_16'
  config.hidden_size = 192
  config.transformer.mlp_dim = 768
  config.transformer.num_heads = 3
  return config


def get_s16_config() -> ml_collections.ConfigDict:
  """ViT-Small/16."""
  config = _base()
  config.name = 'ViT-S_16'
  config.hidden_size = 384
  config.transformer.mlp_dim = 1536
  config.transformer.num_heads = 6
  return config


def get_b16_config() -> ml_collections.ConfigDict:
  """ViT-Base/16."""
  config = _base()
  config.name = 'ViT-B_16'
  return config


def get_cifar_tiny_config() -> ml_collections.ConfigDict:
  """A small ViT with 4x4 patches, sized for training CIFAR-10 from scratch on
  CPU within a reasonable time budget."""
  config = ml_collections.ConfigDict()
  config.name = 'ViT-CIFAR-Tiny'
  config.patches = ml_collections.ConfigDict({'size': (4, 4)})
  config.hidden_size = 128
  config.classifier = 'token'
  config.representation_size = None
  config.transformer = ml_collections.ConfigDict()
  config.transformer.num_layers = 6
  config.transformer.mlp_dim = 256
  config.transformer.num_heads = 4
  config.transformer.dropout_rate = 0.1
  config.transformer.attention_dropout_rate = 0.0
  return config


MODEL_CONFIGS = {
    'ti16': get_ti16_config,
    's16': get_s16_config,
    'b16': get_b16_config,
    'cifar_tiny': get_cifar_tiny_config,
}


def get_config(name: str) -> ml_collections.ConfigDict:
  """Returns the model config for ``name`` (e.g. 'cifar_tiny', 'b16')."""
  if name not in MODEL_CONFIGS:
    raise KeyError(
        f'Unknown model config {name!r}. '
        f'Available: {sorted(MODEL_CONFIGS)}')
  return MODEL_CONFIGS[name]()