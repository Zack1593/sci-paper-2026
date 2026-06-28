"""Loading official pretrained Vision Transformer checkpoints.

The checkpoints published by Google Research
(https://github.com/google-research/vision_transformer#available-vit-models)
are stored as ``.npz`` files under ``gs://vit_models`` / the equivalent public
HTTPS bucket. Each array is keyed by its slash-joined parameter path, e.g.
``Transformer/encoderblock_0/LayerNorm_2/scale``.

Those names were produced by an older version of Flax whose auto-naming used a
single shared counter per ``@nn.compact`` method. Modern Flax (used here) uses a
per-class counter, so a couple of submodule names differ. ``_RENAME`` maps the
checkpoint names onto this project's parameter tree; everything else (array
shapes, ``query/key/value/out``, ``Dense_0/Dense_1`` ...) is identical.

The :func:`load_pretrained` helper mirrors the upstream ``checkpoint.py``: it
drops the ``pre_logits`` block when the model has no representation layer,
re-initialises the classification head when the number of classes differs, and
interpolates the position embeddings when the input resolution changes.
"""

import os
import ssl
import urllib.request
from typing import Any, Dict, Tuple

import jax
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict

BASE_URL = 'https://storage.googleapis.com/vit_models/'

# Official (old-Flax) submodule name -> this project's (new-Flax) name.
_RENAME = {
    'MultiHeadDotProductAttention_1': 'MultiHeadDotProductAttention_0',
    'LayerNorm_2': 'LayerNorm_1',
    'MlpBlock_3': 'MlpBlock_0',
}

PathKey = Tuple[str, ...]


def _rename(key: PathKey) -> PathKey:
  return tuple(_RENAME.get(part, part) for part in key)


def download(filename: str, cache_dir: str) -> str:
  """Downloads ``BASE_URL + filename`` into ``cache_dir`` (cached).

  Args:
    filename: e.g. ``'imagenet21k+imagenet2012/ViT-B_16.npz'``.
    cache_dir: local directory to store the file in.

  Returns:
    Absolute path to the downloaded file.
  """
  os.makedirs(cache_dir, exist_ok=True)
  dest = os.path.join(cache_dir, filename.replace('/', '_'))
  if os.path.exists(dest) and os.path.getsize(dest) > 0:
    return dest

  url = BASE_URL + filename
  print(f'Downloading {url} ...')
  tmp = dest + '.part'
  try:
    with urllib.request.urlopen(url) as resp:
      total = int(resp.headers.get('Content-Length', 0))
      done, last_print = 0, 0
      with open(tmp, 'wb') as out:
        while True:
          chunk = resp.read(1 << 20)
          if not chunk:
            break
          out.write(chunk)
          done += len(chunk)
          if total and done - last_print >= (16 << 20):  # every ~16 MB
            last_print = done
            print(f'\r  {done/1e6:7.1f} / {total/1e6:7.1f} MB '
                  f'({100*done/total:5.1f}%)', end='', flush=True)
  except urllib.error.URLError as err:
    if not isinstance(getattr(err, 'reason', None), ssl.SSLError):
      raise
    raise RuntimeError(
        f'TLS verification failed downloading {url}. Update your CA '
        f'certificates and retry.') from err
  print()
  os.replace(tmp, dest)
  return dest


def load(path: str) -> Dict[PathKey, np.ndarray]:
  """Loads a ``.npz`` checkpoint into a flat ``{path_tuple: array}`` dict."""
  with open(path, 'rb') as f:
    data = np.load(f, allow_pickle=False)
    return {tuple(k.split('/')): np.asarray(v) for k, v in data.items()}


def _resize_posembed(grid: np.ndarray, gs_new: int) -> np.ndarray:
  """Bilinearly resizes a (gs_old*gs_old, dim) grid of position embeddings to
  (gs_new*gs_new, dim)."""
  dim = grid.shape[-1]
  gs_old = int(round(grid.shape[0] ** 0.5))
  grid = grid.reshape(gs_old, gs_old, dim)
  grid = np.asarray(
      jax.image.resize(grid, (gs_new, gs_new, dim), method='bilinear'))
  return grid.reshape(gs_new * gs_new, dim)


def load_pretrained(
    *,
    pretrained_path: str,
    init_params: Dict[str, Any],
    classifier: str = 'token',
    verbose: bool = True,
) -> Dict[str, Any]:
  """Converts an official checkpoint into this project's parameter tree.

  The returned tree has exactly the structure of ``init_params``: checkpoint
  values are used wherever the (renamed) path exists and shapes agree, after
  resizing the position embeddings if necessary. Anything else falls back to
  ``init_params`` -- this is what re-initialises the head for a new number of
  classes and drops the ``pre_logits`` block for models without one.

  Args:
    pretrained_path: path to a downloaded ``.npz`` checkpoint.
    init_params: freshly initialised parameters of the target model.
    classifier: 'token' if a class token is prepended to the sequence.
    verbose: print which parameters were loaded / re-initialised.

  Returns:
    A parameter pytree ready to pass as ``{'params': ...}`` to ``model.apply``.
  """
  restored = {_rename(k): v for k, v in load(pretrained_path).items()}
  init_flat = flatten_dict(init_params)

  posemb_key = ('Transformer', 'posembed_input', 'pos_embedding')
  if posemb_key in restored and posemb_key in init_flat:
    posemb = restored[posemb_key]
    posemb_new = np.asarray(init_flat[posemb_key])
    if posemb.shape != posemb_new.shape:
      ntok_new = posemb_new.shape[1]
      if classifier == 'token':
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
      else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
      gs_new = int(round(ntok_new ** 0.5))
      posemb_grid = _resize_posembed(posemb_grid, gs_new)[None]
      restored[posemb_key] = np.concatenate([posemb_tok, posemb_grid], axis=1)
      if verbose:
        print(f'  resized pos_embedding {posemb.shape} -> '
              f'{restored[posemb_key].shape}')

  merged: Dict[PathKey, Any] = {}
  reinit, loaded = [], 0
  for key, init_val in init_flat.items():
    cand = restored.get(key)
    if cand is not None and tuple(cand.shape) == tuple(init_val.shape):
      merged[key] = jax.numpy.asarray(cand)
      loaded += 1
    else:
      merged[key] = init_val
      reinit.append('/'.join(key))

  if verbose:
    extra = [k for k in restored if k not in init_flat]
    print(f'  loaded {loaded} tensors from checkpoint; '
          f'kept init for {len(reinit)}: {reinit if reinit else "none"}')
    if extra:
      names = ['/'.join(k) for k in extra]
      print(f'  ignored {len(extra)} checkpoint-only tensors '
            f'(e.g. {names[:3]})')

  return unflatten_dict(merged)