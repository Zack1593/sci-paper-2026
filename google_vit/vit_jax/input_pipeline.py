"""Input pipeline for CIFAR-10.

The official repository uses ``tensorflow_datasets``. To keep dependencies light
and to be robust to the (currently expired) TLS certificate on the canonical
CIFAR host, this module parses the CIFAR-10 python pickles directly with the
standard library, reading the archive from ``google_vit/cifar-10-dataset`` and
only downloading it if it isn't already there. Images are scaled to the
``[-1, 1]`` range, matching the preprocessing used upstream.
"""

import hashlib
import os
import pickle
import ssl
import tarfile
import urllib.request
from typing import Iterator, Tuple

import numpy as np

CIFAR10_CLASSES = (
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
)
NUM_CLASSES = len(CIFAR10_CLASSES)
IMAGE_SIZE = 32

_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
# Official SHA-256 of cifar-10-python.tar.gz (also used by tf.keras).
_SHA256 = '6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce'
_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'cifar-10-dataset')
_ARCHIVE = os.path.join(_CACHE_DIR, 'cifar-10-python.tar.gz')
_EXTRACT_DIR = os.path.join(_CACHE_DIR, 'cifar-10-batches-py')


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _download_archive() -> None:
    """Downloads the CIFAR-10 archive, falling back to an unverified TLS context
    if (and only if) certificate verification fails. Integrity is still ensured
    via the official SHA-256."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    if os.path.exists(_ARCHIVE) and _sha256(_ARCHIVE) == _SHA256:
        return
    print(f'Downloading CIFAR-10 from {_URL} ...')
    try:
        with urllib.request.urlopen(_URL) as resp, open(_ARCHIVE, 'wb') as out:
            out.write(resp.read())
    except urllib.error.URLError as err:
        if not isinstance(getattr(err, 'reason', None), ssl.SSLError):
            raise
        print('  TLS certificate verification failed for the CIFAR host; '
              'retrying without verification (archive is checked against the '
              'official SHA-256).')
        ctx = ssl._create_unverified_context()  # noqa: SLF001
        with urllib.request.urlopen(_URL, context=ctx) as resp, \
                open(_ARCHIVE, 'wb') as out:
            out.write(resp.read())

    digest = _sha256(_ARCHIVE)
    if digest != _SHA256:
        os.remove(_ARCHIVE)
        raise RuntimeError(
            f'CIFAR-10 archive hash mismatch: got {digest}, expected {_SHA256}.')


def _ensure_extracted() -> None:
    if os.path.isdir(_EXTRACT_DIR) and os.path.exists(
            os.path.join(_EXTRACT_DIR, 'test_batch')):
        return
    _download_archive()
    print(f'Extracting to {_EXTRACT_DIR} ...')
    with tarfile.open(_ARCHIVE, 'r:gz') as tar:
        tar.extractall(_CACHE_DIR)


def _load_batch(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, 'rb') as f:
        entry = pickle.load(f, encoding='bytes')
    data = entry[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(entry[b'labels'], dtype=np.int32)
    return data, labels


def _preprocess(images: np.ndarray) -> np.ndarray:
    """Scales uint8 images in [0, 255] to float32 in [-1, 1]."""
    return images.astype(np.float32) / 127.5 - 1.0


def load_cifar10(
        train_size: int = 0,
        test_size: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads CIFAR-10 as numpy arrays.

    Args:
      train_size: if > 0, keep only the first ``train_size`` training examples.
      test_size: if > 0, keep only the first ``test_size`` test examples.

    Returns:
      (x_train, y_train, x_test, y_test) where images are float32 in [-1, 1] with
      shape (N, 32, 32, 3) and labels are int32 of shape (N,).
    """
    _ensure_extracted()

    xs, ys = [], []
    for i in range(1, 6):
        x, y = _load_batch(os.path.join(_EXTRACT_DIR, f'data_batch_{i}'))
        xs.append(x)
        ys.append(y)
    x_train = _preprocess(np.concatenate(xs))
    y_train = np.concatenate(ys)

    x_test, y_test = _load_batch(os.path.join(_EXTRACT_DIR, 'test_batch'))
    x_test = _preprocess(x_test)

    if train_size:
        x_train, y_train = x_train[:train_size], y_train[:train_size]
    if test_size:
        x_test, y_test = x_test[:test_size], y_test[:test_size]

    return x_train, y_train, x_test, y_test


def iterate_batches(
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        *,
        shuffle: bool = False,
        seed: int = 0,
        drop_remainder: bool = False,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yields (images, labels) minibatches."""
    n = images.shape[0]
    order = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        if drop_remainder and idx.shape[0] < batch_size:
            break
        yield images[idx], labels[idx]


def num_batches(n: int, batch_size: int, drop_remainder: bool = False) -> int:
    if drop_remainder:
        return n // batch_size
    return (n + batch_size - 1) // batch_size
