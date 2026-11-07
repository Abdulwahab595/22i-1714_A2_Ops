"""One decoding and resizing implementation shared by training and inference."""
import io
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

LABELS = {0: "real", 1: "ai_generated"}
DISPLAY_LABELS = {"real": "Real", "ai_generated": "AI-generated"}


def decode_rgb(content: bytes) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            return np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float32)


def resize_pixels(pixels, image_size):
    import tensorflow as tf

    # Preserve aspect ratio, bilinear interpolation, zero-pad the remaining border.
    return tf.image.resize_with_pad(pixels, image_size, image_size, antialias=True)


def preprocess_bytes(content, image_size=128):
    import tensorflow as tf

    return tf.expand_dims(resize_pixels(decode_rgb(content), image_size), 0)


def _read_rgb(path):
    if hasattr(path, "numpy"):
        path = path.numpy()
    if isinstance(path, np.ndarray):
        path = path.item()
    if isinstance(path, bytes):
        path = path.decode("utf-8")
    return decode_rgb(Path(path).read_bytes())


def make_dataset(frame, raw_root, image_size=128, batch_size=32, training=False, seed=42,
                 cache_dir=None):
    import tensorflow as tf

    paths = [str(Path(raw_root) / p) for p in frame["path"]]
    labels = frame["label"].to_numpy(dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def decode(path, label):
        pixels = tf.numpy_function(_read_rgb, [path], tf.float32)
        pixels.set_shape([None, None, 3])
        return resize_pixels(pixels, image_size), label

    dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    if cache_dir:
        Path(cache_dir).parent.mkdir(parents=True, exist_ok=True)
        dataset = dataset.cache(str(cache_dir))
    if training:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 4
    return dataset.with_options(options)
