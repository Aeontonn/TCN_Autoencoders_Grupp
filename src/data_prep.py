"""Data loading and noise generation for the Fashion-MNIST denoising task.

Denoising autoencoder setup: input = noisy image, target = the original
clean image. The model learns to strip the noise back out.

Fashion-MNIST ships with Keras, so there's nothing to download - it's
fetched (and cached) automatically the first time load_data() runs.
"""

import numpy as np
from tensorflow.keras.datasets import fashion_mnist

NOISE_FACTOR = 0.3  # how strong the added noise is
RANDOM_STATE = 42

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def load_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    return (x_train, y_train), (x_test, y_test)


def normalize(images: np.ndarray) -> np.ndarray:
    """Scale pixel values to [0, 1] and add a channel dimension for Conv2D."""
    images = images.astype("float32") / 255.0
    return images[..., np.newaxis]  # (N, 28, 28) -> (N, 28, 28, 1)


def add_noise(images: np.ndarray, noise_factor: float = NOISE_FACTOR, rng: np.random.Generator = None) -> np.ndarray:
    """Adds Gaussian noise and clips back to a valid pixel range."""
    rng = rng or np.random.default_rng(RANDOM_STATE)
    noisy = images + noise_factor * rng.normal(size=images.shape)
    return np.clip(noisy, 0.0, 1.0)


def prepare(val_fraction: float = 0.1):
    """Convenience wrapper: load -> normalize -> split off a val set -> add noise.

    Returns clean and noisy versions for train/val/test, plus labels
    (not used to train the autoencoder, but handy for inspecting results
    per clothing category during evaluation).
    """
    (x_train_full, y_train_full), (x_test, y_test) = load_data()

    x_train_full = normalize(x_train_full)
    x_test = normalize(x_test)

    n_val = int(len(x_train_full) * val_fraction)
    x_val, x_train = x_train_full[:n_val], x_train_full[n_val:]
    y_val, y_train = y_train_full[:n_val], y_train_full[n_val:]

    # Vi använder samma rng för alla tre delar (train/val/test), så att de
    # inte får exakt likadant brus, men resultatet blir ändå samma varje gång.
    rng = np.random.default_rng(RANDOM_STATE)
    x_train_noisy = add_noise(x_train, rng=rng)
    x_val_noisy = add_noise(x_val, rng=rng)
    x_test_noisy = add_noise(x_test, rng=rng)

    return {
        "train": (x_train_noisy, x_train, y_train),
        "val": (x_val_noisy, x_val, y_val),
        "test": (x_test_noisy, x_test, y_test),
    }


if __name__ == "__main__":
    data = prepare()
    for split_name, (noisy, clean, labels) in data.items():
        print(f"{split_name}: noisy {noisy.shape}, clean {clean.shape}, labels {labels.shape}")
