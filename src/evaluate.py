"""Evaluation for the Fashion-MNIST denoising autoencoder.

No labels/threshold here (this isn't anomaly detection) - denoising is
judged by how close the reconstruction is to the clean original, both
numerically (RMSE/PSNR) and visually.
"""

import numpy as np
from tensorflow import keras

from data_prep import CLASS_NAMES, prepare, add_noise
from train import MODEL_PATH, REPORTS_DIR

RECONSTRUCTIONS_PLOT_PATH = REPORTS_DIR / "denoising_examples.png"


def rmse(clean: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(clean - reconstructed))))


def psnr(clean: np.ndarray, reconstructed: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB - higher is better. Common metric
    for judging image reconstruction/denoising quality."""
    mse = np.mean(np.square(clean - reconstructed))
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(1.0) - 10 * np.log10(mse))  # pixels are in [0, 1]


def plot_examples(noisy, clean, reconstructed, labels, n: int = 8):
    import matplotlib.pyplot as plt

    idx = np.random.default_rng(0).choice(len(noisy), size=n, replace=False)
    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))

    for col, i in enumerate(idx):
        axes[0, col].imshow(noisy[i].squeeze(), cmap="gray", interpolation="nearest")
        axes[0, col].set_title(CLASS_NAMES[labels[i]], fontsize=8)
        axes[1, col].imshow(reconstructed[i].squeeze(), cmap="gray", interpolation="nearest")
        axes[2, col].imshow(clean[i].squeeze(), cmap="gray", interpolation="nearest")
        for row in range(3):
            axes[row, col].axis("off")

    axes[0, 0].set_ylabel("noisy", fontsize=10)
    axes[1, 0].set_ylabel("reconstructed", fontsize=10)
    axes[2, 0].set_ylabel("clean", fontsize=10)
    fig.tight_layout()
    fig.savefig(RECONSTRUCTIONS_PLOT_PATH)
    plt.close(fig)


if __name__ == "__main__":
    data = prepare()
    x_test_noisy, x_test, y_test = data["test"]

    model = keras.models.load_model(MODEL_PATH)
    x_test_reconstructed = model.predict(x_test_noisy)

    print(f"RMSE (noisy vs clean): {rmse(x_test, x_test_noisy):.4f}")
    print(f"PSNR (noisy vs clean): {psnr(x_test, x_test_noisy):.2f} dB")

    print(f"RMSE (denoised vs clean): {rmse(x_test, x_test_reconstructed):.4f}")
    print(f"PSNR (denoised vs clean): {psnr(x_test, x_test_reconstructed):.2f} dB")

    noise_levels = [0.15, 0.30, 0.50, 0.80]

    rmse_results = []
    psnr_results = []

    for noise_factor in noise_levels:
        # add_noise() använder samma seed varje gång den anropas här, så
        # bruset blir samma mönster för varje nivå, bara starkare/svagare.
        noisy = add_noise(x_test, noise_factor=noise_factor)
        reconstructed = model.predict(noisy, verbose=0)

        rmse_results.append(rmse(x_test, reconstructed))
        psnr_results.append(psnr(x_test, reconstructed))

        print(f"Noise {noise_factor:.2f} - RMSE: {rmse(x_test, reconstructed):.4f}")
        print(f"Noise {noise_factor:.2f} - PSNR: {psnr(x_test, reconstructed):.2f} dB")

    import matplotlib.pyplot as plt

    plt.plot(noise_levels, rmse_results, marker="o")
    plt.xlabel("Noise level")
    plt.ylabel("RMSE")
    plt.title("RMSE vs noise level")
    plt.savefig(REPORTS_DIR / "noise_rmse.png")
    plt.close()

    plt.plot(noise_levels, psnr_results, marker="o")
    plt.xlabel("Noise level")
    plt.ylabel("PSNR")
    plt.title("PSNR vs noise level")
    plt.savefig(REPORTS_DIR / "noise_psnr.png")
    plt.close()

    plot_examples(x_test_noisy, x_test, x_test_reconstructed, y_test)
