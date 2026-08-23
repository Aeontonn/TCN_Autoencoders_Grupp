"""Training loop for the Fashion-MNIST denoising autoencoder.

Input = noisy image, target = clean image (NOT the same as input, unlike
a typical reconstruction autoencoder - here the model learns to remove
noise, not just copy its input).
"""

from pathlib import Path

from tensorflow import keras

from data_prep import prepare
from model import build_conv_autoencoder

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
MODEL_PATH = REPORTS_DIR / "autoencoder.keras"
HISTORY_PLOT_PATH = REPORTS_DIR / "training_loss.png"


def train(epochs: int = 20, batch_size: int = 128):
    data = prepare()
    x_train_noisy, x_train, _ = data["train"]
    x_val_noisy, x_val, _ = data["val"]

    autoencoder = build_conv_autoencoder()

    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    history = autoencoder.fit(
        x_train_noisy, x_train,
        validation_data=(x_val_noisy, x_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
    )

    REPORTS_DIR.mkdir(exist_ok=True)
    autoencoder.save(MODEL_PATH)
    plot_history(history)
    return autoencoder, history, data


def plot_history(history):
    import matplotlib.pyplot as plt

    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig(HISTORY_PLOT_PATH)
    plt.close()


if __name__ == "__main__":
    train()
