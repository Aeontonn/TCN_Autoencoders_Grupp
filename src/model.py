"""Convolutional autoencoder for Fashion-MNIST denoising.

TODO (Person B): tune filter counts / number of down-sampling steps if
the reconstructions look too blurry or training is too slow.
"""

from tensorflow import keras
from tensorflow.keras import layers

INPUT_SHAPE = (28, 28, 1)


def build_conv_autoencoder(input_shape: tuple[int, int, int] = INPUT_SHAPE) -> keras.Model:
    inputs = keras.Input(shape=input_shape)

    # Encoder: gör bilden mindre två gånger (28 -> 14 -> 7 pixlar).
    x = layers.Conv2D(32, 3, activation="relu", padding="same", strides=2)(inputs)
    x = layers.Conv2D(64, 3, activation="relu", padding="same", strides=2)(x)

    # Decoder: gör bilden stor igen, samma steg som encodern fast baklänges,
    # så att bilden blir 28x28 igen på slutet.
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", padding="same", strides=2)(x)
    outputs = layers.Conv2D(input_shape[-1], 3, activation="sigmoid", padding="same")(x)

    autoencoder = keras.Model(inputs, outputs, name="conv_autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


if __name__ == "__main__":
    model = build_conv_autoencoder()
    model.summary()
