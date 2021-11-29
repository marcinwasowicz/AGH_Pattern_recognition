import tensorflow as tf


def build_discriminator():
    layers = [tf.keras.layers.Input(shape=(64, 64, 3))]
    for filter_count in [64, 128, 128]:
        layers.extend(
            [
                tf.keras.layers.Conv2D(
                    filters=filter_count, kernel_size=(4, 4), strides=2, padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(alpha=0.2),
            ]
        )

    layers.append(tf.keras.layers.Flatten())
    layers.append(tf.keras.layers.Dropout(0.2))
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))

    return tf.keras.models.Sequential(layers)
