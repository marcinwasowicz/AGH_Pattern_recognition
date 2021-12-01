import tensorflow as tf


def build_upsampling_generator():
    layers = [
        tf.keras.layers.Input(shape=(128,)),
        tf.keras.layers.Dense(8192),
        tf.keras.layers.Reshape((8, 8, 128)),
    ]

    for filter_count in [64, 128, 256]:
        layers.extend(
            [
                tf.keras.layers.UpSampling2D(size=(2, 2)),
                tf.keras.layers.Conv2D(
                    filters=filter_count, kernel_size=(3, 3), strides=1, padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Conv2D(
                    filters=filter_count, kernel_size=(3, 3), strides=1, padding="same"
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(alpha=0.2),
            ]
        )

    layers.append(
        tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=(5, 5),
            activation="sigmoid",
            strides=1,
            padding="same",
        )
    )
    return tf.keras.models.Sequential(layers)
