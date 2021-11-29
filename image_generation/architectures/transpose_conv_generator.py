import tensorflow as tf


def build_transpose_conv_generator():
    layers = [
        tf.keras.layers.Input(shape=(128,)),
        tf.keras.layers.Dense(8192),
        tf.keras.layers.Reshape((8, 8, 128)),
    ]

    for filter_count in [128, 256, 512]:
        layers.extend(
            [
                tf.keras.layers.Conv2DTranspose(
                    filters=filter_count, kernel_size=(4, 4), strides=2, padding="same"
                ),
                tf.keras.layers.LeakyReLU(alpha=0.2),
            ]
        )

    layers.append(
        tf.keras.layers.Conv2D(
            3, kernel_size=(5, 5), strides=1, padding="same", activation="sigmoid"
        )
    )
    return tf.keras.models.Sequential(layers)
