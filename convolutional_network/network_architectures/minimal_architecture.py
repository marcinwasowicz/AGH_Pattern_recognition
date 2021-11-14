import tensorflow as tf


def build_minimal_architecture(filters_count=5):
    minimal_architecture = tf.keras.models.Sequential(
        [
            tf.keras.layers.Rescaling(scale=1.0 / 255),
            tf.keras.layers.Conv2D(
                filters_count, (3, 3), activation="sigmoid", padding="same"
            ),
            tf.keras.layers.Conv2D(
                filters_count, (3, 3), activation="sigmoid", padding="same"
            ),
            tf.keras.layers.MaxPooling2D(8, 8),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    return minimal_architecture
