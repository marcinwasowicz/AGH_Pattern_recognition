import tensorflow as tf


def _build_conv_block(
    filters_count, activation="relu", batch_normalization=False, dropout=None
):
    conv_unit = [
        tf.keras.layers.Conv2D(
            filters_count, (3, 3), activation=activation, padding="same"
        )
    ]
    if batch_normalization:
        conv_unit.append(tf.keras.layers.BatchNormalization())

    pooling_unit = [tf.keras.layers.MaxPooling2D(2, 2)]
    if dropout is not None:
        pooling_unit.append(tf.keras.layers.Dropout(dropout))

    return conv_unit + conv_unit + pooling_unit


def build_conv_sequence_architecture(
    filters_counts,
    conv_activation="relu",
    batch_normalization=False,
    dropouts=None,
    global_average_pooling=False,
):
    layers = [tf.keras.layers.Rescaling(scale=1.0 / 255)]
    for block_id, filters_count in enumerate(filters_counts):
        layers.extend(
            _build_conv_block(
                filters_count,
                activation=conv_activation,
                batch_normalization=batch_normalization,
                dropout=dropouts[block_id] if dropouts else None,
            )
        )
    if global_average_pooling:
        if dropouts:
            last_pooling_index = -1
        else:
            last_pooling_index = -2
        layers[last_pooling_index] = tf.keras.layers.GlobalAveragePooling2D()
    layers.extend(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    return tf.keras.models.Sequential(layers)
