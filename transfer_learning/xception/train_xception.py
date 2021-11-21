import logging
import pickle

import tensorflow as tf

from dataset import (
    load_dataset,
    DUMBELL_KETTLEBELL_BAREBELL,
    LASAGNE_MUSAKA_TART,
    LARGE_IMAGE_SIZE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("xception trainer")

EPOCHS = 30
FINE_TUNING_EPOCHS = 15


def prepare_xception():
    xception = tf.keras.applications.Xception(
        include_top=False, weights="imagenet", input_shape=(256, 256, 3), pooling=None
    )
    xception.trainable = False

    new_gap = tf.keras.layers.GlobalAveragePooling2D()(xception.layers[-1].output)
    new_dense = tf.keras.layers.Dense(3, activation="softmax")(new_gap)
    new_xception = tf.keras.models.Model(inputs=xception.input, outputs=new_dense)
    new_xception.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["acc"],
    )
    return new_xception


def preprare_xception_fine_tuning(trained_xception_save_name):
    trained_xception = tf.keras.models.load_model(
        f"models/{trained_xception_save_name}"
    )
    trained_xception.trainable = True
    trained_xception.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["acc"],
    )
    return trained_xception


def train_network(name, net, x, y, epochs=EPOCHS):
    history = net.fit(x, validation_data=y, epochs=epochs)
    net.save(f"models/{name}")
    with open(f"histories/{name}.pkl", "w+b") as history_binary:
        pickle.dump(history.history, history_binary)


if __name__ == "__main__":
    for dataset_name, data in [
        ("lasagne_musaka_tart", load_dataset(LASAGNE_MUSAKA_TART, LARGE_IMAGE_SIZE)),
        (
            "dumbell_kettlebell_barebell",
            load_dataset(DUMBELL_KETTLEBELL_BAREBELL, LARGE_IMAGE_SIZE),
        ),
    ]:
        train, validation = data
        xception_save_name = "_".join([dataset_name, "xception"])
        logger.info(f" Started {xception_save_name} training.")
        train_network(xception_save_name, prepare_xception(), train, validation)
        logger.info(f" Ended {xception_save_name} training.")

    for dataset_name, data in [
        ("lasagne_musaka_tart", load_dataset(LASAGNE_MUSAKA_TART, LARGE_IMAGE_SIZE)),
        (
            "dumbell_kettlebell_barebell",
            load_dataset(DUMBELL_KETTLEBELL_BAREBELL, LARGE_IMAGE_SIZE),
        ),
    ]:
        train, validation = data
        xception_save_name = "_".join([dataset_name, "xception"])
        fine_tuned_xception_save_name = "_".join([dataset_name, "fine_tuned_xception"])
        logger.info(f" Started {fine_tuned_xception_save_name} training.")
        train_network(
            fine_tuned_xception_save_name,
            preprare_xception_fine_tuning(xception_save_name),
            train,
            validation,
            FINE_TUNING_EPOCHS,
        )
        logger.info(f" Ended {xception_save_name} training.")
