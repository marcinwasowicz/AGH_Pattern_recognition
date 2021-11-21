import logging
import pickle

import tensorflow as tf

from dataset import load_dataset, LASAGNE_MUSAKA_TART, DUMBELL_KETTLEBELL_BAREBELL
from private_architecture import build_convolutional_architecture

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("private architecture trainer")

EPOCHS = 150


def get_cifar10_pretrained(cifar10_path, num_classes=3):
    model1 = tf.keras.models.load_model(cifar10_path)
    for layer in model1.layers:
        layer.trainable = False

    new_dense_layer = tf.keras.layers.Dense(num_classes, activation="softmax")(
        model1.layers[-2].output
    )
    model2 = tf.keras.models.Model(inputs=model1.input, outputs=new_dense_layer)
    model2.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["acc"],
    )
    return model2


def train_network(name, net, x, y):
    history = net.fit(x, validation_data=y, epochs=EPOCHS)
    net.save(f"models/{name}")
    with open(f"histories/{name}.pkl", "w+b") as history_binary:
        pickle.dump(history.history, history_binary)


if __name__ == "__main__":
    for dataset_name, data in [
        ("lasagne_musaka_tart", load_dataset(LASAGNE_MUSAKA_TART)),
        ("dumbell_kettlebell_barebell", load_dataset(DUMBELL_KETTLEBELL_BAREBELL)),
    ]:
        for network_name, network in [
            ("directly_trained", build_convolutional_architecture()),
            ("cifar10_pretrained", get_cifar10_pretrained("models/cifar10_trained")),
        ]:
            save_name = "_".join([dataset_name, network_name])
            logger.info(f" Started {save_name} training.")
            train, validation = data
            train_network(save_name, network, train, validation)
            logger.info(f" Ended {save_name} training.")
