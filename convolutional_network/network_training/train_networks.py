import logging
import pickle

import tensorflow as tf

from data_loading import load_training_dataset, load_batch
from network_architectures import (
    build_minimal_architecture,
    build_conv_sequence_architecture,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("network trainer")


def compile_network(net):
    net.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["acc"],
    )


def train_network(net, name, x_train, y_train, x_test, y_test, batch_size, epochs):
    compile_network(net)
    training_history = net.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    test_score = net.evaluate(x_test, y_test)
    net.save(f"models/{name}")

    with open(f"training_histories/{name}.pkl", "w+b") as training_history_bin:
        pickle.dump(training_history.history, training_history_bin)

    with open(f"test_scores/{name}.pkl", "w+b") as test_score_bin:
        pickle.dump(test_score, test_score_bin)


if __name__ == "__main__":
    train_images, train_labels = load_training_dataset()
    test_images, test_labels = load_batch("test_batch")

    networks = [
        (build_minimal_architecture(), "minimal"),
        (build_minimal_architecture(filters_count=20), "filters_extended_minimal"),
        (
            build_conv_sequence_architecture([20, 40], conv_activation="sigmoid"),
            "conv_seq_short_sigmoid",
        ),
        (build_conv_sequence_architecture([20, 40]), "conv_seq_short"),
        (build_conv_sequence_architecture([20, 40, 80, 160]), "conv_seq"),
        (
            build_conv_sequence_architecture(
                [20, 40, 80, 160], batch_normalization=True
            ),
            "conv_seq_batch_norm",
        ),
        (
            build_conv_sequence_architecture(
                [20, 40, 80, 160],
                batch_normalization=True,
                dropouts=[0.1, 0.2, 0.3, 0.4],
            ),
            "conv_seq_batch_norm_dropout",
        ),
        (
            build_conv_sequence_architecture(
                [20, 40, 80, 160],
                batch_normalization=True,
                dropouts=[0.1, 0.2, 0.3, 0.4],
                global_average_pooling=True,
            ),
            "conv_seq_batch_norm_dropout_gap",
        ),
    ]
    for network, network_name in networks:
        logger.info(f"started {network_name} training.")
        train_network(
            network,
            network_name,
            train_images,
            train_labels,
            test_images,
            test_labels,
            64,
            150,
        )
        logger.info(f"ended {network_name} training.")
