import logging
import pickle

from dataset import load_dataset, LASAGNE_MUSAKA_TART, DUMBELL_KETTLEBELL_BAREBELL
from private_architecture.private_architecture import (
    build_convolutional_architecture,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("private architecture trainer")


def train_network(name, net, x, y):
    history = net.fit(x, validation_data=y, epochs=150)
    net.save(f"models/{name}")
    with open(f"histories/{name}.pkl", "w+b") as history_binary:
        pickle.dump(history.history, history_binary)


if __name__ == "__main__":
    for dataset_name, data in [
        ("lasagne_musaka_tart", load_dataset(LASAGNE_MUSAKA_TART)),
        ("dumbell_kettlebell_barebell", load_dataset(DUMBELL_KETTLEBELL_BAREBELL)),
    ]:
        for network_name, network in [
            ("directly_trained", build_convolutional_architecture())
        ]:
            save_name = "_".join([dataset_name, network_name])
            logger.info(f" Started {save_name} training.")
            train, validation = data
            train_network(save_name, network, train, validation)
            logger.info(f" Ended {save_name} training.")
