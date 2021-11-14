import pickle
from typing import List

import numpy as np
import tensorflow as tf


def one_hot_encode(sparse_labels: List[int]):
    return tf.keras.utils.to_categorical(
        sparse_labels, num_classes=max(sparse_labels) + 1, dtype="uint8"
    )


def one_hot_decode(one_hot_vectors):
    return [tf.math.argmax(vec) for vec in one_hot_vectors]


def load_batch(batch_name):
    with open(f"dataset/cifar-10-batches-py/{batch_name}", "rb") as data_batch:
        data_dict = pickle.load(data_batch, encoding="bytes")

    data = np.array(
        [arr.reshape((3, 1024)).T.reshape((32, 32, 3)) for arr in data_dict[b"data"]]
    )
    labels = one_hot_encode(data_dict[b"labels"])

    return data, labels


def load_training_dataset():
    batch_names = [f"data_batch_{idx}" for idx in [1, 2, 3, 4, 5]]
    batches = [load_batch(batch_name) for batch_name in batch_names]

    data = tf.concat([batch[0] for batch in batches], 0)
    labels = tf.concat([batch[1] for batch in batches], 0)

    return data, labels
