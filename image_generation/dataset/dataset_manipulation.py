import tensorflow as tf
from matplotlib import pyplot as plt

_RANDOM_SEED = 42


def load_dataset(path, image_size):
    data = tf.keras.utils.image_dataset_from_directory(
        path, image_size=image_size, seed=_RANDOM_SEED, labels=None
    ).unbatch()

    rescaling = tf.keras.layers.Rescaling(1.0 / 255)
    data = data.map(lambda x: rescaling(x))
    return data


def augment_dataset(dataset, image_copy_count, shuffle=True):
    data_augmentation = tf.keras.models.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(height_factor=0.15, width_factor=0.15),
        ]
    )
    dataset = dataset.map(
        lambda x: data_augmentation(
            tf.repeat(tf.expand_dims(x, axis=0), repeats=image_copy_count, axis=0)
        )
    ).unbatch()
    if shuffle:
        dataset = dataset.shuffle(1000, seed=_RANDOM_SEED)
    return dataset


def plot_images(dataset, count):
    batch = dataset.take(count)
    for image in batch:
        plt.imshow(image.numpy())
        plt.show()
        plt.clf()
