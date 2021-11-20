import tensorflow as tf


LASAGNE_MUSAKA_TART = "../dataset/lasagne_musaka_tart"
DUMBELL_KETTLEBELL_BAREBELL = "../dataset/dumbell_kettlebell_barebell"

RANDOM_SEED = 42
BATCH_SIZE = 64
SMALL_IMAGE_SIZE = (32, 32)
LARGE_IMAGE_SIZE = (256, 256)


def load_dataset(name, image_size=SMALL_IMAGE_SIZE):
    train = tf.keras.utils.image_dataset_from_directory(
        name,
        validation_split=0.2,
        subset="training",
        batch_size=64,
        image_size=image_size,
        seed=RANDOM_SEED,
    )
    test = tf.keras.utils.image_dataset_from_directory(
        name,
        validation_split=0.2,
        subset="validation",
        batch_size=64,
        image_size=image_size,
        seed=RANDOM_SEED,
    )

    return train, test
