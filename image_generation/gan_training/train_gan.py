import numpy as np
import tensorflow as tf

from architectures import (
    build_transpose_conv_generator,
    build_discriminator,
    build_upsampling_generator,
)
from dataset import load_dataset, augment_dataset
from gan_training.checkpoint_manager import CheckpointManager
from gan_training.gan_trainer import GANTrainer
from gan_training.gan_training_config.transpose_conv_config import transpose_conv_config
from gan_training.gan_training_config.upsampling_config import upsampling_config

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
FIRST_EPOCH = 1
LAST_EPOCH = 100
CHECKPOINT_INTERVAL = 3
AUGMENTATION_FACTOR = 3

DATASET_PATH = "../dataset/data/penguin"
NOISE_SAMPLES_PATH = "noise_samples/noise_samples.npy"


def train_from_config(config, noise, data, discriminator_callback, generator_callback):
    checkpoint_manager = CheckpointManager(
        discriminator_path=config["discriminator_path"],
        generator_path=config["generator_path"],
        noise_samples=noise,
    )
    if "generator" in config:
        generator = tf.keras.models.load_model(config["generator"])
    else:
        generator = generator_callback()

    if "discriminator" in config:
        discriminator = tf.keras.models.load_model(config["discriminator"])
    else:
        discriminator = discriminator_callback()

    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        checkpoint_manager=checkpoint_manager,
        generator_lr=config["generator_lr"],
        discriminator_lr=config["discriminator_lr"],
    )

    trainer.train(
        data,
        first_epoch=config["first_epoch"],
        final_epoch=config["final_epoch"],
        batch_size=BATCH_SIZE,
        checkpoint_interval=CHECKPOINT_INTERVAL,
    )


if __name__ == "__main__":
    dataset = load_dataset(DATASET_PATH, IMAGE_SIZE)
    dataset = augment_dataset(dataset, AUGMENTATION_FACTOR).batch(BATCH_SIZE)
    noise_samples = np.load(NOISE_SAMPLES_PATH)

    for gan_config, dis_callback, gen_callback in [
        (
            transpose_conv_config,
            lambda: build_discriminator(),
            lambda: build_transpose_conv_generator(),
        ),
        (
            upsampling_config,
            lambda: build_discriminator(),
            lambda: build_upsampling_generator(),
        ),
    ]:
        train_from_config(
            gan_config, noise_samples, dataset, dis_callback, gen_callback
        )
