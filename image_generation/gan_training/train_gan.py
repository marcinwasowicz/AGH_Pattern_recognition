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

if __name__ == "__main__":
    dataset = load_dataset(DATASET_PATH, IMAGE_SIZE)
    dataset = augment_dataset(dataset, AUGMENTATION_FACTOR).batch(BATCH_SIZE)
    noise = np.load(NOISE_SAMPLES_PATH)

    transpose_conv_manager = CheckpointManager(
        discriminator_path=transpose_conv_config["discriminator_path"],
        generator_path=transpose_conv_config["generator_path"],
        noise_samples=noise,
    )
    if "generator" in transpose_conv_config:
        transpose_conv_generator = tf.keras.models.load_model(
            transpose_conv_config["generator"]
        )
    else:
        transpose_conv_generator = build_transpose_conv_generator()

    if "discriminator" in transpose_conv_config:
        transpose_conv_discriminator = tf.keras.models.load_model(
            transpose_conv_config["discriminator"]
        )
    else:
        transpose_conv_discriminator = build_discriminator()

    transpose_config_trainer = GANTrainer(
        generator=transpose_conv_generator,
        discriminator=transpose_conv_discriminator,
        checkpoint_manager=transpose_conv_manager,
        generator_lr=transpose_conv_config["generator_lr"],
        discriminator_lr=transpose_conv_config["discriminator_lr"],
    )

    transpose_config_trainer.train(
        dataset,
        first_epoch=transpose_conv_config["first_epoch"],
        final_epoch=transpose_conv_config["final_epoch"],
        batch_size=BATCH_SIZE,
        checkpoint_interval=CHECKPOINT_INTERVAL,
    )

    upsampling_manager = CheckpointManager(
        discriminator_path=upsampling_config["discriminator_path"],
        generator_path=upsampling_config["generator_path"],
        noise_samples=noise,
    )
    if "generator" in upsampling_config:
        upsampling_generator = tf.keras.models.load_model(
            upsampling_config["generator"]
        )
    else:
        upsampling_generator = build_upsampling_generator()

    if "discriminator" in upsampling_config:
        upsampling_discriminator = tf.keras.models.load_model(
            upsampling_config["discriminator"]
        )
    else:
        upsampling_discriminator = build_discriminator()

    upsampling_trainer = GANTrainer(
        generator=upsampling_generator,
        discriminator=upsampling_discriminator,
        checkpoint_manager=upsampling_manager,
        generator_lr=upsampling_config["generator_lr"],
        discriminator_lr=upsampling_config["discriminator_lr"],
    )

    upsampling_trainer.train(
        dataset,
        first_epoch=upsampling_config["first_epoch"],
        final_epoch=upsampling_config["final_epoch"],
        batch_size=BATCH_SIZE,
        checkpoint_interval=CHECKPOINT_INTERVAL,
    )
