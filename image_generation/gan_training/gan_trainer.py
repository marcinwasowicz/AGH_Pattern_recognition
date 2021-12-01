import time

import tensorflow as tf
from tqdm import tqdm


class GANTrainer:
    def __init__(
        self,
        generator,
        discriminator,
        checkpoint_manager,
        discriminator_lr=1e-4,
        generator_lr=1e-4,
    ):
        self.generator = generator
        self.discriminator = discriminator

        self.checkpoint_manager = checkpoint_manager

        self.input_dim = self.generator.layers[0].input_shape[1]
        self.output_shape = self.generator.layers[-1].output_shape[1:]

        self.generator_optimizer = tf.keras.optimizers.Adam(discriminator_lr)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(generator_lr)

        self.loss_function_calculator = tf.keras.losses.BinaryCrossentropy()

    def discriminator_step(self, images, batch_size):
        noise = tf.random.normal([batch_size, self.input_dim])
        generated_images = self.generator(noise)

        with tf.GradientTape() as grad_tape:
            images_output = self.discriminator(images, training=True)
            generated_output = self.discriminator(generated_images, training=True)
            output = tf.concat([images_output, generated_output], axis=0)

            images_labels = tf.ones_like(images_output)
            generated_labels = tf.zeros_like(generated_output)
            labels = tf.concat([images_labels, generated_labels], axis=0)
            labels += tf.random.normal(labels.shape, 0.0, stddev=0.05)

            loss = self.loss_function_calculator(labels, output)

        gradients = grad_tape.gradient(loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(gradients, self.discriminator.trainable_variables)
        )
        return loss

    def generator_step(self, batch_size):
        noise = tf.random.normal([batch_size, self.input_dim])

        with tf.GradientTape() as grad_tape:
            generated_images = self.generator(noise, training=True)
            discriminator_output = self.discriminator(generated_images)

            labels = tf.ones_like(discriminator_output)
            loss = self.loss_function_calculator(labels, discriminator_output)

        gradients = grad_tape.gradient(loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables)
        )
        return loss

    @tf.function
    def step(self, images, batch_size):
        discriminator_loss = self.discriminator_step(images, batch_size)
        generator_loss = self.generator_step(batch_size)
        return discriminator_loss, generator_loss

    def train(
        self,
        dataset: tf.data.Dataset,
        first_epoch,
        final_epoch,
        batch_size,
        checkpoint_interval,
    ):
        for epoch in range(first_epoch, final_epoch):
            discriminator_losses = []
            generator_losses = []
            batch_times = []

            graphic_batch_generator = tqdm(list(dataset), desc=f"Epoch: {epoch}")
            for batch in graphic_batch_generator:
                start = time.time()
                discriminator_loss, generator_loss = self.step(batch, batch_size)

                batch_times.append(time.time() - start)
                discriminator_losses.append(discriminator_loss)
                generator_losses.append(generator_loss)

                epoch_time = sum(batch_times)
                avg_disc_loss = sum(discriminator_losses) / len(discriminator_losses)
                avg_gen_loss = sum(generator_losses) / len(generator_losses)

                graphic_batch_generator.set_postfix(
                    {
                        "time": epoch_time,
                        "discriminator loss": avg_disc_loss.numpy(),
                        "generator_loss": avg_gen_loss.numpy(),
                    }
                )

            if epoch % checkpoint_interval == 0:
                self.checkpoint_manager.dump_gan(
                    epoch,
                    (sum(discriminator_losses) / len(discriminator_losses)).numpy(),
                    (sum(generator_losses) / len(generator_losses)).numpy(),
                    self.discriminator,
                    self.generator,
                )
