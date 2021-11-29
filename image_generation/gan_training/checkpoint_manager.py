import os

from matplotlib import pyplot as plt


class CheckpointManager:
    def __init__(self, discriminator_path, generator_path, noise_samples):
        self.discriminator_path = discriminator_path
        self.generator_path = generator_path
        self.noise_samples = noise_samples

    def dump_discriminator(self, epoch, discriminator):
        path = f"{self.discriminator_path}/discriminator_{epoch}"
        discriminator.save(path)

    def dump_generator(self, epoch, generator):
        generator_path = f"{self.generator_path}/generator_{epoch}"
        images_path = f"{self.generator_path}/generator_{epoch}_images"

        os.makedirs(images_path, exist_ok=True)
        generator.save(generator_path)

        images = generator(self.noise_samples)
        for idx, image in enumerate(images.numpy()):
            image_name = f"image_{idx}"
            plt.imshow(image)
            plt.title(image_name)
            plt.savefig(f"{images_path}/{image_name}.jpg")
            plt.clf()

    def dump_gan(self, epoch, discriminator, generator):
        self.dump_generator(epoch, generator)
        self.dump_discriminator(epoch, discriminator)
