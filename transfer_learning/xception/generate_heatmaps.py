import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from dataset import (
    load_dataset,
    LARGE_IMAGE_SIZE,
    LASAGNE_MUSAKA_TART,
    DUMBELL_KETTLEBELL_BAREBELL,
)

IMAGE_COUNT = 20


def load_xception(path):
    xception = tf.keras.models.load_model(path)
    feature_extractor = tf.keras.models.Model(
        inputs=xception.input, outputs=xception.layers[-3].output
    )
    classifier = tf.keras.models.Model(
        inputs=xception.layers[-1].input, outputs=xception.layers[-1].output
    )

    return xception, feature_extractor, classifier


def generate_heatmap(image, feature_extractor, classifier, label):
    features = feature_extractor(np.array([image])).numpy()[0]
    neurons = classifier.layers[-1].variables[0].numpy()[:, label]
    heatmap = np.apply_along_axis(lambda x: np.dot(x, neurons), axis=2, arr=features)
    return heatmap


def plot_heatmaps(name, correct, incorrect, feature_extractor, classifier):
    for count, (image, label) in enumerate(correct):
        plt.suptitle(name)
        plt.subplot(1, 2, 1)
        plt.imshow(image.numpy().astype("uint8"))
        plt.title(f"Correctly classified as {test.class_names[label]}")
        plt.subplot(1, 2, 2)
        plt.imshow(generate_heatmap(image, feature_extractor, classifier, label))
        plt.title("heatmap")
        plt.savefig(f"heatmaps/{name}/correct_{count}.jpg")
        plt.show()
        plt.clf()

    for count, (image, label, classification) in enumerate(incorrect):
        plt.suptitle(name)
        plt.subplot(1, 3, 1)
        plt.imshow(image.numpy().astype("uint8"))
        plt.title(
            f"Incorrectly classified as {test.class_names[classification]}.\nBelongs to class: {test.class_names[label]}",
            fontdict={"fontsize": 8},
        )
        plt.subplot(1, 3, 2)
        plt.imshow(
            generate_heatmap(image, feature_extractor, classifier, classification)
        )
        plt.title(
            f"heatmap for class {test.class_names[classification]}",
            fontdict={"fontsize": 8},
        )
        plt.subplot(1, 3, 3)
        plt.imshow(generate_heatmap(image, feature_extractor, classifier, label))
        plt.title(
            f"heatmap for class {test.class_names[label]}", fontdict={"fontsize": 8}
        )
        plt.savefig(f"heatmaps/{name}/incorrect_{count}.jpg")
        plt.show()
        plt.clf()


if __name__ == "__main__":
    for network_name, dataset in [
        ("lasagne_musaka_tart_xception", LASAGNE_MUSAKA_TART),
        ("lasagne_musaka_tart_fine_tuned_xception", LASAGNE_MUSAKA_TART),
        ("dumbell_kettlebell_barebell_xception", DUMBELL_KETTLEBELL_BAREBELL),
        (
            "dumbell_kettlebell_barebell_fine_tuned_xception",
            DUMBELL_KETTLEBELL_BAREBELL,
        ),
    ]:
        _train, test = load_dataset(dataset, LARGE_IMAGE_SIZE)
        xception, feature_extractor, classifier = load_xception(
            f"models/{network_name}"
        )

        correctly_classified = []
        incorrectly_classified = []

        for images, labels in test.take(1):
            for i in range(IMAGE_COUNT):
                label = labels[i].numpy()
                classification = np.argmax(xception(np.array([images[i]])))

                if label == classification:
                    correctly_classified.append((images[i], label))
                else:
                    incorrectly_classified.append((images[i], label, classification))

        plot_heatmaps(
            network_name.replace("_", " "),
            correctly_classified,
            incorrectly_classified,
            feature_extractor,
            classifier,
        )
