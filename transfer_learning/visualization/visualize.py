import os
import pickle
import sys

from matplotlib import pyplot as plt


def plot_training_results(name, history_plot_dir, history, epochs, max_loss):
    plt.plot(history["acc"])
    plt.plot(history["loss"])
    plt.plot(history["val_acc"])
    plt.plot(history["val_loss"])
    plt.legend(labels=["train accuracy", "train loss", "test accuracy", "test loss"])
    plt.xlim(1, epochs)
    plt.ylim(0.0, max_loss)
    plt.xlabel("epoch")
    plt.ylabel("accuracy/loss")
    plt.title(name.replace("_", " "))
    plt.savefig(f"{history_plot_dir}/{name}.jpg")
    plt.show()
    plt.clf()


if __name__ == "__main__":
    _script, epochs, max_loss = sys.argv

    for training_history_file in os.scandir("histories"):
        if training_history_file.name.startswith("."):
            continue
        network_name = training_history_file.name.split(".")[-2]
        history_obj = pickle.load(open(training_history_file.path, "rb"))

        plot_training_results(
            network_name, "history_plots", history_obj, int(epochs), float(max_loss)
        )
