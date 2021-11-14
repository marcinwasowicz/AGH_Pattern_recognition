import os
import pickle

from matplotlib import pyplot as plt


def plot_training_results(name, history_plot_dir, history, test_scores):
    plt.plot(history["acc"])
    plt.plot(history["loss"])
    plt.plot([test_scores[1] for _ in history["acc"]])
    plt.plot([test_scores[0] for _ in history["acc"]])
    plt.legend(labels=["train accuracy", "train loss", "test accuracy", "test loss"])
    plt.xlim(1, 150)
    plt.ylim(0.0, 3.0)
    plt.xlabel("epoch")
    plt.ylabel("accuracy/loss")
    plt.title(name.replace("_", " "))
    plt.savefig(f"{history_plot_dir}/{name}.jpg")
    plt.show()
    plt.clf()


if __name__ == "__main__":

    for training_history_file, test_score_file in zip(
        os.scandir("training_histories"), os.scandir("test_scores")
    ):
        network_name = training_history_file.name.split(".")[-2]
        history_obj = pickle.load(open(training_history_file.path, "rb"))
        test_scores_obj = pickle.load(open(test_score_file.path, "rb"))

        plot_training_results(
            network_name, "training_history_plots", history_obj, test_scores_obj
        )
