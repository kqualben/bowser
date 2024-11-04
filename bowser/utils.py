import json
import logging
import os
import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from PIL import Image


def open_image(image_path: str) -> Image:
    """
    function to open an image given it's path.

    :param str image_path: path to image file.

    :return PIL.Image:
    """
    return Image.open(image_path)


def save_json(data: dict, directory: str, filename: str) -> str:
    """
    function to save data dictionary to given directory/filename.

    :param dict data: dictionary
    :param str directory: root directory where file will be saved.
    :param str filename: name and file type.

    :return str: path to where file has been saved.
    """
    location = os.path.join(directory, filename)
    with open(location, "w") as f:
        json.dump(data, f)
    print(f"File saved to: {location}")
    return location


def open_json(path: str) -> Dict:
    """
    function to open an json file given it's path.

    :param str path: path to json file.

    :return dict:
    """
    print(f"Loading: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_pickle(data, directory: str, filename: str) -> str:
    """
    function to save data object to given directory/filename.

    :param pytorch.object data: pytorch model object
    :param str directory: root directory where file will be saved.
    :param str filename: name and file type.

    :return str: path to where file has been saved.
    """
    location = os.path.join(directory, filename)
    with open(location, "wb") as f:
        pickle.dump(data, f)
    print(f"File saved to: {location}")
    return location


def open_pickle(path: str) -> Dict:
    """
    function to open an pickle file given it's path.

    :param str path: path to pickle file.

    :return dict:
    """
    print(f"Loading: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def logger(directory: str, filename: str):
    """
    function to spin up logger module which gets saved to given directory/filename.

    :param str directory: root directory where file will be saved
    :param str filename: name and file type

    :return logging.object:
    """
    logging.basicConfig(
        filename=f"{directory}/{filename}",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return logger


def get_model_path(model_name: str) -> str:
    """
    function to get model within 'model_store' directory.

    :param str model_name: name of subfolder within model_store/training_YYYYMMDD

    :return str: path to PyTorch model.
    """
    for root, _, files in os.walk("../model_store", topdown=False):
        for file in files:
            if model_name in root and "." not in file:
                return os.path.join(root, file)


def get_model_settings(model_path: str) -> str:
    """
    function to get model model settings for a given model_path within 'model_store' directory.
    model settings are generally saved as 'model_settings.pkl'
    use in conjunction with <get_model_path>.

    :param str model_path: path to a model object.

    :return str: path to a model's 'model_settings.pkl' file.
    """
    if model_path is not None:
        settings_path = os.path.join(
            model_path.split("model_epochs")[0], "model_settings.pkl"
        )
        return settings_path


def get_target_image_dict(root: str = "../images/targets/") -> Dict[str, dict]:
    """
    function to package up target image names and paths in one consolidated dictionary.

    :param str root: relative path to image directory

    :return dict: keys: target_name values: dict of file label and path
    """
    image_dict = {name: [] for name in os.listdir(root)}
    for folder in image_dict:
        image_dict[folder] = {
            image.split("/")[-1].split(".")[0]: f"{root+folder}/{image}"
            for image in os.listdir(f"{root}/{folder}")
        }
    return image_dict


def view_model_performance(
    model_name: str,
    save_fig: bool = False,
) -> None:
    """
    function to plot model performance i.e. train vs validation loss.

    :param str model_name: name of subfolder within model_store/training_YYYYMMDD
    :param bool save_fig: when True, save the generated image to the model's relative folder.
    """
    model_performance_path = f"{get_model_path(model_name)}_performance.json"
    model_perf = open_json(model_performance_path)
    title = f"{model_name}"
    accuracy = 100 * model_perf["accuracy"]
    train_losses = model_perf["train_losses"]
    val_losses = model_perf["val_losses"]
    train_accuracies = [x * 100 for x in model_perf["train_accuracies"]]
    val_accuracies = [x * 100 for x in model_perf["val_accuracies"]]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    xrange = range(1, len(train_losses) + 1)
    ax[0].plot(xrange, train_losses, label="Train Loss", marker=".")
    ax[0].plot(xrange, val_losses, label="Val Loss", marker=".", color="red")
    ax[0].set_title(
        f"Precision: {model_perf['precision']:.2f} Recall: {model_perf['recall']:.2f}"
    )
    ax[0].legend()
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")

    ax[1].plot(xrange, train_accuracies, label="Train", marker=".")
    ax[1].plot(xrange, val_accuracies, label="Val", marker=".", color="red")
    ax[1].set_title(f"Accuracy: {accuracy:.2f}%")
    ax[1].legend()
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")

    fig.suptitle(title, size="large", weight="bold")
    fig.tight_layout()
    if save_fig:
        path = model_performance_path.replace(".json", "_plot.jpg")
        plt.savefig(path)
        print(f"Plot saved to: {path}")
    plt.show()


def compare_model_performance(
    model_list: List[Tuple[str, str]], share_yaxis: bool = False
) -> None:
    """
    function to plot and compare models, specifically train vs validation loss.

    :param list model_list: list of model names to compare within the same figure.
    :param bool share_yaxis: when True, all subplots' yaxis will be shared. useful for aligning results on different scales.
    """
    compare_models = [
        (x[0], x[1], f"{get_model_path(x[0])}_performance.json") for x in model_list
    ]

    fig, ax = plt.subplots(
        2,
        len(compare_models),
        figsize=(6 * (len(compare_models)), 6 * 2),
        sharey=share_yaxis,
    )
    for i in range(len(compare_models)):
        model_name, config_label, model_performance_path = compare_models[i]
        model_perf = open_json(model_performance_path)
        accuracy = 100 * model_perf["accuracy"]
        train_losses = model_perf["train_losses"]
        val_losses = model_perf["val_losses"]
        train_accuracies = [x * 100 for x in model_perf["train_accuracies"]]
        val_accuracies = [x * 100 for x in model_perf["val_accuracies"]]
        xrange = range(1, len(train_losses) + 1)
        ax[0][i].plot(xrange, train_losses, label="Train Loss", marker=".")
        ax[0][i].plot(xrange, val_losses, label="Val Loss", marker=".", color="red")
        ax[0][i].set_title(
            f"{model_name} {config_label}\nPrecision: {model_perf['precision']:.2f} Recall: {model_perf['recall']:.2f}"
        )
        ax[0][i].legend()
        ax[0][i].set_xlabel("Epoch")
        ax[0][i].set_ylabel("Loss")

        ax[1][i].plot(xrange, train_accuracies, label="Train")
        ax[1][i].plot(xrange, val_accuracies, label="Val")
        ax[1][i].set_title(f"Accuracy: {accuracy:.2f}%")
        ax[1][i].legend()
        ax[1][i].set_xlabel("Epoch")
        ax[1][i].set_ylabel("Accuracy")

    fig.suptitle("Model Loss Comparison", size="large", weight="bold")
    fig.tight_layout()
    plt.show()
