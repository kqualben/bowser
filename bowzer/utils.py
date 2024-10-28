import json
import logging
import os
from typing import Dict, Literal, List, Tuple

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle


def open_image(image_path: str) -> Image:
    return Image.open(image_path)


def save_json(data: dict, directory: str, filename: str) -> str:
    location = os.path.join(directory, filename)
    with open(location, "w") as f:
        json.dump(data, f)
    print(f"File saved to: {location}")
    return location


def open_json(path: str) -> Dict:
    print(f"Loading: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_pickle(data: dict, directory: str, filename: str) -> str:
    location = os.path.join(directory, filename)
    with open(location, "wb") as f:
        pickle.dump(data, f)
    print(f"File saved to: {location}")
    return location


def open_pickle(path: str) -> Dict:
    print(f"Loading: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def logger(directory: str, filename: str):
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
    for root, _, files in os.walk("model_store", topdown=False):
        for file in files:
            if model_name in root and "." not in file:
                return os.path.join(root, file)


def get_model_settings(model_path: str) -> str:
    if model_path is not None:
        settings_path = os.path.join(
            model_path.split("model_epochs")[0], "model_settings.pkl"
        )
        return settings_path


def get_target_image_dict(root: str = "./images/targets/") -> Dict[str, dict]:
    image_dict = {name: [] for name in os.listdir(root)}
    for folder in image_dict:
        image_dict[folder] = {
            image.split("/")[-1].split(".")[0]: f"{root+folder}/{image}"
            for image in os.listdir(f"{root}/{folder}")
        }
    return image_dict


def compare_model_loss(model_list: List[str], loss: Literal["train", "val"] = "train"):
    compare_models = [(x, f"{get_model_path(x)}_performance.json") for x in model_list]

    loss_target = "avg_loss" if loss == "train" else "avg_val_loss"
    model_avg_losses = {}
    model_accuracy = {}
    for i in range(len(compare_models)):
        model_name, model_performance_path = compare_models[i]
        model_perf = open_json(model_performance_path)
        epoch_avg_loss = []
        for epoch in [x for x in model_perf.keys() if "epoch" in x.lower()]:
            epoch_avg_loss.append(model_perf[epoch][loss_target])
        model_avg_losses[model_name] = epoch_avg_loss
        model_accuracy[model_name] = model_perf["accuracy"] * 100

    model_avg_losses = {
        key: value
        for key, value in sorted(
            model_avg_losses.items(), key=lambda item: len(item[1]), reverse=True
        )
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in model_avg_losses:
        total_loss = model_avg_losses[label]
        epoch_range = range(1, len(total_loss) + 1)
        plt.scatter(x=epoch_range, y=total_loss, label=f"{label}")
        ax.annotate(
            f"Accuracy:\n{model_accuracy[label]:.2f}%",
            xy=(epoch_range[-1], total_loss[-1]),
            xytext=(0, 25),
            textcoords="offset points",
            arrowprops=dict(facecolor="black", arrowstyle="->"),
            horizontalalignment="center",
            verticalalignment="bottom",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{loss.title()} Loss")
    plt.legend()
    plt.title(f"Bowser Model {loss.title()} Loss Comparison")
    fig.tight_layout()
    plt.show()


def compare_model_performance(
    model_list: List[Tuple[str, str]], share_yaxis: bool = False
) -> None:
    compare_models = [
        (x[0], x[1], f"{get_model_path(x[0])}_performance.json") for x in model_list
    ]

    fig, ax = plt.subplots(
        1,
        len(compare_models),
        figsize=(6 * (len(compare_models)), 6),
        sharey=share_yaxis,
    )
    for i in range(len(compare_models)):
        model_name, config_label, model_performance_path = compare_models[i]
        model_perf = open_json(model_performance_path)
        model_epochs = [x for x in model_perf.keys() if "epoch" in x.lower()]
        accuracy = 100 * model_perf["accuracy"]
        model_train_loss = [model_perf[epoch]["avg_loss"] for epoch in model_epochs]
        model_val_loss = [model_perf[epoch]["avg_val_loss"] for epoch in model_epochs]
        ax[i].scatter(
            x=range(1, len(model_epochs) + 1), y=model_train_loss, label="Train Loss"
        )
        ax[i].scatter(
            x=range(1, len(model_epochs) + 1),
            y=model_val_loss,
            label="Val loss",
            color="red",
        )
        ax[i].set_title(f"{model_name} {config_label} Accuracy: {accuracy:.2f}%")
        ax[i].legend()
        ax[i].set_xlabel("Epoch")
        ax[i].set_ylabel("Loss")

    fig.suptitle("Model Loss Comparison", size="large", weight="bold")
    fig.tight_layout()
    plt.show()


def view_model_performance(
    model_name: str,
    save_fig: bool = False,
) -> None:
    model_performance_path = f"{get_model_path(model_name)}_performance.json"
    model_perf = open_json(model_performance_path)
    epoch_keys = [x for x in model_perf.keys() if "epoch" in x.lower()]
    title = f"{model_name}\nPrecision: {model_perf['precision']:.2f} Recall: {model_perf['recall']:.2f}"
    accuracy = 100 * model_perf["accuracy"]
    epoch_avg_loss = [model_perf[epoch]["avg_loss"] for epoch in epoch_keys]
    epoch_val_loss = [model_perf[epoch]["avg_val_loss"] for epoch in epoch_keys]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(x=range(1, len(epoch_keys) + 1), y=epoch_avg_loss, label="Train Loss")
    ax.scatter(
        x=range(1, len(epoch_keys) + 1),
        y=epoch_val_loss,
        label="Val loss",
        color="red",
    )
    ax.set_title(f"Accuracy: {accuracy:.2f}%")
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    fig.suptitle(title, size="large", weight="bold")
    fig.tight_layout()
    if save_fig:
        path = model_performance_path.replace(".json", "_plot.jpg")
        plt.savefig(path)
        print(f"Plot saved to: {path}")
    plt.show()
