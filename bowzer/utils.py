import json
import logging
import os
from typing import Dict, Literal

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


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


def get_models(
    model_store_path: str = "model_store/trained_20241023/",
) -> Dict[str, str]:
    model_dirs = [x for x in os.listdir(model_store_path)][::-1]
    models = {}
    for model in model_dirs:
        dir_files = os.listdir(f"{model_store_path}{model}")
        perf_filename = [
            x for x in dir_files if "epoch" in x and "performance" not in x
        ][0]
        models[perf_filename] = f"{model_store_path}{model}/{perf_filename}"
    return models


def get_target_image_dict(root: str = "./images/targets/") -> Dict[str, dict]:
    image_dict = {name: [] for name in os.listdir(root)}
    for folder in image_dict:
        image_dict[folder] = {
            image.split("/")[-1].split(".")[0]: f"{root+folder}/{image}"
            for image in os.listdir(f"{root}/{folder}")
        }
    return image_dict


def compare_model_loss(model_dict: dict, loss: Literal["train", "val"] = "train"):
    loss_target = "avg_loss" if loss == "train" else "avg_val_loss"
    model_avg_losses = {}
    for label in model_dict:
        model_perf = open_json(f"{model_dict[label]}_performance.json")
        epoch_avg_loss = []
        for epoch in [x for x in model_perf.keys() if "epoch" in x.lower()]:
            epoch_avg_loss.append(model_perf[epoch][loss_target])
        model_avg_losses[label] = {
            "loss": epoch_avg_loss,
            "accuracy": model_perf["accuracy"] * 100,
        }

    fig, ax = plt.subplots(figsize=(10, 6))
    for label in model_avg_losses:
        total_loss = model_avg_losses[label]["loss"]
        acc = model_avg_losses[label]["accuracy"]
        epoch_range = range(1, len(total_loss) + 1)
        plt.scatter(x=epoch_range, y=total_loss, label=f"{label}")
        ax.annotate(
            f"Accuracy:\n{acc:.2f}%",
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


def view_model_performance(
    model_name: str,
    model_performance_path: str,
    save_fig: bool = False,
    show_batch_losses: bool = False,
) -> None:
    subplots = 1
    plot_batch_losses = False
    model_perf = open_json(model_performance_path)
    epoch_keys = [x for x in model_perf.keys() if "epoch" in x.lower()]
    title = f"{model_name}\nPrecision: {model_perf['precision']:.2f} Recall: {model_perf['recall']:.2f}"
    if (
        "loss_list" in model_perf[epoch_keys[0]].keys()
        and show_batch_losses is not False
    ):
        plot_batch_losses = True
        subplots += 2

    accuracy = 100 * model_perf["accuracy"]
    epoch_avg_loss = []
    epoch_val_loss = []
    fig, ax = plt.subplots(1, subplots, figsize=(6 * (subplots), 6))
    for idx, epoch in enumerate(epoch_keys):
        epoch_avg_loss.append(model_perf[epoch]["avg_loss"])
        epoch_val_loss.append(model_perf[epoch]["avg_val_loss"])
        if plot_batch_losses:
            epoch_losses = model_perf[epoch]["loss_list"]
            ax[1].scatter(
                x=np.linspace(1, len(epoch_losses), len(epoch_losses)).astype(int),
                y=epoch_losses,
                label=epoch,
                marker=["o" if idx < 20 else "*" if idx >= 20 and idx < 40 else "s"][0],
            )
            ax[2].hist(
                epoch_losses,
                alpha=np.linspace(0.25, 0.75, len(epoch_keys)).astype(float)[idx],
                label=epoch,
            )
    epoch_range = range(1, len(epoch_avg_loss) + 1)
    if plot_batch_losses:
        ax_ph = ax[0]
    else:
        ax_ph = ax
    ax_ph.scatter(x=epoch_range, y=epoch_avg_loss, label="Train Loss")
    ax_ph.scatter(x=epoch_range, y=epoch_val_loss, label="Val loss", color="red")
    ax_ph.set_title(f"Accuracy: {accuracy:.2f}%")
    ax_ph.legend()
    ax_ph.set_xlabel("Epoch")
    ax_ph.set_ylabel("Loss")
    if plot_batch_losses:
        ax[1].set_title("Batch Loss")
        ax[2].set_title("Loss Distribution")
        ax[1].set_xlabel("Batch")
        ax[2].set_ylabel("Loss")
        ax[2].set_xlabel("Loss")
        ax[2].set_ylabel("Frequency")
        lines, labels = (
            ax[1].get_legend_handles_labels()[0],
            ax[1].get_legend_handles_labels()[1],
        )
        fig.legend(
            lines,
            labels,
            loc="upper left",
            bbox_to_anchor=(1, 0.86),
            ncol=(round(len(epoch_keys) ** (1 / 4)) if len(epoch_keys) > 10 else 1),
        )

    fig.suptitle(title, size="large", weight="bold")
    fig.tight_layout()
    if save_fig:
        path = model_performance_path.replace(".json", "_plot.jpg")
        plt.savefig(path)
        print(f"Plot saved to: {path}")
    plt.show()
