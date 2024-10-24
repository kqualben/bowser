from PIL import Image
import logging
import json
import os
from typing import Dict


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


def get_target_image_dict(root: str = "./images/testing/") -> Dict[str, dict]:
    image_dict = {name: [] for name in os.listdir(root)}
    for folder in image_dict:
        image_dict[folder] = {
            image.split("/")[-1].split(".")[0]: f"{root+folder}/{image}"
            for image in os.listdir(f"{root}/{folder}")
        }
    return image_dict
