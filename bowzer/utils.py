from PIL import Image
import logging
import json
import os

def open_image(image_path:str) -> Image:
    return Image.open(image_path)

def save_json(data: dict, directory: str, filename: str) -> str:
    location = os.path.join(directory, filename)
    with open(location, 'w') as f:
        json.dump(data, f)
    print(f"File saved to: {location}")
    return location

def open_json(path: str) -> dict:
    print(f"Loading: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def logger(directory: str, filename: str):
    logging.basicConfig(filename=f"{directory}/{filename}", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return logger