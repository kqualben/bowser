from PIL import Image
import json
import os

def open_image(image_path:str) -> Image:
    return Image.open(image_path)

def save_json(data: dict, directory: str, filename: str) -> None:
    location = os.path.join(directory, filename)
    with open(location, 'w') as f:
        json.dump(data, f)
    print(f"File saved to: {location}")

def open_json(path: str) -> dict:
    print(f"Loading: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    return data