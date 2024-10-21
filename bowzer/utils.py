from PIL import Image

def open_image(image_path:str) -> Image:
    return Image.open(image_path)