import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams["savefig.bbox"] = 'tight'

def open_image(image_path:str) -> Image:
    return Image.open(image_path)