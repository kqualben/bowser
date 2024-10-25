import os
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from .constants import BATCH_SIZE, CAT_CLASSES, DIR, PIN_MEMORY
from .utils import open_image

plt.rcParams["savefig.bbox"] = "tight"


class CustomOxfordIIITPet(OxfordIIITPet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.image_paths = []
        self.labels = []

        for img_name, label in zip(self._images, self._labels):
            img_path = os.path.join(self.root, "images", img_name)
            self.image_paths.append(img_path)
            self.labels.append(label)

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        image_path = self.image_paths[index]
        return image, label, image_path

    def __len__(self):
        return len(self.image_paths)


class Transform:
    """
    Load and Transform OxfordIIITPet

    :param resize_n: integer to resize images

    :ivar train_transforms:
    :ivar test_transforms:
    :ivar saved_images:
    """

    def __init__(self, resize_n: int):
        self.resize_n = resize_n
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize((self.resize_n, self.resize_n), antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(45),
                transforms.RandomGrayscale(0.50),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.test_transforms = transforms.Compose(
            [
                transforms.Resize((self.resize_n, self.resize_n), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self._train_set = None
        self._test_set = None
        self._class_dict = None
        self.saved_images = []

    @staticmethod
    def _load_transform(transform: torch.Tensor, **kwargs) -> Dataset:
        return CustomOxfordIIITPet(
            root=DIR, download=True, transform=transform, **kwargs
        )

    @property
    def train_set(self) -> Dataset:
        if self._train_set is None:
            self._train_set = self._load_transform(self.train_transforms)
        return self._train_set

    @property
    def test_set(self) -> Dataset:
        if self._test_set is None:
            self._test_set = self._load_transform(self.test_transforms, split="test")
        return self._test_set

    @property
    def class_dict(self) -> Dict:
        if self._class_dict is None:
            self._class_dict = self.train_set.class_to_idx
        return self._class_dict

    @property
    def idx_dict(self) -> Dict:
        return {j: k for k, j in self.class_dict.items()}

    @staticmethod
    def _dataloader(data: Dataset, **kwargs) -> Callable:
        return DataLoader(
            data, shuffle=True, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY, **kwargs
        )

    def process(self) -> Tuple[Callable, Callable]:
        train = self._dataloader(self.train_set)
        test = self._dataloader(self.test_set)
        return train, test

    @property
    def train_image_paths(self) -> List:
        return self.train_set._images

    def get_label_idx(self, label: str) -> int:
        return self.class_dict[label]

    def get_idx_label(self, idx: int) -> str:
        return self.idx_dict[idx]

    def get_dog_names(self) -> List[str]:
        return [x for x in self.train_set.classes if x not in list(CAT_CLASSES.keys())]

    def get_breed_image(self, breeds: List[str]) -> Dict[str, str]:
        paths = {}
        counter = 0
        while counter < len(breeds):
            for im in self.train_image_paths:
                clean = im.as_posix().replace("_", " ").title()
                if any(map(clean.__contains__, breeds)):
                    dog = [dog for dog in breeds if dog in clean][0]
                    paths[dog] = im.as_posix()
                    breeds.remove(dog)
                counter += 1
        return paths

    def show_image_transforms(
        self, image_paths: Dict[str, str], train: bool = True, save: bool = False
    ) -> None:
        transformer = self.train_transforms if train else self.test_transforms
        for name, img in image_paths.items():
            fig, axes = plt.subplots(ncols=2, squeeze=True)
            img = open_image(img)
            axes[0].imshow(np.asarray(img))
            axes[1].imshow(transformer(img).squeeze(0).permute(1, 2, 0))
            axes[0].set_title(f"{name}", size="medium")
            axes[1].set_title("Transformed", size="medium")
            fig.tight_layout()
            if save:
                path = f"/tmp/{name.replace(' ','_').lower()}_{'train' if train else 'test'}_transform.png"
                plt.savefig(path)
                self.saved_images.append(path)
            plt.show()
