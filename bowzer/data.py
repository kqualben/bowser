import torch
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Callable, List, Dict
import matplotlib.pyplot as plt
import numpy as np

from .utils import (
    open_image
)

from .constants import (
    DIR,
    SEED,
    PIN_MEMORY,
    BATCH_SIZE,
    CAT_CLASSES
)

plt.rcParams["savefig.bbox"] = 'tight'
torch.manual_seed(SEED)

class Transform(OxfordIIITPet):
    """
    Load and Transform OxfordIIITPet

    :param resize_n: integer to resize images

    :ivar train_transforms:
    :ivar test_transforms:
    :ivar saved_images:
    """
    def __init__(self, resize_n:int):
        self.resize_n = resize_n
        self.train_transforms = (
            transforms
            .Compose([
                transforms.Resize((self.resize_n,self.resize_n), antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(45),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        )
        self.test_transforms = (
            transforms
            .Compose([
                transforms.Resize((self.resize_n,self.resize_n), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        )
        self._train_set = None
        self._test_set = None
        self.saved_images = []

    @staticmethod
    def _load_transform(transform:torch.Tensor, **kwargs) -> Dataset:
        dataset = OxfordIIITPet(root=DIR, download=True, transform=transform, **kwargs)
        return dataset
    
    @property
    def train_set(self) -> Dataset:
        if self._train_set is None:
            self._train_set = self._load_transform(self.train_transforms)
        return self._train_set
    
    @property
    def test_set(self) -> Dataset:
        if self._test_set is None:
            self._test_set = self._load_transform(self.test_transforms, split = 'test')
        return self._test_set

    @staticmethod
    def _dataloader(data: Dataset, **kwargs) -> Callable:
        return DataLoader(data, shuffle=True, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY, **kwargs)
    
    @staticmethod
    def _custom_collate_fn(batch: int, drop_class_labels: List[str]) -> Tuple:
        filtered_batch = [(img, label) for img, label in batch if label not in drop_class_labels]
        return zip(*filtered_batch) if filtered_batch else ([], [])

    def process(self, ignore_cats: bool = False) -> Tuple[Callable, Callable]:
        if ignore_cats:
            train = self._dataloader(self.train_set, collate_fn = lambda x: self._custom_collate_fn(x, list(CAT_CLASSES.keys())))
            test = self._dataloader(self.test_set, collate_fn = lambda x: self._custom_collate_fn(x, list(CAT_CLASSES.keys())))
        else:
            train = self._dataloader(self.train_set)
            test = self._dataloader(self.test_set)
        return train, test

    @property
    def train_image_paths(self) -> List:
        return self.train_set._images
    
    def get_dog_names(self) -> List[str]:
        classes = self.train_set.classes
        return [x for x in classes if x not in list(CAT_CLASSES.keys())]

    def get_breed_image(self, breeds: List[str]) -> Dict[str,str]:
        paths = {}
        counter = 0
        while counter < len(breeds):
            for im in self.train_image_paths:
                clean = im.as_posix().replace('_',' ').title()
                if any(map(clean.__contains__, breeds)):
                    dog = [dog for dog in breeds if dog in clean][0]
                    paths[dog] = im.as_posix()
                    breeds.remove(dog)
                counter += 1
        return paths

    def show_training_images(self, breed_images: Dict[str,str], save: bool = False) -> None:
        for i, (dog, img) in enumerate(breed_images.items()):
            fig, axes = plt.subplots(ncols=2, squeeze=True)
            img = open_image(img)
            axes[0].imshow(np.asarray(img))
            axes[1].imshow(self.train_transforms(img).squeeze(0).permute(1,2,0))
            axes[0].set_title(f'{dog}', size = 'medium')
            axes[1].set_title('Transformed', size = 'medium')
            fig.tight_layout()
            if save:
                path = f"/tmp/{dog.replace(' ','_').lower()}_transform.png"
                plt.savefig(path)
                self.saved_images.append(path)
            plt.show()
