import torch
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Tuple

from .constants import (
    DIR,
    SEED,
    PIN_MEMORY,
    BATCH_SIZE
)

torch.manual_seed(SEED)

class Dataset():
    """
    Load and Transform OxfordIIITPet

    :param resize_n: integer to resize images

    :ivar train_transforms:
    :ivar test_transforms:
    """
    def __init__(self, resize_n:int):
        self.resize_n = resize_n
        self.train_transforms = (
            transforms
            .Compose([
                transforms.RandomHorizontalFlip(),
                #transforms.RandomAutocontrast(),
                transforms.RandomRotation(45),
                transforms.ToTensor(),
                transforms.Resize((self.resize_n,self.resize_n), antialias=True),
                #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        )
        self.test_transforms = (
            transforms
            .Compose([
                transforms.ToTensor(),
                transforms.Resize((self.resize_n,self.resize_n)),
                #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        )
        self._num_classes = None
        self._train_set = None
        self._test_set = None

    @staticmethod
    def _load_transform(transform, **kwargs):
        return OxfordIIITPet(root=DIR, download=True, transform=transform, **kwargs)
    
    @property
    def train_set(self):
        if self._train_set is None:
            self._train_set = self._load_transform(self.train_transforms)
        return self._train_set
    
    @property
    def test_set(self):
        if self._test_set is None:
            self._test_set = self._load_transform(self.test_transforms, split = 'test')
        return self._test_set

    @staticmethod
    def _dataloader(data):
        return DataLoader(data, shuffle=True, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY)

    def process(self) -> Tuple:
        train = self._dataloader(self.train_set)
        self.num_classes = len(set(self.train_set._labels))
        test = self._dataloader(self.test_set)
        return train, test

            
