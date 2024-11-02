import os
from typing import Callable, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import OxfordIIITPet

from .config import ModelSettings
from .constants import CAT_CLASSES, DIR, PIN_MEMORY, SEED
from .utils import open_image

plt.rcParams["savefig.bbox"] = "tight"

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class CustomDataset(OxfordIIITPet):
    """
    Customized the OxfordIIITPet dataset.

    Excludes Cat Breeds.
    __getitem__ returns image, label, and image_path
    """

    def __init__(self, include_cats: bool, *args, **kwargs):
        """
        Params:
        include_cats: bool when True, allow cat breeds to be trained on
        args: see OxfordIIITPet documentation
        kwargs: see OxfordIIITPet documentation
        """
        super().__init__(*args, **kwargs)

        self.image_tensors = []
        self.labels = []
        self.image_paths = []

        if include_cats:
            self.class_dict = self.class_to_idx
        else:
            self.class_dict = {
                breed: idx
                for breed, idx in self.class_to_idx.items()
                if breed not in CAT_CLASSES.keys()
            }

        for idx, (image_path, label) in enumerate(zip(self._images, self._labels)):
            if label in self.class_dict.values():
                image, label = super().__getitem__(idx)
                self.image_tensors.append(image)
                self.labels.append(label)
                self.image_paths.append(image_path.as_posix())
        # ran into issue due to dropping labels from training -> need to zero index class dict and labels
        if not include_cats:
            self.updated_class_labels = {
                breed: (old_label, idx)
                for idx, (breed, old_label) in enumerate(self.class_dict.items())
            }
            new_labels = np.array(self.labels)
            for breed, (old, new) in self.updated_class_labels.items():
                new_labels[new_labels == old] = new
            # reassign labels and class_dict
            self.labels = new_labels.tolist()
            self.class_dict = {
                breed: new for breed, (old, new) in self.updated_class_labels.items()
            }

    def __getitem__(self, index):
        """
        overwrite OxfordIIITPet.__getitem__() since we're excluding specific breeds
        required for customization with OxfordIIITPet Dataset.
        """
        image = self.image_tensors[index]
        label = self.labels[index]
        image_path = self.image_paths[index]
        return image, label, image_path

    def __len__(self):
        """
        overwrite OxfordIIITPet.__len___() since we're excluding specific breeds
        required for customization with OxfordIIITPet Dataset.
        """
        return len(self.image_tensors)


class Transform:
    """
    Load and Transform OxfordIIITPet

    Param:
    model_settings: ModelSettings dataclass
    """

    def __init__(self, model_settings: ModelSettings):
        print(f"Model Settings Info: {model_settings.info}")
        self.model_settings = model_settings
        self.train_transforms = self.model_settings.train_transform
        self.test_transforms = self.model_settings.test_transform
        self.include_cats = self.model_settings.include_cats
        self._trainval_dataset = None
        self._train_val_split = None
        self._train_set = None
        self._val_set = None
        self._test_set = None
        self._class_dict = None
        self.saved_images = []

    @staticmethod
    def _load_transform(transform: torch.Tensor, include_cats: bool, **kwargs):
        """
        method to load and transform CustomDataset.

        Params:
        transform: transformation to apply on dataset
        include_cats: bool when True, allow cat breeds to be trained on
        kwargs: see OxfordIIITPet documentation
        """
        return CustomDataset(
            root=DIR,
            download=True,
            transform=transform,
            include_cats=include_cats,
            **kwargs,
        )

    @property
    def trainval_dataset(self) -> Dataset:
        """
        property which returns the trainval_dataset dataset class attribute
        """
        if self._trainval_dataset is None:
            self._trainval_dataset = self._load_transform(
                include_cats=self.include_cats,
                transform=self.train_transforms,
                split="trainval",
            )
        return self._trainval_dataset

    @staticmethod
    def _split_dataset(
        dataset: Dataset, train_size: float = 0.85
    ) -> Tuple[Dataset, Dataset]:
        """
        method to split a dataset into train, val segments.

        Params:
        dataset: pytorch Dataset
        train_size: float percent of dataset which will be used for training.

        Returns:
        Tuple[0] is the resulting train Dataset
        Tuple[1] is the resulting val Dataset
        """
        split_n = int(train_size * len(dataset))
        train, val = random_split(
            dataset,
            [split_n, len(dataset) - split_n],
            generator=torch.Generator().manual_seed(SEED),
        )
        return train, val

    @property
    def train_val_split(self) -> Dataset:
        """
        property which returns the train_val_split datasets
        """
        if self._train_val_split is None:
            self._train_val_split = self._split_dataset(self.trainval_dataset)
        return self._train_val_split

    @property
    def train_set(self) -> Dataset:
        """
        property which returns the train_set dataset class attribute
        """
        if self._train_set is None:
            train, val = self.train_val_split
            self._train_set = train
            print(f"Train rows: {len(self._train_set)}")
        return self._train_set

    @property
    def val_set(self) -> Dataset:
        """
        property which returns the val_set dataset class attribute
        """
        if self._val_set is None:
            train, val = self.train_val_split
            self._val_set = val
            print(f"Val rows: {len(self._val_set)}")
        return self._val_set

    @property
    def test_set(self) -> Dataset:
        """
        property which returns the test_set dataset class attribute
        """
        if self._test_set is None:
            self._test_set = self._load_transform(
                include_cats=self.include_cats,
                transform=self.test_transforms,
                split="test",
            )
            print(f"Test rows: {len(self._test_set)}")
        return self._test_set

    @property
    def class_dict(self) -> Dict:
        """
        property which returns a dictionary mapping breed name to their index label.

        Return:
        Dict keys are are str breed names and values integer (labels)
        """
        if self._class_dict is None:
            self._class_dict = self.train_set.dataset.class_dict
        return self._class_dict

    @property
    def idx_dict(self) -> Dict:
        """
        property which returns class_dict with keys and values are flipped.
        useful since labels in the dataset are integers rather that breed names.

        Return:
        Dict keys are integer (labels) and values are str breed names
        """
        return {j: k for k, j in self.class_dict.items()}

    def _dataloader(self, data: Dataset, **kwargs):
        """
        helper function to get PyTorch Dataloader

        Params:
        data: PyTorch Dataset. (train, val, or test)
        kwargs: see PyTorch DataLoader docs
        """
        return DataLoader(
            data,
            batch_size=self.model_settings.batch_size,
            pin_memory=PIN_MEMORY,
            **kwargs,
        )

    def process(self) -> Tuple:
        """
        function to load and process data for training

        Returns:
        Tuple of dataloaders. train, val, test.
        """
        train = self._dataloader(self.train_set, shuffle=True, drop_last=True)
        val = self._dataloader(self.val_set, shuffle=False, drop_last=False)
        test = self._dataloader(self.test_set, shuffle=False, drop_last=False)
        return train, val, test

    @property
    def num_classes(self) -> int:
        return len(self.class_dict)

    @property
    def train_image_paths(self) -> List:
        """
        property which returns a list of paths to images used in training
        """
        return self.train_set.dataset.image_paths

    def get_label_idx(self, label: str) -> int:
        """
        function to get the given label given it's breed name

        Params:
        label: str breed name

        Returns:
        int: index
        """
        return self.class_dict[label]

    def get_idx_label(self, idx: int) -> str:
        """
        function to get the given breed name given it's label

        Params:
        idx: int index

        Returns:
        str: label i.e. breed name
        """
        return self.idx_dict[idx]

    def get_dog_names(self) -> List[str]:
        """
        function to return a list of dog breed names, defined by what is not a breed in CAT_CLASSES.
        """
        return [x for x in self.class_dict.keys() if x not in list(CAT_CLASSES.keys())]

    def get_breed_image(self, breeds: List[str]) -> Dict[str, str]:
        """
        function which generates a dictionary containing an image path for each given breed

        Params:
        breeds: List of breed names which exist in the OxfordPet dataset

        Returns:
        Dictionary where keys are the breed names and paths correspond to their image locations.
        """
        paths = {}
        counter = 0
        while counter < len(breeds):
            for im in self.train_image_paths:
                clean = im.replace("_", " ").title()
                if any(map(clean.__contains__, breeds)):
                    dog = [dog for dog in breeds if dog in clean][0]
                    paths[dog] = im
                    breeds.remove(dog)
                counter += 1
        if len(breeds) > 0:
            print(f"Excluded Breeds: {breeds}")
        return paths

    def show_image_transforms(
        self, image_paths: Dict[str, str], train: bool = True, save: bool = False
    ) -> None:
        """
        function to show images and an example of their transformation

        Params:
        image_paths: dict where the key is the breed name and value is the path to it's image
        train: bool when True, apply train transformation. else apply test transformation.
        save: bool when True, save the image to the images/training_data_samples/ dir.
        """
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
                path = f"./images/training_data_samples/{name.replace(' ','_').lower()}_{'train' if train else 'test'}_transform.png"
                plt.savefig(path)
                self.saved_images.append(path)
            plt.show()
