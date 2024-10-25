from dataclasses import dataclass
import torchvision
from typing import Type


@dataclass
class ModelSettings:
    info: str
    epochs: int
    batch_size: int
    resize_n: int
    train_transform: torchvision.transforms.Compose
    test_transform: torchvision.transforms.Compose

    @classmethod
    def from_dict(cls: Type["ModelSettings"], config="dict") -> "ModelSettings":
        return cls(
            info=config["info"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            resize_n=config["resize_n"],
            train_transform=config["train_transform"],
            test_transform=config["test_transform"],
        )
