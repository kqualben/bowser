from dataclasses import dataclass
from torchvision.transforms import v2
from typing import Type


@dataclass
class ModelSettings:
    """
    Dataclass to manage config settings.
    """

    info: str
    epochs: int
    learning_rate: float
    batch_size: int
    resize_n: int
    train_transform: v2.Compose
    test_transform: v2.Compose

    @classmethod
    def from_dict(cls: Type["ModelSettings"], config="dict") -> "ModelSettings":
        """
        class method to extract values from config dictionary.
        """
        return cls(
            info=config["info"],
            epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            resize_n=config["resize_n"],
            train_transform=config["train_transform"],
            test_transform=config["test_transform"],
        )
