import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class BowzerNet(nn.Module):
    """
    Bowzer Model Module
    """

    def __init__(self, num_classes: int):
        """
        num_classes: number of classes in dataset
        """
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 128),
            nn.Dropout(0.7),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class CustomNet(nn.Module):
    def __init__(self, num_classes: int, resize_n: int):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, resize_n // 2, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(resize_n // 2, resize_n, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )

        self.classifier = nn.Linear(
            resize_n * (resize_n // 4) * (resize_n // 4), num_classes
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
