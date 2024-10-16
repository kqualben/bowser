import torch.nn as nn

class Net(nn.Module):
    def __init__(self, num_classes:int, resize_n:int):
        super().__init__()
        self.feature_extractor = nn.Sequential(
           nn.Conv2d(3, resize_n//2, kernel_size=3, padding=1),
           nn.ELU(),
           nn.MaxPool2d(kernel_size=2),
           nn.Conv2d(resize_n//2, resize_n, kernel_size=3, padding=1),
           nn.ELU(),
           nn.MaxPool2d(kernel_size=2),
           nn.Flatten(),
           )
        
        self.classifier = nn.Linear(resize_n*(resize_n//4)*(resize_n//4), num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x