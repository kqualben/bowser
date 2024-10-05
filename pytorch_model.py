import torch
import torch.nn as nn
from torchvision.datasets import OxfordIIITPet, ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics import Precision, Recall
import matplotlib.pyplot as plt

from torchvision.transforms import v2
from torchvision.io import read_image

plt.rcParams["savefig.bbox"] = 'tight'

torch.manual_seed(1)

class Net(nn.Module):
    def __init__(self, num_classes:int, resize_n:int):
        super().__init__()
        # Define feature extractor
        self.feature_extractor = nn.Sequential(
           nn.Conv2d(3, int(resize_n/2), kernel_size=3, padding=1),
           nn.ELU(),
           nn.MaxPool2d(kernel_size=2),
           nn.Flatten(),
           )
        # Define classifier
        self.classifier = nn.Linear(int(resize_n/2)*int(resize_n/8)*int(resize_n/8), num_classes)

    def forward(self, x):
        # Pass input through feature extractor and classifier
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

class Bowzer():
    def __init__(self, resize_n:int = 128):
        self.resize_n = resize_n
        self.train_transforms = (
            transforms
            .Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(45),
                transforms.ToTensor(),
                transforms.Resize((self.resize_n,self.resize_n))
            ])
        )
        self.test_transforms = (
            transforms
            .Compose([
                transforms.ToTensor(),
                transforms.Resize((self.resize_n,self.resize_n))
            ])
        )
        self.dataset_train = OxfordIIITPet(
                root='./data/oxford-pets',
                target_types='category',
                download=True,
                transform=self.train_transforms
                )
        self.dataloader_train = (
            DataLoader(
                self.dataset_train,
                shuffle=True, 
                batch_size=16
                )
                )
    
        self.dataset_test = OxfordIIITPet(
                root='./data/oxford-pets',
                target_types='category',
                download=True,
                split="test",
                transform=self.test_transforms
                )
        # self.dataset_predict =(
        #       ImageFolder("puppy_pics",
        #                   transform=self.test_transforms
        #                   )
        #       )
        
    def train(self, epochs:int=3):
        #Define the model
        net = Net(num_classes =7, resize_n=self.resize_n)
        # Define the loss function
        criterion = nn.CrossEntropyLoss()
        # Define the optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
        for epoch in range(epochs):
            running_loss = 0.0
            # Iterate over training batches
            for images, labels in self.dataloader_train:
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / len(self.dataloader_train)
            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
