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
PIC_PATH='/Users/kristinaqualben/Desktop/Fun/puppy_pics/data/oxford-pets'
torch.manual_seed(1)

class Net(nn.Module):
    def __init__(self, num_classes:int, resize_n:int):
        super().__init__()
        # Define feature extractor
        self.feature_extractor = nn.Sequential(
           nn.Conv2d(3, resize_n/2, kernel_size=3, padding=1),
           nn.ELU(),
           nn.MaxPool2d(kernel_size=2),
           nn.Conv2d(resize_n/2, resize_n, kernel_size=3, padding=1),
           nn.ELU(),
           nn.MaxPool2d(kernel_size=2),
           nn.Flatten(),
           )
        # Define classifier
        self.classifier = nn.Linear(resize_n**2, num_classes)

    def forward(self, x):
        # Pass input through feature extractor and classifier
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

class Bowzer():
    def __init__(self, resize_n:int = 64):
        self.resize_n = resize_n
        self.train_transforms = (
            transforms
            .Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(45),
                transforms.RandomAutocontrast(),
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
                root=PIC_PATH,
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
                root=PIC_PATH,
                target_types='category',
                download=True,
                split="test",
                transform=self.test_transforms
                )
        self.dataloader_test = (
            DataLoader(
                self.dataset_test,
                shuffle=True, 
                batch_size=16
                )
                )
        
        
    def train(self, epochs:int=3):
        #Define the model
        self.net = Net(num_classes=len(self.dataset_train.classes), resize_n=self.resize_n)
        # Define the loss function
        criterion = nn.CrossEntropyLoss()
        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 0.001)
        for epoch in range(epochs):
            self.running_loss = 0.0
            # Iterate over training batches
            for images, labels in self.dataloader_train:
                self.optimizer.zero_grad()
                outputs = self.net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.running_loss += loss.item()
            epoch_loss = self.running_loss / len(self.dataloader_train)
            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    def evaluate(self):
        metric_precision = Precision(
            task='multiclass',
            num_classes=len(self.dataset_train.classes),
            average='macro')
        metric_recall = Recall(
            task='multiclass',
            num_classes=len(self.dataset_train.classes),
            average='macro')
        self.net.eval()
        with torch.no_grad():
            for images, labels in self.dataloader_test:
                outputs = self.net(images)
                _, preds = torch.max(outputs, 1)
                metric_precision(preds, labels)
                metric_recall(preds, labels)
            self.precision = metric_precision.compute()
            self.recall = metric_recall.compute()
            print(f"Precision: {self.precision}")
            print(f"Recall: {self.recall}")

            
