import torch
import torch.nn as nn
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics import Precision, Recall
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, Literal
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

plt.rcParams["savefig.bbox"] = 'tight'
DIR ='/Volumes/KCQDrive/projects'

torch.manual_seed(42)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {DEVICE}')
torch.device(DEVICE)

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
    

class Bowzer():
    def __init__(self, resize_n:int = 64):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
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
        self.train_data = OxfordIIITPet(root=DIR, download=True, transform=self.train_transforms)
        self.test_data = OxfordIIITPet(root=DIR, download=True,split="test", transform=self.test_transforms)
        #extract num_classes
        self.num_classes = len(set(self.train_data._labels))
        #dataloaders
        self.dataloader_train = DataLoader(self.train_data, shuffle=True, batch_size=self.num_classes)
        self.dataloader_test = DataLoader(self.test_data, shuffle=True, batch_size=self.num_classes)

        self.writer_ = None
        self.writer_path = f'{DIR}/bowzer/runs/trainer_{self.timestamp}'

    @property
    def writer(self):
        return self.writer_

    
    def show_img(self, image, transform):
        plt.imshow(transform(Image.open(image)).squeeze(0).permute(1,2,0))
        plt.show()

    def train(self, epochs:int=3):
        #Define the model
        self.net = Net(num_classes=self.num_classes, resize_n=self.resize_n)
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
    
    def train_epoch(self, epoc_idx):
        running_loss, last_loss = 0, 0
        for i, data in enumerate(self.dataloader_train):
            images, labels = data
            self.optimizer.zero_grad()
            outputs = self.net(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000
                print(f"batch {i + 1} loss: {last_loss}")
                self.writer.add_scalar('Loss/Train', last_loss, (epoch_index * len(self.dataloader_train) + i + 1))
                running_loss = 0
        return last_loss

    def train_eval(self, epochs:int = 5):
        best_loss = 1_000_000
        #Define the model
        self.net = Net(num_classes=self.num_classes, resize_n=self.resize_n)
        # Define the loss function
        self.criterion = nn.CrossEntropyLoss()
        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 0.001)
        for epoch in range(epochs):
            print(f"EPOCH {epoch + 1}:")
            self.net.train(True)
            avg_loss = self.train_epoch(epoch)

            running_loss = 0.0
            self.net.eval()
            with torch.no_grad():
                for i, data in enumerate(self.dataloader_test):
                    images, labels = data
                    outputs = self.net(images)
                    loss = self.criterion(outputs, labels)
                    running_loss += loss
            avg_test_loss = running_loss / (i + 1)
            print(f"LOSS Train {avg_loss} Test: {avg_test_loss}")

            self.writer.add_scalars('Training vs. Test Loss', {'Training': avg_loss, 'Test': avg_test_loss}, epoch + 1)
            self.writer.flush()
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                torch.save(self.net.state_dict(), f'{DIR}/bowzer/runs/trainer_{self.timestamp}/model_{self.timestamp}_{epoch}')


def evaluate(model, num_classes, dataloader_test, average: Literal['macro','micro','weighted',None]):
    metric_precision = Precision(
        task='multiclass',
        num_classes=num_classes,
        average=average)
    metric_recall = Recall(
        task='multiclass',
        num_classes=num_classes,
        average=average)
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader_test:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            metric_precision(preds, labels)
            metric_recall(preds, labels)
    precision = metric_precision.compute()
    recall = metric_recall.compute()
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    return precision, recall

            
