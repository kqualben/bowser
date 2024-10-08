import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics import Precision, Recall
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from torchvision.transforms import v2
from torchvision.io import read_image

plt.rcParams["savefig.bbox"] = 'tight'
PIC_PATH ='/Users/kristinaqualben/Desktop/Fun/puppy_pics/data/oxford-pets'
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


class preprocessDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        augmented_image = self.transform(image)
        return augmented_image, target
    

class Bowzer():
    def __init__(self, resize_n:int = 64):
        self.resize_n = resize_n
        weights = torchvision.models.resnet.ResNet34_Weights.DEFAULT
        self.raw_train_data = OxfordIIITPet(root=PIC_PATH, target_types='category', download=True)
        self.raw_test_data = OxfordIIITPet(root=PIC_PATH,target_types='category',download=True,split="test")

        # self.train_transforms = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomRotation(45),
        #     transforms.RandomAutocontrast(),
        #     transforms.ToTensor(),
        #     transforms.Resize((self.resize_n,self.resize_n)),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        #     ])
        # self.test_transforms = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((self.resize_n,self.resize_n)),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        #     ])

        self.train_data = preprocessDataset(self.raw_train_data,  weights.transforms())
        self.test_data = preprocessDataset(self.raw_test_data,  weights.transforms())
        self.train_data, self.val_data = train_test_split(self.train_data, test_size=0.2, random_state=0)
        self.dataloader_train = DataLoader(self.train_data, shuffle=True, batch_size=64)
        self.dataloader_val = DataLoader(self.val_data, shuffle=True, batch_size=64)
        self.dataloader_test = DataLoader(self.test_data, shuffle=True, batch_size=64)

    def create_grid(self, raw_train_data, train_data):
        raw_sample = []
        train_sample = []

        rw = iter(raw_train_data)
        tr = iter(train_data)

        for i in range(16):
            raw_sample.append(next(rw)[0])
            train_sample.append(next(tr)[0])

        grid1 = torchvision.utils.make_grid([transforms.Resize((224,224),antialias=True)(transforms.ToTensor()(i)) for i in raw_sample], nrow=8)
        grid2 = torchvision.utils.make_grid([i for i in train_sample], nrow=8)
        return grid1, grid2

    @staticmethod 
    def view_grid(grid):
        return transforms.ToPILImage()(grid)
        

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

            
