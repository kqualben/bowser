import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics import Precision, Recall
import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
from tqdm import tqdm
from torchvision.transforms import v2
from torchvision.io import read_image
from PIL import Image
from typing import Dict, Literal

plt.rcParams["savefig.bbox"] = 'tight'
DIR ='/Volumes/KCQDrive/projects'
DATASET = 'stanford_dogs'
torch.manual_seed(1)

class Net(nn.Module):
    def __init__(self, num_classes:int, resize_n:int):
        super().__init__()
        # Define feature extractor
        self.feature_extractor = nn.Sequential(
            #input image: 3 * resize_n * resize_n
           nn.Conv2d(3, resize_n//2, kernel_size=3, padding=1), #resize/2 output filters
           #resize_n/2 * resize_n * resize_n
           nn.ELU(),
           nn.MaxPool2d(kernel_size=2), #cut h and w in half
           #resize_n/2 * resize_n/2 * resize_n/2
           nn.Conv2d(resize_n//2, resize_n, kernel_size=3, padding=1), #resize_n output filters
           #resize_n * resize_n/2 * resize_n/2
           nn.ELU(),
           nn.MaxPool2d(kernel_size=2), #cut h and w in half
           #resize_n * (resize_n/2)/2 * (resize_n/2)/2
           #resize_n * resize_n/4 * resize_n/4
           nn.Flatten(),
           )
        # Define classifier
        #resize_n * resize_n/4 * resize_n/4
        self.classifier = nn.Linear(resize_n*(resize_n//4)*(resize_n//4), num_classes)

    def forward(self, x):
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
        
        self.num_classes = len(set(self.train_data._labels))
        
        #weights = torchvision.models.resnet.ResNet50_Weights.DEFAULT
        #self.raw_train_data, self.train_info = tfds.load(DATASET, split='train', shuffle_files=True, data_dir = f"{DIR}/tensorflow_datasets/", with_info=True)
        #self.raw_test_data, self.test_info = tfds.load(DATASET, split='test', shuffle_files=True, data_dir = f"{DIR}/tensorflow_datasets/", with_info=True)

        #self.raw_train_data = tfds.as_numpy(self.raw_train_data)
        #self.raw_test_data = tfds.as_numpy(self.raw_test_data)
        #self.ds = tfds.data_source(DATASET, data_dir = f"{DIR}/tensorflow_datasets/", file_format='array_record', download=True)
        #builder = tfds.builder(DATASET, file_format='array_record', download=True)
        #builder.download_and_prepare()
        #self.ds = builder.as_data_source()
        #self.raw_train_data = self.ds['train']
        #self.raw_test_data = self.ds['test']


        #self.train_data = preprocessDataset(self.raw_train_data,  weights.transforms())
        #self.test_data = preprocessDataset(self.raw_test_data,  weights.transforms())
        #self.train_data, self.val_data = train_test_split(self.train_data, test_size=0.2, random_state=0)
        self.dataloader_train = DataLoader(self.train_data, shuffle=True, batch_size=self.num_classes)
        #self.dataloader_val = DataLoader(self.val_data, shuffle=True, batch_size=64)
        self.dataloader_test = DataLoader(self.test_data, shuffle=True, batch_size=self.num_classes)

    
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

            
