import torch
from torch.nn import CrossEntropyLoss
from typing import Dict, Literal
from datetime import datetime

from .model import Net
from .data_processing import Dataset

from .constants import (
    DIR,
    SEED
)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu')
print(f'Running on device: {DEVICE}')

class Model():
    def __init__(self, resize_n:int):
        self.resize_n = resize_n
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._data_module = Dataset(self.resize_n)
        self.dataloader_train, self.dataloader_test = self.data_module.process()
        self.num_classes = self.data_module.num_classes
        self.model = Net(self.num_classes,  self.resize_n)

        
    def train(self, epochs:int=3):
        #Define the model
        self.net = self.model(num_classes=self.num_classes, resize_n=self.resize_n)
        # Define the loss function
        criterion = CrossEntropyLoss()
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
    
    def train_epoch(self, epoch_index):
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
                running_loss = 0
        return last_loss

    def train_eval(self, epochs:int = 5):
        best_loss = 1_000_000
        counter = 0
        #Define the model
        self.net = Net(num_classes=self.num_classes, resize_n=self.resize_n)
        # Define the loss function
        self.criterion = CrossEntropyLoss()
        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 0.001)
        for epoch in range(epochs):
            print(f"EPOCH {counter + 1}:")
            self.net.train(True)
            avg_loss = self.train_epoch(counter)

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

            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                torch.save(self.net.state_dict(), f'{DIR}/bowzer/runs/trainer_{self.timestamp}/model_{self.timestamp}_{counter}')
            counter += 1



            
