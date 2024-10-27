import os
from contextlib import suppress
from datetime import datetime
from typing import Dict, List, Tuple
import torch
from torch.nn import CrossEntropyLoss
from torchmetrics import Precision, Recall

from .config import ModelSettings
from .constants import SEED
from .data import Transform
from .model import BowzerNet
from .utils import logger, save_json, save_pickle


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class BowzerClassifier:
    def __init__(self, model_settings: ModelSettings):
        print(f"Running on device: {DEVICE}")
        self.model_settings = model_settings
        self.epochs = self.model_settings.epochs
        self.lr = self.model_settings.learning_rate
        self.data_module = Transform(self.model_settings)
        self.dataloader_train, self.dataloader_val, self.dataloader_test = (
            self.data_module.process()
        )
        self.dog_names = self.data_module.get_dog_names()

    def view_sample_transformations(
        self,
        breeds: List[str] = ["Beagle", "German Shorthaired", "Chihuahua"],
        save_images: bool = False,
    ) -> None:
        _paths = self.data_module.get_breed_image(breeds)
        self.data_module.show_image_transforms(_paths, save=save_images)

    def train_epoch(
        self,
        idx: int,
        save_batch_model: bool = False,
    ) -> Tuple[float, float, List, List]:

        loss_list = []
        val_losses = []

        running_loss = 0.0
        for batch, (images, labels, image_paths) in enumerate(self.dataloader_train):
            self.model.train()
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            self.optimizer.zero_grad()
            preds = self.model(images)
            loss = self.loss_fn(preds, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            loss_list.append(loss.item())
            if batch % 10 == 0:
                self.logger.info(
                    f"loss: {loss.item():>7f}  [{(batch + 1) * len(images):>5d}/{len(self.dataloader_train.dataset):>5d}]"
                )
                if save_batch_model:
                    torch.save(
                        self.model.state_dict(),
                        f"{self.batch_path}/model_{idx}_{batch}",
                    )
        avg_loss = running_loss / (batch + 1)

        self.model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, labels, image_paths in self.dataloader_val:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                output = self.model(images)
                val_loss = self.loss_fn(output, labels)
                running_val_loss += val_loss.item()
                val_losses.append(val_loss.item())
        avg_val_loss = running_val_loss / len(self.dataloader_val)
        return avg_loss, avg_val_loss, loss_list, val_losses

    def train_eval(
        self,
        save_batch_models: bool = False,
        save_epoch_loss_lists: bool = False,
    ) -> Dict:
        self.run_time = datetime.now()
        self.batch_path = f"model_store/trained_{self.run_time.strftime('%Y%m%d')}/model_{self.run_time.strftime('%H%M%S')}"
        if save_batch_models:
            self.batch_path += "/batches"
        with suppress(FileExistsError):
            os.makedirs(self.batch_path)

        self.model_path = self.batch_path.replace("/batches", "")
        self.logger = logger(
            directory=self.model_path,
            filename=f"training_log.log",
        )
        self.logger.info(f"Model Results will be saved to: {self.model_path}")

        num_classes = self.data_module.num_classes
        self.model = BowzerNet(num_classes).to(DEVICE)
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.logger.info(f"Training {self.epochs} epochs, lr: {self.lr}...")
        performance = {}
        for n in range(self.epochs):
            self.logger.info(f"EPOCH {n + 1}:")
            avg_loss, avg_val_loss, loss_list, val_losses = self.train_epoch(
                n,
                save_batch_model=save_batch_models,
            )
            epoch_results = {"avg_loss": avg_loss, "avg_val_loss": avg_val_loss}
            if save_epoch_loss_lists:
                epoch_results["loss_list"] = loss_list
                epoch_results["val_loss"] = val_losses
            performance[f"epoch_{n}"] = epoch_results

        self.logger.info("Final Evaluation...")
        metric_precision = Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(DEVICE)
        metric_recall = Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(DEVICE)
        total = 0
        correct = 0
        val_losses = []
        self.model.eval()
        for images, labels, image_paths in self.dataloader_test:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            output = self.model(images)
            val_loss = self.loss_fn(output, labels)
            val_losses.append(val_loss.item())
            _, preds = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            metric_precision(preds, labels)
            metric_recall(preds, labels)
        performance["precision"] = metric_precision.compute().item()
        performance["recall"] = metric_recall.compute().item()
        performance["accuracy"] = correct / total

        self.logger.info(
            f"Accuracy  {(100 * correct / total):.2f}% [correct: {correct}, total: {total}]"
        )
        self.logger.info(f"Precision: {performance['precision']}")
        self.logger.info(f"Recall: {performance['recall']}")
        torch.save(
            self.model.state_dict(),
            f"{self.model_path}/model_epochs_{self.epochs}",
        )
        performance_path = save_json(
            performance,
            directory=f"{self.model_path}/",
            filename=f"model_epochs_{self.epochs}_performance.json",
        )
        save_pickle(self.model_settings, f"{self.model_path}/", "model_settings.pkl")
        self.logger.handlers.clear()
        return performance_path
