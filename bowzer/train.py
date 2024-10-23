import torch
from torch.nn import CrossEntropyLoss
from torchmetrics import Precision, Recall
from typing import Dict, List, Tuple
from .model import BowzerNet
from .data import Transform
import os
from contextlib import suppress
from datetime import datetime
from .constants import SEED
from .utils import save_json, logger

torch.manual_seed(SEED)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


class BowzerClassifier:
    def __init__(self, resize_n: int):
        print(f"Running on device: {DEVICE}")
        self.resize_n = resize_n
        self.data_module = Transform(self.resize_n)
        self.dataloader_train, self.dataloader_test = self.data_module.process()
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
        model,
        dataloader_train,
        optimizer,
        loss_fn,
        save_batch_model: bool = False,
    ) -> Tuple[List, List]:
        loss_list = []
        running_loss = 0.0
        model.train()
        for batch, (images, labels) in enumerate(dataloader_train):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            preds = model(images)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loss_list.append(loss.item())
            if batch % 10 == 0:
                self.logger.info(
                    f"loss: {loss.item():>7f}  [{(batch + 1) * len(images):>5d}/{len(dataloader_train.dataset):>5d}]"
                )
                if save_batch_model:
                    torch.save(
                        model.state_dict(),
                        f"{self.batch_path}/model_{idx+1}_{batch}",
                    )
        avg_loss = running_loss / (batch + 1)
        return avg_loss, loss_list

    def train_eval(self, epochs: int, save_batch_models: bool = False) -> str:
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

        num_classes = len(self.dataloader_train.dataset.classes)
        model = BowzerNet(num_classes).to(DEVICE)
        loss_fn = CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        self.logger.info(f"Training {epochs} epochs...")
        performance = {}
        for n in range(epochs):
            self.logger.info(f"EPOCH {n + 1}:")
            avg_loss, loss_list = self.train_epoch(
                n,
                model,
                self.dataloader_train,
                optimizer,
                loss_fn,
                save_batch_model=save_batch_models,
            )
            performance[f"epoch_{n}"] = {"avg_loss": avg_loss, "loss_list": loss_list}

        self.logger.info("Evaluation...")
        metric_precision = Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(DEVICE)
        metric_recall = Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(DEVICE)
        model.eval()
        total, correct = [], []
        val_losses = []
        for images, labels in self.dataloader_test:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            output = model(images)
            val_loss = loss_fn(output, labels)
            val_losses.append(val_loss.item())
            _, preds = torch.max(output.data, 1)
            total += [labels.size(0)]
            correct += [(preds == labels).sum().item()]
            metric_precision(preds, labels)
            metric_recall(preds, labels)
        performance["precision"] = metric_precision.compute().item()
        performance["recall"] = metric_recall.compute().item()
        performance["total"] = total
        performance["correct"] = correct
        performance["val_losses"] = val_losses

        self.logger.info(
            f"Accuracy  {100 * sum(performance['correct']) / sum(performance['total']):.2f}% [correct: {sum(performance['correct'])}, total: {sum(performance['total'])}]"
        )
        self.logger.info(f"Precision: {performance['precision']}")
        self.logger.info(f"Recall: {performance['recall']}")
        torch.save(
            model.state_dict(),
            f"{self.model_path}/model_epochs_{epochs}",
        )
        performance_path = save_json(
            performance,
            directory=f"{self.model_path}/",
            filename=f"model_epochs_{epochs}_performance.json",
        )
        self.logger.handlers.clear()
        return performance_path
