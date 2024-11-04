import os
from contextlib import suppress
from datetime import datetime
from typing import Dict, List
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
    """
    Class to train BowzerNet using ModelSettings.
    """

    def __init__(self, model_settings: ModelSettings):
        """
        :param ModelSettings model_settings: see bowzer.config.ModelSettings
        """
        print(f"Running on device: {DEVICE}")
        self.model_settings = model_settings
        self.epochs = self.model_settings.epochs
        self.lr = self.model_settings.learning_rate
        self.data_module = Transform(self.model_settings)
        self.dataloader_train, self.dataloader_val, self.dataloader_test = (
            self.data_module.process()
        )

    def view_sample_transformations(
        self,
        breeds: List[str] = ["Beagle", "German Shorthaired", "Chihuahua"],
        save_images: bool = False,
    ) -> None:
        """
        function to view a few sample transformations within the training environment.

        :param breeds: list of breed names
        :type breeds: list
        :param save_images: when True, the image will be save to a tmp file when
        :type save_images: bool
        :rtype: None
        """
        _paths = self.data_module.get_breed_image(breeds)
        self.data_module.show_image_transforms(_paths, save=save_images)

    def train_epoch(
        self,
        idx: int,
        save_batch_model: bool = False,
        save_epoch_loss_lists: bool = False,
    ) -> Dict:
        """
        function to train a single epoch.

        :param int idx: epoch index
        :param bool save_batch_model: when True, save model state during epoch training.
        :param bool save_epoch_loss_lists: when True, save train and val loss within the epoch.

        :return Dict: Dictionary containing epoch loss and accuracy metrics
        """
        epoch_train_losses = []
        running_loss = 0.0
        train_correct = 0.0
        train_total = 0.0
        self.model.train()
        for batch, (images, labels, image_paths) in enumerate(self.dataloader_train):
            torch.cuda.empty_cache()  # clear space
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            epoch_train_losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            if batch % 10 == 0:
                # self.logger.info(f"loss: {loss.item():>7f}  [{(batch + 1) * len(images):>5d}/{len(self.dataloader_train.dataset):>5d}]")
                if save_batch_model:
                    torch.save(
                        self.model.state_dict(),
                        f"{self.batch_path}/model_{idx}_{batch}",
                    )
        train_loss = running_loss / (batch + 1)
        train_accuracy = train_correct / train_total

        epoch_val_losses = []
        running_val_loss = 0.0
        val_correct = 0.0
        val_total = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch, (images, labels, image_paths) in enumerate(self.dataloader_val):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                val_outputs = self.model(images)
                val_loss = self.loss_fn(val_outputs, labels)
                running_val_loss += val_loss.item()
                epoch_val_losses.append(val_loss.item())
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (val_predicted == labels).sum().item()
        val_loss = running_val_loss / (batch + 1)
        val_accuracy = val_correct / val_total
        return_dict = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
        }

        self.logger.info(
            " ".join([f"{key}: {value:.4f}" for key, value in return_dict.items()])
        )
        if save_epoch_loss_lists:
            return_dict["epoch_train_losses"] = epoch_train_losses
            return_dict["epoch_val_losses"] = epoch_val_losses
        return return_dict

    def train_eval(
        self,
        save_batch_models: bool = False,
        save_epoch_loss_lists: bool = False,
    ) -> Dict:
        """
        function to train and evaluate a model.

        :param bool save_batch_models: see :func:`train_epoch`
        :param bool save_epoch_loss_lists: see :func:`train_epoch`

        :return Dict: Dictionary containing model loss, accuracy metrics as well as epoch info.
        """
        self.run_time = datetime.now()
        self.batch_path = f"../model_store/trained_{self.run_time.strftime('%Y%m%d')}/model_{self.run_time.strftime('%H%M%S')}"
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
        self.logger.info(f"Number of classes: {num_classes}")
        self.model = BowzerNet(num_classes).to(DEVICE)
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.logger.info(f"Training {self.epochs} epochs, lr: {self.lr}...")
        performance = {}
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        for n in range(self.epochs):
            self.logger.info(f"EPOCH {n + 1}:")
            epoch_results = self.train_epoch(
                n,
                save_batch_model=save_batch_models,
                save_epoch_loss_lists=save_epoch_loss_lists,
            )
            train_losses.append(epoch_results["train_loss"])
            val_losses.append(epoch_results["val_loss"])
            train_accuracies.append(epoch_results["train_accuracy"])
            val_accuracies.append(epoch_results["val_accuracy"])
            if save_epoch_loss_lists:
                performance[f"epoch_{n}"] = {
                    "epoch_train_losses": epoch_results["epoch_train_losses"],
                    "epoch_val_losses": epoch_results["epoch_val_losses"],
                }
        performance["train_losses"] = train_losses
        performance["val_losses"] = val_losses
        performance["train_accuracies"] = train_accuracies
        performance["val_accuracies"] = val_accuracies

        self.logger.info("Final Evaluation...")
        metric_precision = Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(DEVICE)
        metric_recall = Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(DEVICE)
        total = 0
        correct = 0
        test_losses = []
        self.model.eval()
        for images, labels, image_paths in self.dataloader_test:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            output = self.model(images)
            test_loss = self.loss_fn(output, labels)
            test_losses.append(test_loss.item())
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
