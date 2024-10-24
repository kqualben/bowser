import torch
from .data import Transform
from .model import BowzerNet
from .utils import open_image
from typing import List, Dict, Tuple
from .constants import RESIZE_N, SEED
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

plt.rcParams["savefig.bbox"] = "tight"
torch.manual_seed(SEED)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


class DataProcessing:
    def __init__(self):
        print(f"Running on device: {DEVICE}")
        self.data_module = Transform(RESIZE_N)
        self.dataloader_train, self.dataloader_test = self.data_module.process()
        self.num_classes = len(self.dataloader_train.dataset.classes)


class Predictor(DataProcessing):
    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self.model = BowzerNet(self.num_classes).to(DEVICE)
        self.model.load_state_dict(torch.load(self.model_path, weights_only=False))
        self.train_embeddings, self.train_batch_labels, self.train_batch_image_paths = (
            self.get_embeddings_labels(self.model, self.dataloader_train)
        )
        self.saved_images = []

    @staticmethod
    def get_embeddings_labels(model, dataloader) -> Tuple[np.ndarray, List]:
        model.eval()
        embeddings = []
        batch_labels = []
        batch_paths = []
        with torch.no_grad():
            for images, labels, image_paths in dataloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                embeddings.append(outputs.cpu().numpy())
                batch_labels.extend(labels)
                batch_paths.extend(image_paths)
        return np.vstack(embeddings), batch_labels, batch_paths

    def image_to_tensor(self, image_path: str) -> torch.Tensor:
        target_image = open_image(image_path)
        image_tensor = (
            self.data_module.train_transforms(target_image).unsqueeze(0).to(DEVICE)
        )
        return image_tensor

    def image_prediction(self, image_path: str):
        print(f"Making predictions for: {image_path}")
        image_tensor = self.image_to_tensor(image_path)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(image_tensor)
        return pred

    def predict(self, target_image_path: str) -> None:
        self.target_image_path = target_image_path
        self._target_predictions = self.image_prediction(self.target_image_path)
        self._ranked_breeds = None
        self._breed_proba = None

    @property
    def target_predictions(self):
        return self._target_predictions

    @staticmethod
    def prediction_embedding(preds) -> np.ndarray:
        new_image_embedding = preds.cpu().numpy()
        return new_image_embedding

    @staticmethod
    def prediction_probabilities(preds) -> np.ndarray:
        pred_proba = preds.squeeze(0).softmax(0)
        return pred_proba

    def _target_breed_ranking(self) -> List[Tuple[str, str]]:
        target_embedding = self.prediction_embedding(self.target_predictions)
        scores = cosine_similarity(self.train_embeddings, target_embedding).flatten()
        ranked_cls_id = np.argsort(scores)[::-1]
        ranked_breeds = [
            (
                self.data_module.get_idx_label(self.train_batch_labels[i].item()),
                self.train_batch_image_paths[i],
            )
            for i in ranked_cls_id
        ]
        return ranked_breeds

    @property
    def breed_ranking(self) -> List[Tuple[str, str]]:
        if self._ranked_breeds is None:
            self._ranked_breeds = self._target_breed_ranking()
        return self._ranked_breeds

    def _target_breed_probabilities(self) -> Dict[str, float]:
        preds_proba = self.prediction_probabilities(self.target_predictions)
        breed_proba = {
            k: preds_proba.data[v].item()
            for k, v in self.data_module.class_dict.items()
        }
        return breed_proba

    @property
    def breed_probabilities(self) -> Dict[str, float]:
        if self._breed_proba is None:
            self._breed_proba = self._target_breed_probabilities()
        return self._breed_proba

    def get_top_breed_prediction(self, n: int = 1) -> List:
        return self.breed_ranking[:n]

    def predict_target_class(self, image_path: str) -> torch.Tensor:
        target_image = open_image(image_path)
        image_tensor = self.data_module.train_transforms(target_image).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(image_tensor).squeeze(0)
        pred_cls = pred.softmax(0)  # convert tensors to probabilities
        return pred_cls

    def show_predicted_images(
        self, top_n_breeds: int, scaler: int = 3, save: bool = False
    ):
        top_n_breeds_info = self.get_top_breed_prediction(top_n_breeds)
        fig, axes = plt.subplots(
            ncols=(1 + top_n_breeds),
            figsize=((1 + top_n_breeds) * scaler, scaler * 1.25),
        )
        target_image = open_image(self.target_image_path)
        axes[0].imshow(np.asarray(target_image))
        axes[0].set_title(f"Target", size="medium")
        axes[0].axis("off")
        for i, (breed_name, path) in enumerate(top_n_breeds_info):
            pred_image = open_image(path)
            axes[i + 1].imshow(np.asarray(pred_image))
            axes[i + 1].axis("off")
            axes[i + 1].set_title(
                f"{i+1}: {breed_name}",
                size="medium",
            )
        fig.tight_layout()
        if save:
            path = f"/tmp/{self.target_image_path.split('/')[-1].split('.')[0]}_top_{top_n_breeds}_matches.png"
            plt.savefig(path)
            self.saved_images.append(path)
        plt.show()
