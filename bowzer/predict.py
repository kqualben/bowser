from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from .constants import SEED
from .data import Transform
from .model import BowzerNet
from .utils import open_image, open_pickle, get_model_path, get_model_settings

plt.rcParams["savefig.bbox"] = "tight"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Predictor:
    def __init__(self, model_name: str):
        print(f"Running on device: {DEVICE}")
        self.model_name = model_name
        self.model_path = get_model_path(model_name)
        self.model_settings = open_pickle(get_model_settings(self.model_path))
        self.data_module = Transform(self.model_settings)
        self.dataloader_train, self.dataloader_val, self.dataloader_test = (
            self.data_module.process()
        )
        self.num_classes = self.data_module.num_classes
        self.model = BowzerNet(self.num_classes).to(DEVICE)
        self.model.load_state_dict(torch.load(self.model_path, weights_only=False))
        self.train_embeddings, self.train_batch_labels, self.train_batch_image_paths = (
            self.get_embeddings_labels(self.model, self.dataloader_train)
        )
        self.saved_images = []
        self.target_label = None

    @staticmethod
    def get_embeddings_labels(model, dataloader) -> Tuple[np.ndarray, List, List]:
        model.eval()
        embeddings = []
        batch_labels = []
        batch_paths = []
        with torch.inference_mode():
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
        with torch.inference_mode():
            pred = self.model(image_tensor)
        return pred

    @staticmethod
    def prediction_embedding(preds) -> np.ndarray:
        new_image_embedding = preds.cpu().numpy()
        return new_image_embedding

    @staticmethod
    def prediction_probabilities(preds) -> np.ndarray:
        pred_proba = preds.squeeze(0).softmax(0)
        return pred_proba

    @staticmethod
    def similarity_scores(
        training_images_embeddings: np.ndarray, target_image_embedding: np.ndarray
    ) -> np.ndarray:
        cos = torch.nn.CosineSimilarity(dim=1)
        sim_scores = cos(
            torch.Tensor(training_images_embeddings),
            torch.Tensor(target_image_embedding),
        ).numpy()
        return sim_scores

    def predict(self, target_image_path: str) -> None:
        self.target_image_path = target_image_path
        self._target_predictions = self.image_prediction(self.target_image_path)
        self._target_embedding = self.prediction_embedding(self._target_predictions)
        self._target_prediction_scores = self.similarity_scores(
            self.train_embeddings, self._target_embedding
        )
        self._ranked_breeds = None
        self._breed_proba = None

    @property
    def target_predictions(self):
        return self._target_predictions

    @property
    def target_prediction_scores(self):
        return self._target_prediction_scores

    @property
    def target_image_embedding(self):
        return self._target_embedding

    def _target_breed_ranking(self) -> List[Tuple[str, str, float]]:
        scores_ranked = np.argsort(self.target_prediction_scores)[::-1]
        ranked_breeds = [
            (
                self.data_module.get_idx_label(self.train_batch_labels[i].item()),
                self.train_batch_image_paths[i],
                (self.target_prediction_scores[i].item() + 1) / 2,  # similarity score
            )
            for i in scores_ranked
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

    def show_predicted_images(
        self, top_n_breeds: int, scaler: int = 3, save: bool = False
    ):
        top_n_breeds_info = self.get_top_breed_prediction(top_n_breeds)
        fig, axes = plt.subplots(
            ncols=(1 + top_n_breeds),
            figsize=((1 + top_n_breeds) * scaler, scaler * 1.25),
        )
        target_image = open_image(self.target_image_path)
        target_label = f"{'Target' if self.target_label is None else self.target_label}"
        axes[0].imshow(np.asarray(target_image))
        axes[0].set_title(target_label, size="medium")
        axes[0].axis("off")
        for i, (breed_name, path, similarity) in enumerate(top_n_breeds_info):
            if i == 0:
                top_breed_name = breed_name
            pred_image = open_image(path)
            axes[i + 1].imshow(np.asarray(pred_image))
            axes[i + 1].axis("off")
            axes[i + 1].set_title(
                f"{i+1}: {breed_name}\n" + rf"Similarity Score: {similarity:.4f}",
                size="medium",
            )
        fig.suptitle(f"Top Match for {target_label}: {top_breed_name}")
        fig.tight_layout()
        if save:
            path = self.target_image_path.replace("targets", "predictions").replace(
                ".jpg",
                f"_top_{top_n_breeds}_breeds_{'_'.join(self.model_name.split('_')[::-1][:2])}.jpg",
            )
            plt.savefig(path)
            self.saved_images.append(path)
        plt.show()

    def target_image_dict_loop(
        self,
        image_dict: Dict,
        top_n_breeds: int = 5,
        save: bool = False,
    ) -> None:
        for character in image_dict:
            print(f"Predicting {character} images...")
            char_images = image_dict[character]
            for image in char_images:
                self.target_label = character.title()
                self.predict(char_images[image])
                self.show_predicted_images(top_n_breeds=top_n_breeds, save=save)
