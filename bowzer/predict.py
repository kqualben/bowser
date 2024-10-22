import torch
from .data import Transform
from .model import BowzerNet
from .utils import open_image
from typing import List, Dict
from .constants import RESIZE_N, SEED
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["savefig.bbox"] = "tight"
torch.manual_seed(SEED)


class DataProcessing:
    def __init__(self):
        self.data_module = Transform(RESIZE_N)
        self.dataloader_train, self.dataloader_test = self.data_module.process()
        self.num_classes = len(self.dataloader_train.dataset.classes)


class Bowzer(DataProcessing):
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.model = BowzerNet(self.num_classes)
        self.model.load_state_dict(torch.load(self.path))

    def predict_target_class(self, image_path: str) -> torch.Tensor:
        target_image = open_image(image_path)
        image_tensor = self.data_module.train_transforms(target_image).unsqueeze(0)
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            pred = self.model(image_tensor).squeeze(0)
        pred_cls = pred.softmax(0)  # tensors to probabilities
        return pred_cls

    def generate_predictions(
        self, image_paths: List[str], show_top_prediction: bool = True
    ) -> Dict:
        results = {}
        for target in image_paths:
            preds = self.predict_target_class(target)
            cls_id = preds.argmax().item()
            print(f"{target} -> {self.data_module.get_idx_label(cls_id)}")
            results[target] = {
                "idx": cls_id,
                "label": self.data_module.get_idx_label(cls_id),
                "all_matches": {
                    k: preds.data[v].item()
                    for k, v in self.data_module.class_dict.items()
                },
            }
        if show_top_prediction:
            self.show_predicted_images(results, top_n=1, save=True)
        return results

    def show_predicted_images(
        self, results: Dict, top_n: int = 1, save: bool = False
    ) -> None:
        for target in results:
            fig, axes = plt.subplots(ncols=(1 + top_n), squeeze=True)
            target_image = open_image(target)
            axes[0].imshow(np.asarray(target_image))
            axes[0].set_title(f"Target", size="medium")

            counter = 1
            preds = list(
                dict(
                    sorted(
                        results[target]["all_matches"].items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                ).keys()
            )[:top_n]
            for predicted_breed in preds:
                pred_image_path = self.data_module.get_breed_image([predicted_breed])[
                    predicted_breed
                ]
                pred_image = open_image(pred_image_path)
                axes[counter].imshow(np.asarray(pred_image))
                axes[counter].set_title(
                    f"Predicted:\n{predicted_breed} {100*(results[target]['all_matches'][predicted_breed]):.2f}%",
                    size="medium",
                )
                counter += 1
            fig.tight_layout()
            if save:
                path = f"/tmp/{target.split('/')[-1].split('.')[0]}_top_{top_n}_matches.png"
                plt.savefig(path)
                self.data_module.saved_images.append(path)
            plt.show()
