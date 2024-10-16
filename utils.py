import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchmetrics import Precision, Recall
from typing import Literal

plt.rcParams["savefig.bbox"] = 'tight'

def show_image(image, transform):
    plt.imshow(transform(Image.open(image)).squeeze(0).permute(1,2,0))
    plt.show()

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
