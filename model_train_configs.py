import torch
from torchvision.transforms import v2 as transforms

#### First Pass ###
config_0_base = {
    "batch_size": 64,
    "resize_n": 128,
    "learning_rate": 0.00001,
    "include_cats": False,
}
config_0_transforms = {
    "train_transform": transforms.Compose(
        [
            transforms.ToDtype(torch.uint8, scale=True),
            transforms.Resize(
                (config_0_base["resize_n"], config_0_base["resize_n"]), antialias=True
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "test_transform": transforms.Compose(
        [
            transforms.ToDtype(torch.uint8, scale=True),
            transforms.Resize(
                (config_0_base["resize_n"], config_0_base["resize_n"]), antialias=True
            ),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}

config_0_e50 = {"info": "First Trial. Trained with 50 Epochs", "epochs": 50}
config_0_e50.update(config_0_base)
config_0_e50.update(config_0_transforms)


#### Retrain to Adjust for Overfitting ###
prod_config_base = {
    "batch_size": 16,
    "resize_n": 128,
    "learning_rate": 0.00001,
    "include_cats": False,
}
prod_config_transforms = {
    "train_transform": transforms.Compose(
        [
            transforms.ToDtype(torch.uint8, scale=True),
            transforms.Resize(
                (config_0_base["resize_n"], config_0_base["resize_n"]), antialias=True
            ),
            transforms.RandomGrayscale(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "test_transform": transforms.Compose(
        [
            transforms.ToDtype(torch.uint8, scale=True),
            transforms.Resize(
                (config_0_base["resize_n"], config_0_base["resize_n"]), antialias=True
            ),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}


prod_config = {
    "info": "Final Model Config.",
    "epochs": 50,
}
prod_config.update(prod_config_base)
prod_config.update(prod_config_transforms)

# prod_config_base = {
#     "batch_size": 128,
#     "resize_n": 192,
#     "learning_rate": 0.00001,
#     "include_cats": False,
# }
# prod_config_transforms = {
#     "train_transform": transforms.Compose(
#         [
#             transforms.ToDtype(torch.uint8, scale=True),
#             transforms.RandomResizedCrop(prod_config_base["resize_n"], antialias=True),
#             transforms.RandomAutocontrast(),
#             transforms.RandomHorizontalFlip(),
#             # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
#             transforms.RandomRotation(45),
#             # Because ToTensor() is deprecated:
#             transforms.ToImage(),
#             transforms.ToDtype(torch.float32, scale=True),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ]
#     ),
#     "test_transform": transforms.Compose(
#         [
#             transforms.ToDtype(torch.uint8, scale=True),
#             transforms.RandomResizedCrop(prod_config_base["resize_n"], antialias=True),
#             # Because ToTensor() is deprecated:
#             transforms.ToImage(),
#             transforms.ToDtype(torch.float32, scale=True),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ]
#     ),
# }

# prod_config = {
#     "info": "Final Model Config.",
#     "epochs": 50,
# }
# prod_config.update(prod_config_base)
# prod_config.update(prod_config_transforms)
