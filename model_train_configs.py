import torch
from torchvision.transforms import v2 as transforms

#### First Pass ###
config_0_resize_n = 128
config_0_transforms = {
    "train_transform": transforms.Compose(
        [
            transforms.ToDtype(torch.uint8, scale=True),
            transforms.Resize((config_0_resize_n, config_0_resize_n), antialias=True),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(45),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "test_transform": transforms.Compose(
        [
            transforms.ToDtype(torch.uint8, scale=True),
            transforms.Resize((config_0_resize_n, config_0_resize_n), antialias=True),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}

config_0 = {
    "info": "First Trial.",
    "epochs": 50,
    "learning_rate": 0.00001,
    "include_cats": False,
    "batch_size": 16,
    "resize_n": config_0_resize_n,
}
config_0.update(config_0_transforms)

config_0_cats = {
    "info": "First Trial with Cats Included.",
    "epochs": 50,
    "learning_rate": 0.00001,
    "include_cats": True,
    "batch_size": 16,
    "resize_n": config_0_resize_n,
}
config_0_cats.update(config_0_transforms)


#### Retrain to Adjust for Overfitting ###
prod_config_resize_n = 128
prod_config_transforms = {
    "train_transform": transforms.Compose(
        [
            transforms.ToDtype(torch.uint8, scale=True),
            transforms.Resize(
                (prod_config_resize_n, prod_config_resize_n), antialias=True
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAutocontrast(),
            transforms.RandomGrayscale(),
            # transforms.ElasticTransform(), #these didn't improve performance
            # transforms.RandomRotation(45), #these didn't improve performance
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "test_transform": transforms.Compose(
        [
            transforms.ToDtype(torch.uint8, scale=True),
            transforms.Resize(
                (prod_config_resize_n, prod_config_resize_n), antialias=True
            ),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}


prod_config = {
    "info": "Final Model Config.",
    "epochs": 100,
    "learning_rate": 0.00001,
    "include_cats": False,
    "batch_size": 32,
    "resize_n": prod_config_resize_n,
}
prod_config.update(prod_config_transforms)

prod_config_with_cats = {
    "info": "Final Model Config.",
    "epochs": 50,
    "learning_rate": 0.00001,
    "include_cats": True,
    "batch_size": 32,
    "resize_n": prod_config_resize_n,
}
prod_config_with_cats.update(prod_config_transforms)
