from torchvision import transforms

config_0_base = {"batch_size": 64, "resize_n": 128, "learning_rate": 0.001}
config_0_transforms = {
    "train_transform": transforms.Compose(
        [
            transforms.Resize(
                (config_0_base["resize_n"], config_0_base["resize_n"]), antialias=True
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45),
            transforms.RandomGrayscale(0.50),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "test_transform": transforms.Compose(
        [
            transforms.Resize(
                (config_0_base["resize_n"], config_0_base["resize_n"]), antialias=True
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}

config_0_e5 = {"info": "First Trial. Trained with 5 Epochs", "epochs": 5}
config_0_e5.update(config_0_base)
config_0_e5.update(config_0_transforms)

config_0_e10 = {"info": "First Trial. Trained with 10 Epochs", "epochs": 10}
config_0_e10.update(config_0_base)
config_0_e10.update(config_0_transforms)

config_0_e25 = {"info": "First Trial. Trained with 25 Epochs", "epochs": 25}
config_0_e25.update(config_0_base)
config_0_e25.update(config_0_transforms)

config_0_e50 = {"info": "First Trial. Trained with 50 Epochs", "epochs": 50}
config_0_e50.update(config_0_base)
config_0_e50.update(config_0_transforms)


#### Retrain to Adjust for Overfitting ###
config_1_base = {"batch_size": 64, "resize_n": 128, "learning_rate": 0.001}
config_1_transforms = {
    "train_transform": transforms.Compose(
        [
            transforms.RandomResizedCrop(config_1_base["resize_n"], antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(kernel_size=3),
            transforms.RandomRotation(45),
            transforms.RandomGrayscale(0.50),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "test_transform": transforms.Compose(
        [
            transforms.RandomResizedCrop(config_1_base["resize_n"], antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}


config_1_e5 = {
    "info": "Exploring additional transformations. Change crop to random resize and added gaussian blur. Trained with 5 Epochs",
    "epochs": 5,
}
config_1_e5.update(config_1_base)
config_1_e5.update(config_1_transforms)

config_1_e10 = {
    "info": "Exploring additional transformations. Change crop to random resize and added gaussian blur. Trained with 10 Epochs",
    "epochs": 10,
}
config_1_e10.update(config_1_base)
config_1_e10.update(config_1_transforms)

config_1_e50 = {
    "info": "Exploring additional transformations. Change crop to random resize and added gaussian blur. Trained with 50 Epochs",
    "epochs": 50,
}
config_1_e50.update(config_1_base)
config_1_e50.update(config_1_transforms)


#### Retrain to Adjust for Overfitting ###
config_2_base = {"batch_size": 128, "resize_n": 128, "learning_rate": 0.01}
config_2_transforms = {
    "train_transform": transforms.Compose(
        [
            transforms.RandomResizedCrop(config_2_base["resize_n"], antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(45),
            transforms.RandomGrayscale(0.50),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "test_transform": transforms.Compose(
        [
            transforms.RandomResizedCrop(config_2_base["resize_n"], antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}
config_2_e10 = {
    "info": "Adding more tranformations. Changed crop to random resize, added gaussian blur and color jitter. increasing batch size. Trained with 10 Epochs",
    "epochs": 10,
}
config_2_e10.update(config_2_base)
config_2_e10.update(config_2_transforms)

#### Retrain to Adjust for Overfitting ###
config_2_base = {"batch_size": 64, "resize_n": 128, "learning_rate": 0.01}
config_2_transforms = {
    "train_transform": transforms.Compose(
        [
            transforms.RandomResizedCrop(config_2_base["resize_n"], antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(45),
            transforms.RandomGrayscale(0.50),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "test_transform": transforms.Compose(
        [
            transforms.RandomResizedCrop(config_2_base["resize_n"], antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}
config_2_e10 = {
    "info": "Adding more tranformations. Changed crop to random resize, added gaussian blur and color jitter. increasing batch size. Trained with 10 Epochs",
    "epochs": 10,
}
config_2_e10.update(config_2_base)
config_2_e10.update(config_2_transforms)

#### Retrain to Adjust for Overfitting ###
config_3_base = {"batch_size": 64, "resize_n": 128, "learning_rate": 0.001}
config_3_transforms = {
    "train_transform": transforms.Compose(
        [
            transforms.RandomResizedCrop(config_3_base["resize_n"], antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(45),
            transforms.RandomGrayscale(0.50),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "test_transform": transforms.Compose(
        [
            transforms.RandomResizedCrop(config_3_base["resize_n"], antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}
config_3_e50 = {
    "info": "Adding more tranformations. Changed crop to random resize, added gaussian blur and color jitter. Trained with 50 Epochs",
    "epochs": 50,
}
config_3_e50.update(config_3_base)
config_3_e50.update(config_3_transforms)


config_3_e75 = {
    "info": "Adding more tranformations. Changed crop to random resize, added gaussian blur and color jitter. Trained with 75 Epochs",
    "epochs": 75,
}
config_3_e75.update(config_3_base)
config_3_e75.update(config_3_transforms)