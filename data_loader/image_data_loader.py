import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

# Set up a logger for this module
logger = logging.getLogger(__name__)


def _get_default_train_transform() -> transforms.Compose:
    """Returns a default set of augmentations for training."""
    return transforms.Compose(
        [
            # Handles all necessary positional and scale jitter.
            transforms.RandomResizedCrop(size=(240, 240), scale=(0.8, 1.0)),
            # Orientation augmentations.
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),  # Keep if up/down doesn't matter
            transforms.RandomRotation(degrees=90),  # Keep if 90-degree turns are valid
            # Color augmentations.
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomGrayscale(p=0.2),
            # Final steps.
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _get_default_val_transform() -> transforms.Compose:
    """Returns a default transform for validation/testing."""
    return transforms.Compose(
        [
            transforms.Resize((240, 240)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_dataloaders(config: dict) -> dict[str, DataLoader]:
    """
    The main factory function for creating train and validation dataloaders.
    This is the function you will call from your main training script.

    Args:
        config (dict): A dictionary containing dataloader configuration,
                       typically from a YAML file. Expected keys:
                       `data_dir`, `batch_size`, `num_workers`, `validation_split`.

    Returns:
        A dictionary containing 'train' and 'val' dataloaders.
        e.g., {'train': DataLoader, 'val': DataLoader}
    """
    # 1. Configuration Validation
    data_dir = Path(config["data_dir"])
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    batch_size = config["batch_size"]
    num_workers = config.get("num_workers", 4)
    validation_split = config.get("validation_split", 0.2)
    random_seed = config.get("random_seed", 42)
    shuffle = config.get("shuffle", True)

    # 2. Create Dataset instances with appropriate transforms
    # We create one dataset instance for training (with augmentation) and one for validation.
    train_dataset = datasets.ImageFolder(
        root=data_dir, transform=_get_default_train_transform()
    )
    # The validation dataset should NOT have augmentations.
    # Note: We create a new instance pointing to the same data.
    val_dataset = datasets.ImageFolder(
        root=data_dir, transform=_get_default_val_transform()
    )

    # 3. Perform a stratified split to get indices for train and validation
    targets = train_dataset.targets
    indices = list(range(len(targets)))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=validation_split,
        random_state=random_seed,
        stratify=targets,
        shuffle=shuffle,
    )

    logger.info(
        f"Dataset split: {len(train_indices)} training samples, {len(val_indices)} validation samples."
    )

    # 4. Create PyTorch Subset objects
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    use_sampler = config.get("use_weighted_sampler", False)
    sampler = None
    if use_sampler:
        # Get targets for the train subset
        train_targets = np.array(train_dataset.targets)[train_indices]
        class_sample_count = np.bincount(train_targets)
        class_weights = 1.0 / class_sample_count
        sample_weights = class_weights[train_targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

        if shuffle:
            shuffle = False
            logger.warning("shuffle is set to False due to WeightedRandomSampler.")

    # 5. Create the DataLoaders
    # The training loader should be shuffled.
    train_loader = DataLoader(
        dataset=train_subset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    # The validation loader should NOT be shuffled.
    val_loader = DataLoader(
        dataset=val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {"train": train_loader, "val": val_loader}


def get_test_dataloader(config: dict) -> DataLoader:
    """
    Factory function to create a test dataloader.
    This is useful for evaluating the model on a separate test set.

    Args:
        config (dict): A dictionary containing dataloader configuration,
                       typically from a YAML file. Expected keys:
                       `data_dir`, `batch_size`, `num_workers`.

    Returns:
        A DataLoader for the test dataset.
    """
    data_dir = Path(config["data_dir"])
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    batch_size = config["batch_size"]
    num_workers = config.get("num_workers", 4)

    # Create the test dataset with no augmentations
    test_dataset = datasets.ImageFolder(
        root=data_dir, transform=_get_default_val_transform()
    )

    # Create the DataLoader for the test dataset
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader
