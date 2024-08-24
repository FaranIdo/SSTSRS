from torch.utils.data import DataLoader, random_split, Subset
from .landsat_ts import LandsatSpectralDataset
import logging


class LandsatDataLoader:
    def __init__(self, dataset_path: str, batch_size: int, split_rate: float, num_workers: int = 4, window_size: int = 5):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.split_rate = split_rate
        self.num_workers = num_workers
        self.dataset = LandsatSpectralDataset(self.dataset_path, window_size)

    def create_data_loaders(self):
        logging.info(f"Creating data loaders with batch_size: {self.batch_size}, split_rate: {self.split_rate}, num_workers: {self.num_workers}")

        # Calculate the split

        train_size = int(self.split_rate * len(self.dataset))
        val_size = len(self.dataset) - train_size

        logging.info(f"Dataset split started with {len(self.dataset)} samples")
        # Split the dataset
        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])
        logging.info(f"Dataset split completed")

        # take only 1% of the training dataset and 1% of the validation dataset as actual training and validation dataset
        train_dataset = Subset(train_dataset, range(int(0.01 * len(train_dataset))))
        val_dataset = Subset(val_dataset, range(int(0.001 * len(val_dataset))))

        logging.info(f"Dataset split: {train_size} training samples, {val_size} validation samples")

        # Create DataLoaders
        logging.info("Creating train loader...")
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        logging.info(f"Train loader created")

        logging.info("Creating validation loader...")
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        logging.info("Validation loader created")

        logging.info("Data loaders created successfully")
        return train_loader, val_loader
