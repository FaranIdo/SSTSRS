from torch.utils.data import DataLoader, random_split, Subset
from .landsat_ts import LandsatSpectralDataset
import logging


class LandsatDataLoader:

    def __init__(self, dataset_path: str, batch_size: int, train_rate: float = 0.1, val_rate: float = 0.05, num_workers: int = 4, window_size: int = 5):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.train_rate = train_rate
        self.val_rate = val_rate
        self.num_workers = num_workers
        self.dataset = LandsatSpectralDataset(self.dataset_path, window_size)

    def create_data_loaders(self):
        logging.info(f"Creating data loaders with batch_size: {self.batch_size}, train_rate: {self.train_rate}, val_rate: {self.val_rate}, num_workers: {self.num_workers}")

        # Calculate the split
        train_size = int(self.train_rate * len(self.dataset))
        val_size = int(self.val_rate * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size

        logging.info(f"Dataset split started with {len(self.dataset)} samples")
        # Split the dataset - don't use test dataset
        train_dataset, val_dataset, _ = random_split(self.dataset, [train_size, val_size, test_size])
        logging.info(f"Dataset split completed")

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
