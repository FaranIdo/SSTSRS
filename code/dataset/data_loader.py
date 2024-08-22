from torch.utils.data import DataLoader, random_split
from .landsat_ts import LandsatSpectralDataset
import logging


class LandsatDataLoader:
    def __init__(self, dataset_path: str, batch_size: int, split_rate: float, num_workers: int = 4):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.split_rate = split_rate
        self.num_workers = num_workers
        self.dataset = LandsatSpectralDataset(self.dataset_path)

    def create_data_loaders(self):
        logging.info("Creating data loaders...")

        # Calculate the split
        train_size = int(self.split_rate * len(self.dataset))
        val_size = len(self.dataset) - train_size
        logging.info(f"Dataset split: {train_size} training samples, {val_size} validation samples")

        # Split the dataset
        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])
        logging.info(f"Dataset split completed")

        # Create DataLoaders
        logging.info("Creating train loader...")
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        logging.info(f"Train loader created")

        logging.info("Creating validation loader...")
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        logging.info("Validation loader created")

        logging.info("Data loaders created successfully")
        return train_loader, val_loader
