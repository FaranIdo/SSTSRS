import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import os
from tqdm import tqdm
import logging


class LandsatSpectralDataset(Dataset):

    def __init__(self, dataset_path: str, window_size: int):
        self.processed_data_path: str = os.path.join(os.path.dirname(dataset_path), f"processed_data_window_size_{window_size}_timesteps.npz")
        self.window_size = window_size

        if os.path.exists(self.processed_data_path):
            logging.info("Loading pre-processed data...")
            self.load_processed_data()
            logging.info("Pre-processed data loaded successfully.")
        else:
            logging.info("Processing new data...")
            self.process_and_save_data(dataset_path)
            logging.info("New data processed and saved successfully.")

    def process_and_save_data(self, dataset_path: str) -> None:
        with rasterio.open(dataset_path) as src:
            ndvi_data: np.ndarray = src.read()
            logging.info(f"Original NDVI data shape: {ndvi_data.shape}")

        ndvi_data = self.reshape_data(ndvi_data)
        logging.info(f"Reshaped NDVI data shape: {ndvi_data.shape}")

        self.X, self.y = self.create_input_output_pairs(ndvi_data, self.window_size)
        logging.info(f"Input shape: {self.X.shape}, Output shape: {self.y.shape}, Window size: {self.window_size}")

        logging.info("Saving processed data...")
        np.savez_compressed(self.processed_data_path, X=self.X, y=self.y)
        logging.info(f"Processed data saved to {self.processed_data_path}")

        # Convert to PyTorch tensors
        self.X = torch.FloatTensor(self.X)
        self.y = torch.FloatTensor(self.y)

    def load_processed_data(self) -> None:
        data = np.load(self.processed_data_path)
        self.X = torch.FloatTensor(data["X"])
        self.y = torch.FloatTensor(data["y"])
        logging.info(f"Loaded data shapes - Input: {self.X.shape}, Output: {self.y.shape}, Window size: {self.window_size}")

    @staticmethod
    def reshape_data(ndvi_data: np.ndarray) -> np.ndarray:
        return ndvi_data.reshape(-1, ndvi_data.shape[0]).T  # Now shape is (seq_len, num_pixels)

    @staticmethod
    def create_input_output_pairs(ndvi_data: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
        logging.info("Creating input-output pairs...")
        num_pixels = ndvi_data.shape[1]
        seq_len = ndvi_data.shape[0]
        num_samples = num_pixels * (seq_len - window_size)

        X = np.zeros((num_samples, window_size), dtype=ndvi_data.dtype)
        y = np.zeros(num_samples, dtype=ndvi_data.dtype)

        sample_idx = 0
        for pixel in tqdm(range(num_pixels), desc="Processing pixels"):
            for i in range(seq_len - window_size):
                X[sample_idx] = ndvi_data[i : i + window_size, pixel]
                y[sample_idx] = ndvi_data[i + window_size, pixel]
                sample_idx += 1

        logging.info("Input-output pairs created.")
        return X, y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
