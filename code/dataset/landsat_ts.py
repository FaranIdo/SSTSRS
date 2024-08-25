import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import os
from tqdm import tqdm
import logging
import re

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

        logging.info(f"Landsat spectral dataset initialized with window size: {self.window_size}")

    def process_and_save_data(self, dataset_path: str) -> None:
        with rasterio.open(dataset_path) as src:
            ndvi_data: np.ndarray = src.read()
            band_names = src.descriptions
            logging.info(f"Original NDVI data shape: {ndvi_data.shape}")

        # Extracting year information from band names
        year_info = self.extract_year_info(band_names)
        logging.info(f"Year information extracted: {year_info}")

        ndvi_year_data = self.combine_ndvi_with_year(ndvi_data, year_info)
        logging.info(f"Reshaped NDVI data shape: {ndvi_data.shape}")

        self.X, self.y = self.create_input_output_pairs(ndvi_year_data, self.window_size)
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
    def combine_ndvi_with_year(ndvi_data: np.ndarray, year_info: dict) -> np.ndarray:
        num_pixels = ndvi_data.shape[1] * ndvi_data.shape[2]  # Total number of pixels
        combined_data = np.zeros((ndvi_data.shape[0], num_pixels, 2), dtype=np.float32)  # (seq_len, num_pixels, 2)
        for i, year in enumerate(year_info.values()):
            combined_data[i, :, 0] = ndvi_data[i].flatten()  # NDVI value
            combined_data[i, :, 1] = year  # Year value
        # Combine NDVI and year information into a single array
        return combined_data  # Now shape is (seq_len, num_pixels, 2)

    @staticmethod
    def extract_year_info(band_names: list[str]) -> dict:
        year_info = {}
        for i, band in enumerate(band_names):
            # Extract the year from the band name - the last 4 digits is multi year
            # Example: Band 0: Landsat_NDVI_spring_10-1983_04-1984 -> year = 1984
            # Example: Band 3: Landsat_NDVI_fall_05-1985_09-1985 -> year = 1985.5
            matches = re.findall(r"\d{4}", band)
            if matches:
                year = int(matches[-1])  # Take the last match
                # Check if the months are between May and September to add 0.5 to the year
                if re.search(r"05-\d{4}_09-\d{4}", band):
                    year += 0.5
                year_info[i] = year
        return year_info

    @staticmethod
    def create_input_output_pairs(ndvi_year_data: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
        logging.info("Creating input-output pairs...")
        num_pixels = ndvi_year_data.shape[1]
        seq_len = ndvi_year_data.shape[0]
        num_samples = num_pixels * (seq_len - window_size)

        X = np.zeros((num_samples, window_size, 2), dtype=ndvi_year_data.dtype)  # (num_samples, window_size, 2)
        y = np.zeros((num_samples, 2), dtype=ndvi_year_data.dtype)  # (num_samples, 2)

        sample_idx = 0
        for pixel in tqdm(range(num_pixels), desc="Processing pixels"):

            for i in range(seq_len - window_size):
                # Extracting a window of size 'window_size' from the NDVI data for the current pixel and time step 'i'.
                # This window will serve as the input (X) for the model.
                X[sample_idx] = ndvi_year_data[i : i + window_size, pixel]
                # The value of the NDVI data at the next time step (i + window_size) for the current pixel is used as the output (y).
                # This output is what the model is expected to predict based on the input window.
                y[sample_idx] = ndvi_year_data[i + window_size, pixel]
                sample_idx += 1

        logging.info("Input-output pairs created.")
        return X, y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
