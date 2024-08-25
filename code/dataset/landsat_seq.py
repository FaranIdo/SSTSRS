import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import logging
import re
import random

# Set a global seed for reproducibility
SEED = 42
random.seed(SEED)


class LandsatSeqDataset(Dataset):

    def __init__(self, dataset_path: str, window_size: int):
        self.window_size = window_size
        self.dataset_path = dataset_path
        self.ndvi_year_data = self.load_data(dataset_path)
        self.seq_len = self.ndvi_year_data.shape[0]

        # assert seq_len is larger the window size
        assert self.seq_len > self.window_size, "Sequence length must be larger than window size"
        logging.info(f"Landsat seq dataset initialized with window size: {self.window_size}")

    def load_data(self, dataset_path: str) -> np.ndarray:
        with rasterio.open(dataset_path) as src:
            ndvi_data: np.ndarray = src.read()
            band_names = src.descriptions
            logging.info(f"Original NDVI data shape: {ndvi_data.shape}")
            # shape: (seq_len, cols, rows)

        # Extracting year information from band names
        year_info = self.extract_year_info(band_names)
        logging.info(f"Year information extracted: {year_info}")

        return self.combine_ndvi_with_year(ndvi_data, year_info)

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

    def __len__(self) -> int:
        return self.ndvi_year_data.shape[1]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Randomly select a starting point for the window -
        # Minus 1 to make sure both X and y can fit in the sequence
        start_idx = random.randint(0, self.seq_len - self.window_size - 1)

        # Extract the sequence of NDVI and year data for the given index based on the window size
        X = self.ndvi_year_data[start_idx : start_idx + self.window_size, idx, :]
        y = self.ndvi_year_data[start_idx + self.window_size, idx, :]

        # Convert to torch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return X_tensor, y_tensor


if __name__ == "__main__":
    # test the dataset
    dataset = LandsatSeqDataset(dataset_path="data/Landsat_NDVI_time_series_1984_to_2024.tif", window_size=5)
    print("Dataset sample: ", dataset[20])
    print("Dataset sample2: ", dataset[100])
    print("Dataset length: ", len(dataset))
