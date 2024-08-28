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


class LandsatFutureDataset(Dataset):

    def __init__(self, dataset_path: str, window_size: int, max_distance: int = 0):
        """
        Initialize the LandsatFutureDataset.

        Args:
            dataset_path (str): Path to the dataset file.
            window_size (int): Size of the window to extract sequences. Note that last NDVI is used as target, so lookback window is actually window_size -1.
            # Example: If window_size is 11, it means we will look at the past 10 images, which corresponds to 5 years with 2 seasons each, and the 11th value will be the target.
            max_distance (int, optional): Maximum distance to select the target value. Defaults to 0 = next season.
        """
        self.window_size = window_size
        self.max_distance = max_distance
        self.dataset_path = dataset_path
        self.ndvi, self.years = self.load_data(dataset_path)
        self.seasons = torch.where(torch.from_numpy(self.years) % 1 == 0, 0, 1)
        self.years = self.years.astype(int)
        self.seq_len = self.ndvi.shape[0]

        assert self.ndvi.shape[0] == self.years.shape[0], "NDVI and years must have the same length"
        # assert seq_len is larger than window size + max_distance
        assert self.seq_len > self.window_size + self.max_distance, "Sequence length must be larger than window size + max_distance"
        logging.info(f"Landsat seq dataset initialized with window size: {self.window_size} and max_distance: {self.max_distance}")

    def load_data(self, dataset_path: str) -> tuple[np.ndarray, np.ndarray]:
        with rasterio.open(dataset_path) as src:
            ndvi_data: np.ndarray = src.read()
            band_names = src.descriptions
            logging.info(f"Original NDVI data shape: {ndvi_data.shape}")
            # shape: (seq_len, cols, rows)

        # Extracting year information from band names
        year_info = self.extract_year_info(band_names)
        logging.info(f"Year information extracted: {year_info}")
        return self.split_ndvi_and_year(ndvi_data, year_info)

    @staticmethod
    def split_ndvi_and_year(ndvi_data: np.ndarray, year_info: dict) -> tuple[np.ndarray, np.ndarray]:
        num_pixels = ndvi_data.shape[1] * ndvi_data.shape[2]  # Total number of pixels
        ndvi = np.zeros((ndvi_data.shape[0], num_pixels), dtype=np.float32)  # (seq_len, num_pixels)
        years = np.zeros((ndvi_data.shape[0], num_pixels), dtype=np.float32)  # (seq_len, num_pixels)
        # take all NDVIs that match the current year
        for i, year in enumerate(year_info.values()):
            ndvi[i, :] = ndvi_data[i].flatten()  # NDVI value
            years[i, :] = year  # Year value
        return ndvi, years

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
        return self.ndvi.shape[1]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Randomly select a starting point for the window -
        # Minus max_distance to make sure both X and y can fit in the sequence
        start_idx = random.randint(0, self.seq_len - self.window_size - self.max_distance - 1)

        # Extract the sequence of NDVI, year, and season data for the given index based on the window size
        ndvi_seq = self.ndvi[start_idx : start_idx + self.window_size, idx]
        years_seq = self.years[start_idx : start_idx + self.window_size, idx].tolist()
        seasons_seq = self.seasons[start_idx : start_idx + self.window_size, idx].tolist()

        # Randomly select a "far" value within the max_distance range
        far_idx = random.randint(start_idx + self.window_size, start_idx + self.window_size + self.max_distance)
        y = self.ndvi[far_idx, idx]

        # Combine the year + season of "far_idx" to (X years)
        years_seq = np.append(years_seq, self.years[far_idx, idx])
        seasons_seq = np.append(seasons_seq, self.seasons[far_idx, idx])

        # Pad the NDVI of X by one
        ndvi_seq = np.pad(ndvi_seq, (0, 1), "constant", constant_values=0)

        # Stack NDVI, years, and seasons to create the final X
        X = np.stack((ndvi_seq, years_seq, seasons_seq), axis=-1)

        # Convert to torch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return X_tensor, y_tensor


if __name__ == "__main__":
    # test the dataset
    dataset = LandsatFutureDataset(dataset_path="data/Landsat_NDVI_time_series_1984_to_2024.tif", window_size=5, max_distance=0)
    def print_sample(dataset, index):
        X, y = dataset[index]
        y_value = y.item()
        X_ndvi = [ndvi.item() for ndvi in X[:, 0]]
        X_years = [year.item() for year in X[:, 1]]
        X_seasons = [season.item() for season in X[:, 2]]
        print(f"Dataset sample {index}: NDVI data shape: {X.shape}")
        print(f"X NDVI: {X_ndvi}")
        print(f"X Years: {X_years}")
        print(f"X Seasons: {X_seasons}")
        print(f"y Value: {y_value}")

    print_sample(dataset, 20)
    print_sample(dataset, 100)
    print_sample(dataset, 15)
    print_sample(dataset, 18)
    print("Dataset length: ", len(dataset))
