from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import logging
import os
import json
from tqdm import tqdm
from dataset.landsat_future_ts import LandsatFutureDataset
from dataset.data_loader import LandsatDataLoader
from model.svm_model import SVMNDVIModel


def setup_logging():
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")


def load_data(dataset_path, window_size, max_distance):
    dataset = LandsatFutureDataset(dataset_path=dataset_path, window_size=window_size, max_distance=max_distance)
    return dataset


def create_data_splits(dataset, batch_size, num_workers):
    loader = LandsatDataLoader(dataset, batch_size, train_rate=0.01, val_rate=0.0   1, num_workers=num_workers)
    train_loader, val_loader = loader.create_data_loaders()
    return train_loader, val_loader


def prepare_data_for_svm(data_loader):
    X, y = [], []
    for inputs, targets in tqdm(data_loader, desc="Preparing data"):
        ndvi = inputs[:, :, 0]
        X.append(ndvi.numpy().reshape(ndvi.shape[0], -1))  # Flatten NDVI sequence
        y.append(targets.numpy())
    return np.vstack(X), np.concatenate(y)


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    l1_loss = mean_absolute_error(y_test, y_pred)
    mse_loss = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return l1_loss, mse_loss, r2


def save_evaluation_results(experiment_folder, l1_loss, mse_loss, r2):
    results = {"L1 Loss": float(l1_loss), "MSE Loss": float(mse_loss), "R2 Score": float(r2)}
    os.makedirs(experiment_folder, exist_ok=True)
    results_path = os.path.join(experiment_folder, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f)
    logging.info(f"Saved evaluation results to {results_path}")


def main():
    setup_logging()
    logging.info("Starting SVM model evaluation")

    # Define hyperparameters
    dataset_path = "data/Landsat_NDVI_time_series_1984_to_2024.tif"
    window_size = 10  # Adjust this based on your sequence length
    max_distance = 0
    batch_size = 64
    num_workers = 4

    # Load data
    dataset = load_data(dataset_path, window_size, max_distance)

    # Create data splits
    train_loader, val_loader = create_data_splits(dataset, batch_size, num_workers)

    # Prepare data for SVM
    X_train, y_train = prepare_data_for_svm(train_loader)
    X_test, y_test = prepare_data_for_svm(val_loader)

    # Initialize model
    model = SVMNDVIModel()

    # Train model
    logging.info("Training SVM model")
    train_model(model, X_train, y_train)

    # Evaluate model
    logging.info("Evaluating SVM model")
    l1_loss, mse_loss, r2 = evaluate_model(model, X_test, y_test)

    # Log results
    logging.info(f"SVM Model - L1 Loss: {l1_loss:.4f}, MSE Loss: {mse_loss:.4f}, R2 Score: {r2:.4f}")

    # Save results
    experiment_folder = "runs/svm_model"
    save_evaluation_results(experiment_folder, l1_loss, mse_loss, r2)


if __name__ == "__main__":
    main()
