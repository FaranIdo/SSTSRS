import torch
from torch.utils.data import DataLoader
from model import TemporalPositionalNDVITransformer
from dataset import LandsatDataLoader, LandsatFutureDataset
import logging
import argparse
import os
from train import TemporalTrainer
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

torch.manual_seed(0)


def load_model(checkpoint_path: str, model, num_features: int):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def sample_data(val_loader: DataLoader, num_samples: int = 10) -> list:
    samples = []
    for i, data in enumerate(val_loader):
        if i >= num_samples:
            break
        samples.append(data)
    return samples


def evaluate_temporal_model(model: TemporalPositionalNDVITransformer, samples: list) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = torch.nn.MSELoss(reduction="mean")
    mae_criterion = torch.nn.L1Loss(reduction="mean")
    for i, (inputs, targets) in enumerate(samples):
        ndvi, year_seq_int, seasons, y = TemporalTrainer.convert_inputs_targets_to_model(inputs, targets, device)

        with torch.no_grad():
            outputs = model(ndvi, year_seq_int, seasons)
        loss = criterion(y, outputs)
        mae = mae_criterion(y, outputs)

        print(f"Sample {i+1}:")
        # print(f"  X: {x.cpu().numpy()}")
        print(f"  True Value: {y.cpu().numpy()}")
        print(f"  Predicted Value: {outputs.cpu().numpy()}")
        print(f"  MSE Loss: {loss.item()}")
        print(f"  MAE: {mae.item()}")
        print()


def evaluate_transformer_models_grid(experiment_name):
    setup_logging()
    h = define_hyperparameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sequence_lengths = [5, 10, 20, 30, 40]
    max_distances = [0, 2, 4, 6, 8]

    results = []
    for seq_len in sequence_lengths:
        for max_dist in max_distances:
            model = TemporalPositionalNDVITransformer(embedding_dim=256, num_encoder_layers=3, sequence_length=seq_len + 1, start_year=1984, end_year=2024, attn_heads=8, dropout=0.2)
            experiment = f"{experiment_name}_seq{seq_len}_dist{max_dist}"

            logging.info(f"Evaluating model: {experiment}")

            model = model.to(device)
            experiment_folder = os.path.join("runs", experiment_name)

            # Load pre-trained weights
            checkpoint_path = os.path.join(experiment_folder, "checkpoint.tar")
            if os.path.exists(checkpoint_path):
                model = load_model(checkpoint_path, model, 1)
                logging.info(f"Loaded pre-trained weights from {checkpoint_path}")
            else:
                logging.warning(f"No pre-trained weights found for {experiment}. Skipping...")
                continue

            _, val_loader = create_dataloader(h["dataset_path"], seq_len, h["batch_size"], h["num_workers"], max_dist)

            l1_loss, mse_loss, r2 = evaluate_model(model, val_loader, device)
            logging.info(f"Model {experiment} - L1 Loss: {l1_loss:.4f}, MSE Loss: {mse_loss:.4f}, R2 Score: {r2:.4f}")

            results.append((seq_len, max_dist, l1_loss, mse_loss, r2))

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_experiment_name = f"{experiment_name}/grid_seq{min(sequence_lengths)}-{max(sequence_lengths)}_dist{min(max_distances)}-{max(max_distances)}_{current_time}"
    print_and_save_results_table(results, results_experiment_name)
    create_sequence_length_vs_loss_plot(results, results_experiment_name)


def print_and_save_results_table(results, experiment_name):
    df = pd.DataFrame(results, columns=["Sequence Length", "Max Distance", "L1 Loss", "MSE Loss", "R2 Score"])
    pivot_table = df.pivot(index="Sequence Length", columns="Max Distance", values="L1 Loss")

    print("\nResults Table (L1 Loss):")
    print(pivot_table.to_string(float_format="{:.4f}".format))

    runs_folder = "runs"
    os.makedirs(runs_folder, exist_ok=True)
    csv_path = os.path.join(runs_folder, f"{experiment_name}_results.csv")
    pivot_table.to_csv(csv_path, float_format="%.4f")
    print(f"\nResults saved to: {csv_path}")


def create_sequence_length_vs_loss_plot(results, experiment_name):
    df = pd.DataFrame(results, columns=["Sequence Length", "Max Distance", "L1 Loss", "MSE Loss", "R2 Score"])
    df["Years (N/2)"] = df["Sequence Length"] / 2
    pivot_table = df.pivot(index="Years (N/2)", columns="Max Distance", values="L1 Loss")

    plt.figure(figsize=(12, 8))
    for max_dist in pivot_table.columns:
        years_delta = int(int(max_dist) / 2 + 1)
        plt.plot(pivot_table.index, pivot_table[max_dist], marker="o", label=f"{years_delta} Years Delta")

    plt.xlabel("Sequence Length in Years (N/2)")
    plt.ylabel("L1 (MAE) Loss")
    plt.title("Past Sequence Length (Years) vs L1 Loss for Different Future Delta (Years)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    runs_folder = "runs"
    os.makedirs(runs_folder, exist_ok=True)
    plot_path = os.path.join(runs_folder, f"{experiment_name}_sequence_length_vs_loss.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nPlot saved to: {plot_path}")


def setup_logging():
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")


def define_hyperparameters():
    return {"dataset_path": "data/Landsat_NDVI_time_series_1984_to_2024.tif", "batch_size": 64, "num_workers": 4, "num_epochs": 10, "lr": 1e-3}


def create_dataloader(dataset_path, window_size, batch_size, num_workers, max_distance):
    dataset = LandsatFutureDataset(dataset_path=dataset_path, window_size=window_size, max_distance=max_distance)
    loader = LandsatDataLoader(dataset, batch_size, train_rate=0.1, val_rate=0.005, num_workers=num_workers)
    train_loader, val_loader = loader.create_data_loaders()
    return train_loader, val_loader


def evaluate_model(model, val_loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc=f"Evaluating {model.__class__.__name__}"):
            ndvi, year_seq_int, seasons, targets = TemporalTrainer.convert_inputs_targets_to_model(inputs, targets, device)
            outputs = model(ndvi, year_seq_int, seasons)

            y_true.extend(targets.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    l1_loss = mean_absolute_error(y_true, y_pred)
    mse_loss = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return l1_loss, mse_loss, r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True, help="Name of the experiment in runs/ folder.")
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

    logging.info("Loading validation dataset...")
    batch_size = 1
    train_rate = 0.1
    val_rate = 0.1
    num_workers = 8
    window_size = 40
    future_window_size = 10
    # loader = LandsatDataLoader("data/Landsat_NDVI_time_series_1984_to_2024.tif", batch_size, split_rate, num_workers, window_size)
    dataset = LandsatFutureDataset("data/Landsat_NDVI_time_series_1984_to_2024.tif", window_size, max_distance=future_window_size, exact_distance=True)
    loader = LandsatDataLoader(dataset, batch_size, train_rate, val_rate, num_workers)
    _, val_loader = loader.create_data_loaders()

    logging.info("Creating model...")
    # bert = BERT(num_features=1, hidden=256, n_layers=3, attn_heads=8, dropout=0.1)
    # model = BERTPrediction(bert, 1)
    model = TemporalPositionalNDVITransformer(embedding_dim=256, num_encoder_layers=3, sequence_length=window_size, start_year=1984, end_year=2024, attn_heads=8, dropout=0.0)

    logging.info("Loading model...")
    checkpoint_path = os.path.join("runs", args.experiment, "checkpoint.tar")
    model = load_model(checkpoint_path, model, 1)

    logging.info("Sampling data...")
    samples = sample_data(val_loader)

    logging.info("Evaluating model...")
    evaluate_temporal_model(model, samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True, help="Name of the experiment in runs/ folder.")
    args = parser.parse_args()
    evaluate_transformer_models_grid(args.experiment)
