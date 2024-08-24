import torch
from torch.utils.data import DataLoader
from model import BERT, BERTPrediction
from dataset import LandsatDataLoader
import logging
import argparse
import os

torch.manual_seed(0)


def load_model(checkpoint_path: str, model: BERTPrediction, num_features: int) -> BERTPrediction:
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


def evaluate_model(model: BERTPrediction, samples: list) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = torch.nn.MSELoss(reduction="mean")
    mae_criterion = torch.nn.L1Loss(reduction="mean")

    for i, (inputs, targets) in enumerate(samples):
        x, year_seq = inputs.split(1, dim=-1)
        year_seq = year_seq.squeeze(-1)
        x = x.to(device)
        year_seq = year_seq.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            outputs = model(x, year_seq)
            y = targets[:, 0]
            outputs = outputs[:, 0]
            loss = criterion(y, outputs)
            mae = mae_criterion(y, outputs)

        print(f"Sample {i+1}:")
        print(f"  True Value: {y.cpu().numpy()}")
        print(f"  Predicted Value: {outputs.cpu().numpy()}")
        print(f"  MSE Loss: {loss.item()}")
        print(f"  MAE: {mae.item()}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True, help="Name of the experiment in runs/ folder.")
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

    logging.info("Loading validation dataset...")
    batch_size = 128
    split_rate = 0.1
    num_workers = 8
    window_size = 5
    loader = LandsatDataLoader("data/Landsat_NDVI_time_series_1984_to_2024.tif", batch_size, split_rate, num_workers, window_size)
    _, val_loader = loader.create_data_loaders()

    logging.info("Creating model...")
    bert = BERT(num_features=1, hidden=256, n_layers=3, attn_heads=8, dropout=0.1)
    model = BERTPrediction(bert, 1)

    logging.info("Loading model...")
    checkpoint_path = os.path.join("runs", args.experiment, "checkpoint.tar")
    model = load_model(checkpoint_path, model, 1)

    logging.info("Sampling data...")
    samples = sample_data(val_loader)

    logging.info("Evaluating model...")
    evaluate_model(model, samples)


if __name__ == "__main__":
    main()
