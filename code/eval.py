import torch
from model import LinearNDVIModel, TemporalPositionalNDVITransformer, FullyConnectedNDVIModel, TSLinearNDVIModel, TSFullyConnectedNDVIModel
from dataset.landsat_seq import LandsatSeqDataset
from dataset.data_loader import LandsatDataLoader
from trainer.temporal_trainer import TemporalTrainer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import os
import json
from tqdm import tqdm
import numpy as np


def setup_logging():
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")


def define_hyperparameters():
    return {"dataset_path": "data/Landsat_NDVI_time_series_1984_to_2024.tif", "batch_size": 1024, "num_workers": 4, "num_epochs": 5, "lr": 1e-3}


def create_dataloader(dataset_path, window_size, batch_size, num_workers):
    dataset = LandsatSeqDataset(dataset_path=dataset_path, window_size=window_size)
    # use LandsatDataLoader
    loader = LandsatDataLoader(dataset, batch_size, train_rate=0.1, val_rate=0.5, num_workers=num_workers)
    train_loader, val_loader = loader.create_data_loaders()
    return train_loader, val_loader


def create_trainer(model, train_loader, valid_loader, lr, experiment):
    return TemporalTrainer(
        model,
        num_features=1,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=lr,
        experiment_folder="runs/" + experiment,
    )


def train_model(trainer, num_epochs):
    for epoch in range(num_epochs):
        train_loss, valid_loss = trainer.train(epoch)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")


def evaluate_model(model, val_loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc=f"Evaluating {model.__class__.__name__}"):
            x, year_seq = inputs.split(1, dim=-1)
            year_seq = year_seq.squeeze(-1)
            x = x.to(device)
            year_seq = year_seq.to(device)

            # Extract NDVI, year, and season from targets
            # y_ndvi = targets[:, 0].to(device)
            # y_year = targets[:, 1].to(device)
            # y_season = torch.where(y_year % 1 == 0, 0, 1)
            # y_year = y_year.int()
            seasons = torch.where(year_seq % 1 == 0, 0, 1)

            year_seq_int = year_seq.int()

            with torch.no_grad():
                outputs = model(x, year_seq_int, seasons)

            # takes only the first dimension of the targets cause we don't need the year, also remove it from the outputs
            y = targets[:, 0]
            outputs = outputs[:, 0]

            y_true.extend(y.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    l1_loss = mean_absolute_error(y_true, y_pred)
    mse_loss = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return l1_loss, mse_loss, r2


def save_evaluation_results(experiment_folder, l1_loss, mse_loss, r2):
    results = {"L1 Loss": float(l1_loss), "MSE Loss": float(mse_loss), "R2 Score": float(r2)}
    results_path = os.path.join(experiment_folder, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f)
    logging.info(f"Saved evaluation results to {results_path}")


import pandas as pd


def log_evaluation_summary(model_names, results):
    data = {
        "Experiment": [model_name[1] for model_name in model_names],
        "Model": [model_name[0] for model_name in model_names],
        "L1 Loss": [f"{metrics[0]:.4f}" for metrics in results],
        "MSE Loss": [f"{metrics[1]:.4f}" for metrics in results],
        "R2 Score": [f"{metrics[2]:.4f}" for metrics in results],
    }

    df = pd.DataFrame(data)
    logging.info(f"\nEvaluation Summary:\n{df.to_string(index=False)}")
    # logging.info(f"\nEvaluation Summary:\n{table}")


def train_and_evaluate_models(models, dataset_path, batch_size, num_workers, num_epochs, lr, device):
    for i, (model, experiment) in enumerate(models):
        logging.info(f"Training and evaluating model {i+1}/{len(models)}")

        model = model.to(device)

        # Create experiment folder for the model
        experiment_folder = os.path.join("runs", experiment)
        os.makedirs(experiment_folder, exist_ok=True)

        # Create data loader for the model
        train_loader, val_loader = create_dataloader(dataset_path, model.sequence_length, batch_size, num_workers)

        # Check if the model has already been trained
        checkpoint_path = os.path.join(experiment_folder, "checkpoint.tar")
        if os.path.exists(checkpoint_path):
            logging.info(f"Loading pre-trained weights for model '{experiment}'")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            trainer = create_trainer(model, train_loader, val_loader, lr, experiment)
            train_model(trainer, num_epochs)
            trainer.save()
            trainer.plot_losses()

        l1_loss, mse_loss, r2 = evaluate_model(model, val_loader, device)
        logging.info(f"Model {experiment} - L1 Loss: {l1_loss:.4f}, MSE Loss: {mse_loss:.4f}, R2 Score: {r2:.4f}")

        # Save evaluation results to file
        save_evaluation_results(experiment_folder, l1_loss, mse_loss, r2)


def main(models):
    setup_logging()
    h = define_hyperparameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    model_names = []
    for i, (model, experiment) in enumerate(models):
        logging.info(f"Training and evaluating model {i+1}/{len(models)}")

        model = model.to(device)

        # Create experiment folder for the model
        experiment_folder = os.path.join("runs", experiment)
        os.makedirs(experiment_folder, exist_ok=True)

        # Create data loader for the model
        train_loader, val_loader = create_dataloader(h["dataset_path"], model.sequence_length, h["batch_size"], h["num_workers"])

        # Check if the model has already been trained
        checkpoint_path = os.path.join(experiment_folder, "checkpoint.tar")
        if os.path.exists(checkpoint_path):
            logging.info(f"Loading pre-trained weights for model '{experiment}'")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            trainer = create_trainer(model, train_loader, val_loader, h["lr"], experiment)
            train_model(trainer, h["num_epochs"])
            trainer.save()
            trainer.plot_losses()

        l1_loss, mse_loss, r2 = evaluate_model(model, val_loader, device)
        logging.info(f"Model {experiment} - L1 Loss: {l1_loss:.4f}, MSE Loss: {mse_loss:.4f}, R2 Score: {r2:.4f}")

        # Save evaluation results to file
        save_evaluation_results(experiment_folder, l1_loss, mse_loss, r2)

        results.append((l1_loss, mse_loss, r2))
        model_names.append((model.__class__.__name__, experiment))

    log_evaluation_summary(model_names, results)


if __name__ == "__main__":
    # set const seed
    torch.manual_seed(1)
    np.random.seed(1)

    # Create a list of models to evaluate
    models = [
        # (TSLinearNDVIModel(sequence_length=5, ndvi_hidden_size=64, year_embed_size=16, season_embed_size=16, num_years=41, num_seasons=2, combined_hidden_size=64), "ts_linear_5"),
        # (TSFullyConnectedNDVIModel(sequence_length=5, num_years=41, num_seasons=2, year_embed_size=16, season_embed_size=16, hidden_sizes=[64, 64]), "ts_fc_5"),
        # (TemporalPositionalNDVITransformer(embedding_dim=256, num_encoder_layers=3, sequence_length=5, start_year=1984, end_year=2024, attn_heads=8, dropout=0.0), "24-08-26_00-06_temporal_200_d2"),
        (LinearNDVIModel(num_features=1, sequence_length=5), "linear_5"),
        # (LinearNDVIModel(num_features=1, sequence_length=10), "linear_10"),
        # (LinearNDVIModel(num_features=1, sequence_length=20), "linear_20"),
        # (FullyConnectedNDVIModel(num_features=1, sequence_length=5, hidden_size=64, num_layers=2), "fc_seq5_layers2"),
        # (FullyConnectedNDVIModel(num_features=1, sequence_length=10, hidden_size=64, num_layers=2), "fc_seq10_layers2"),
        # (FullyConnectedNDVIModel(num_features=1, sequence_length=20, hidden_size=64, num_layers=2), "fc_seq20_layers2"),
        # (FullyConnectedNDVIModel(num_features=1, sequence_length=5, hidden_size=64, num_layers=4), "fc_seq5_layers4"),
        # (FullyConnectedNDVIModel(num_features=1, sequence_length=10, hidden_size=64, num_layers=4), "fc_seq10_layers4"),
        # (FullyConnectedNDVIModel(num_features=1, sequence_length=20, hidden_size=64, num_layers=4), "fc_seq20_layers4"),
        # Add more models here
    ]

    main(models)
