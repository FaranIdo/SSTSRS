import torch
from model import LinearNDVIModel
from dataset.landsat_seq import LandsatSeqDataset
from dataset.data_loader import LandsatDataLoader
from trainer.temporal_trainer import TemporalTrainer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import os
import json


def setup_logging():
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")


def define_hyperparameters():
    return {"dataset_path": "data/Landsat_NDVI_time_series_1984_to_2024.tif", "batch_size": 1024, "num_workers": 4, "num_epochs": 10, "lr": 1e-3}


def create_dataloader(dataset_path, window_size, batch_size, num_workers):
    dataset = LandsatSeqDataset(dataset_path=dataset_path, window_size=window_size)
    # use LandsatDataLoader
    loader = LandsatDataLoader(dataset, batch_size, train_rate=0.1, val_rate=0.05, num_workers=num_workers)
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
        for inputs, targets in val_loader:
            x, year_seq = inputs.split(1, dim=-1)
            year_seq = year_seq.squeeze(-1)
            x = x.to(device)
            year_seq = year_seq.to(device)
            targets = targets.to(device)

            # we need to convert create season encoding from year_seq by
            # if year is full number (1980, 1981, 1982) - season is 0
            # if year is full number + 0.5 (1980.5, 1981.5, 1982.5) - season is 1
            seasons = torch.where(year_seq % 1 == 0, 0, 1)

            # convert year_seq to int
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

    train_and_evaluate_models(models, h["dataset_path"], h["batch_size"], h["num_workers"], h["num_epochs"], h["lr"], device)


if __name__ == "__main__":
    # Create a list of models to evaluate
    models = [
        (LinearNDVIModel(num_features=1, sequence_length=5), "linear_5"),
        (LinearNDVIModel(num_features=1, sequence_length=10), "linear_10"),
        # Add more models here
    ]

    main(models)
