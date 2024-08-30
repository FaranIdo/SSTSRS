import torch
from model import LinearNDVIModel, TemporalPositionalNDVITransformer, FullyConnectedNDVIModel, TSLinearEmbeddingNDVIModel, TSFCEmbeddingNDVIModel
from dataset.landsat_seq import LandsatSeqDataset
from dataset.landsat_future_ts import LandsatFutureDataset
from dataset.data_loader import LandsatDataLoader
from trainer.temporal_trainer import TemporalTrainer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def setup_logging():
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")


def define_hyperparameters():
    return {"dataset_path": "data/Landsat_NDVI_time_series_1984_to_2024.tif", "batch_size": 64, "num_workers": 4, "num_epochs": 10, "lr": 1e-3}


def create_dataloader(dataset_path, window_size, batch_size, num_workers, max_distance):
    dataset = LandsatFutureDataset(dataset_path=dataset_path, window_size=window_size, max_distance=max_distance)
    # use LandsatDataLoader
    loader = LandsatDataLoader(dataset, batch_size, train_rate=0.1, val_rate=0.005, num_workers=num_workers)
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
            ndvi, year_seq_int, seasons, targets = TemporalTrainer.convert_inputs_targets_to_model(inputs, targets, device)

            with torch.no_grad():
                outputs = model(ndvi, year_seq_int, seasons)

            y_true.extend(targets.cpu().numpy())
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


def train_and_evaluate_models(models, dataset_path, batch_size, num_workers, num_epochs, lr, device, use_pretrained):
    for i, (model, experiment, max_distance) in enumerate(models):
        logging.info(f"Training and evaluating model {i+1}/{len(models)}")

        model = model.to(device)

        # Create experiment folder for the model
        experiment_folder = os.path.join("runs", experiment)
        os.makedirs(experiment_folder, exist_ok=True)

        # Create data loader for the model
        train_loader, val_loader = create_dataloader(dataset_path, model.sequence_length - 1, batch_size, num_workers, max_distance)

        # Check if the model has already been trained and use_pretrained is True
        checkpoint_path = os.path.join(experiment_folder, "checkpoint.tar")
        if use_pretrained and os.path.exists(checkpoint_path):
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


def main(models, use_pretrained):
    setup_logging()
    h = define_hyperparameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_and_evaluate_models(models, h["dataset_path"], h["batch_size"], h["num_workers"], h["num_epochs"], h["lr"], device, use_pretrained)


def evaluate_fc_models_grid(use_pretrained):
    setup_logging()
    h = define_hyperparameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #   sequence_lengths = [5, 40]
    #    max_distances = [0, 10]
    sequence_lengths = [5, 10, 20, 30, 40]
    max_distances = [0, 2, 4, 6, 8]

    results = []
    for seq_len in sequence_lengths:
        for max_dist in max_distances:
            model = FullyConnectedNDVIModel(sequence_length=seq_len + 1, hidden_size=64, num_layers=3)
            experiment = f"fc_seq{seq_len}_dist{max_dist}"

            logging.info(f"Training and evaluating model: {experiment}")

            model = model.to(device)
            experiment_folder = os.path.join("runs", experiment)
            os.makedirs(experiment_folder, exist_ok=True)

            train_loader, val_loader = create_dataloader(h["dataset_path"], seq_len, h["batch_size"], h["num_workers"], max_dist)

            if use_pretrained and os.path.exists(experiment_folder):
                logging.info(f"Loading pre-trained weights for model '{experiment}'")
                checkpoint = torch.load(experiment_folder + "/checkpoint.tar")
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                trainer = create_trainer(model, train_loader, val_loader, h["lr"], experiment)
                train_model(trainer, h["num_epochs"])
                trainer.save()
                trainer.plot_losses()

            l1_loss, _, _ = evaluate_model(model, val_loader, device)
            logging.info(f"Model {experiment} - L1 Loss: {l1_loss:.4f}")

            results.append((seq_len, max_dist, l1_loss))

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"fc_models_grid_seq{min(sequence_lengths)}-{max(sequence_lengths)}_dist{min(max_distances)}-{max(max_distances)}_{current_time}"
    print_and_save_results_table(results, experiment_name)
    create_sequence_length_vs_loss_plot(results, experiment_name)


def print_and_save_results_table(results, experiment_name):
    df = pd.DataFrame(results, columns=["Sequence Length", "Max Distance", "L1 Loss"])
    pivot_table = df.pivot(index="Sequence Length", columns="Max Distance", values="L1 Loss")

    # Print the table
    print("\nResults Table (L1 Loss):")
    print(pivot_table.to_string(float_format="{:.4f}".format))

    # Save the table as CSV
    runs_folder = "runs"
    os.makedirs(runs_folder, exist_ok=True)
    csv_path = os.path.join(runs_folder, f"{experiment_name}_results.csv")
    pivot_table.to_csv(csv_path, float_format="%.4f")
    print(f"\nResults saved to: {csv_path}")


def create_sequence_length_vs_loss_plot(results, experiment_name):
    df = pd.DataFrame(results, columns=["Sequence Length", "Max Distance", "L1 Loss"])
    df["Years (N/2)"] = df["Sequence Length"] / 2  # Convert sequence length to years
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

    # Save the plot
    runs_folder = "runs"
    os.makedirs(runs_folder, exist_ok=True)
    plot_path = os.path.join(runs_folder, f"{experiment_name}_sequence_length_vs_loss.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nPlot saved to: {plot_path}")


def train_and_eval_multiple_models(use_pretrained):
    torch.manual_seed(1)
    np.random.seed(1)

    # Create a list of models to evaluate
    models = [
        # (LinearNDVIModel(sequence_length=6), "linear_test_seq6", 0),
        # (LinearNDVIModel(sequence_length=11), "linear_test_seq11", 0),
        # (TSLinearEmbeddingNDVIModel(sequence_length=6, ndvi_hidden_size=64, embed_size=15), "linear_embed_seq6", 0),
        # (TSLinearEmbeddingNDVIModel(sequence_length=11, ndvi_hidden_size=64, embed_size=15), "linear_embed_seq11", 0),
        # (FullyConnectedNDVIModel(sequence_length=6, hidden_size=64, num_layers=3), "fc_seq5_layers3", 0),
        # (TSFCEmbeddingNDVIModel(sequence_length=6, ndvi_hidden_size=64, embed_size=15, hidden_sizes=[64, 64]), "ts_fc_5", 0),
        # (TSFullyConnectedNDVIModel(sequence_length=5, num_years=41, num_seasons=2, year_embed_size=16, season_embed_size=16, hidden_sizes=[64, 64]), "ts_fc_5", 0),
        # (FullyConnectedNDVIModel(sequence_length=5, hidden_size=64, num_layers=2), "fc_seq5_layers2", 0),
        # (FullyConnectedNDVIModel(sequence_length=10, hidden_size=64, num_layers=2), "fc_seq10_layers2", 0),
        # (FullyConnectedNDVIModel(sequence_length=20, hidden_size=64, num_layers=2), "fc_seq20_layers2", 0),
        # (FullyConnectedNDVIModel(sequence_length=5, hidden_size=64, num_layers=4), "fc_seq5_layers4", 0),
        # (FullyConnectedNDVIModel(sequence_length=10, hidden_size=64, num_layers=4), "fc_seq10_layers4", 0),
        # (FullyConnectedNDVIModel(sequence_length=20, hidden_size=64, num_layers=4), "fc_seq20_layers4", 0),
        # Add more models here
        # (TSFullyConnectedNDVIModel(sequence_length=5, num_years=41, num_seasons=2, year_embed_size=16, season_embed_size=16, hidden_sizes=[64, 64]), "ts_fc_5", 0),
        (FullyConnectedNDVIModel(sequence_length=11, hidden_size=64, num_layers=3), "fc_seq11_layers3", 0),
        (TemporalPositionalNDVITransformer(embedding_dim=256, num_encoder_layers=3, sequence_length=11, start_year=1984, end_year=2024, attn_heads=8, dropout=0.2), "transformer_seq11", 0),
    ]

    main(models, use_pretrained)


if __name__ == "__main__":
    # Choose which evaluation to run by uncommenting the desired function call
    use_pretrained = True  # Set this to False if you want to train models from scratch

    # Run the existing evaluation
    train_and_eval_multiple_models(use_pretrained)

    # Run the new grid evaluation for FullyConnectedNDVIModel
    # evaluate_fc_models_grid(use_pretrained)
