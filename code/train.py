import torch
import numpy as np
import random
import argparse
from dataset import LandsatDataLoader
import logging
from model import BERT
from trainer import BERTTrainer


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def Config():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--dataset_path", type=str, help="Path to the unlabeled dataset.", default="data/Landsat_NDVI_time_series_1984_to_2024.tif")
    parser.add_argument(
        "--checkpoints_path",
        default="checkpoints/",
        type=str,
        required=False,
        help="The output directory where the training checkpoints will be written.",
    )
    parser.add_argument(
        "--with_cuda",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether cuda is available.",
    )
    parser.add_argument(
        "--cuda_devices",
        default=None,
        type=list,
        help="List of cuda devices.",
    )
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Number of loader worker processes.",
    )
    parser.add_argument("--split_rate", default=0.1, type=float, help="Proportion of samples used for validation")
    parser.add_argument(
        "--max_length",
        default=75,
        type=int,
        help="The maximum length of input time series. Sequences longer " "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--num_features",
        default=1,
        type=int,
        help="The spectral dimensionality of satellite observations.",
    )
    parser.add_argument(
        "--hidden_size",
        default=256,
        type=int,
        help="Number of hidden neurons of the Transformer network.",
    )
    parser.add_argument(
        "--layers",
        default=3,
        type=int,
        help="Number of layers of the Transformer network.",
    )
    parser.add_argument(
        "--attn_heads",
        default=8,
        type=int,
        help="Number of attention heads of the Transformer network.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-3,
        type=float,
        help="",
    )
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="",
    )
    parser.add_argument(
        "--batch_size",
        default=1024,
        type=int,
        help="",
    )
    parser.add_argument(
        "--warmup_epochs",
        default=10,
        type=int,
        help="",
    )
    parser.add_argument(
        "--decay_gamma",
        default=0.99,
        type=float,
        help="",
    )
    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="",
    )
    parser.add_argument(
        "--gradient_clipping",
        default=5.0,
        type=float,
        help="",
    )
    parser.add_argument("--window_size", default=5, type=int, help="time series window size of the data - how many timesteps to look at to predict the next timestep")
    parser.add_argument("--name", type=str, default="experiment", help="Name of the experiment for TensorBoard logging")
    return parser.parse_args()


def main():
    setup_seed(0)
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
    config = Config()

    logging.info("Loading datasets...")
    loader = LandsatDataLoader(config.dataset_path, config.batch_size, config.split_rate, config.num_workers, config.window_size)

    train_loader, val_loader = loader.create_data_loaders()

    # NDVI Data so num_features = 1
    bert = BERT(num_features=1, hidden=256, n_layers=config.layers, attn_heads=config.attn_heads, dropout=config.dropout)

    trainer = BERTTrainer(
        bert,
        config.num_features,
        train_loader,
        val_loader,
        lr=config.learning_rate,
        warmup_epochs=config.warmup_epochs,
        decay_gamma=config.decay_gamma,
        gradient_clipping_value=config.gradient_clipping,
        with_cuda=config.with_cuda,
        cuda_devices=config.cuda_devices,
        experiment_name=config.name,
    )

    mini_loss = np.Inf
    for epoch in range(config.epochs):
        train_loss, valida_loss = trainer.train(epoch)
        if mini_loss > valida_loss:
            mini_loss = valida_loss
            trainer.save(epoch, config.checkpoints_path)


if __name__ == "__main__":
    main()
