from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import BERT, BERTPrediction
import logging
from datetime import datetime
import matplotlib.pyplot as plt

torch.manual_seed(0)


class LossLogger:
    def __init__(self, experiment_name):
        self.writer = SummaryWriter(f"runs/{experiment_name}")
        self.train_losses = {}
        self.valid_losses = {}

    def log_train_loss(self, loss, epoch):
        self.train_losses[epoch] = loss
        self.writer.add_scalar("Loss/train", loss, epoch)

    def log_valid_loss(self, loss, epoch):
        self.valid_losses[epoch] = loss
        self.writer.add_scalar("Loss/validation", loss, epoch)

    def log_lr_decay(self, lr, epoch):
        self.writer.add_scalar("cosine_lr_decay", lr, global_step=epoch)

    def plot_losses(self, experiment_name):
        plt.figure(figsize=(10, 5))
        epochs = list(self.train_losses.keys())
        train_losses = list(self.train_losses.values())
        valid_losses = list(self.valid_losses.values())
        plt.plot(epochs, train_losses, label="train")
        plt.plot(epochs, valid_losses, label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss")
        plt.savefig(f"runs/{experiment_name}/loss_plot.png")
        plt.close()


class BERTTrainer:

    def __init__(
        self,
        bert: BERT,
        num_features: int,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        decay_gamma: float = 0.99,
        gradient_clipping_value=5.0,
        with_cuda: bool = True,
        cuda_devices=None,
        experiment_name: str = "experiment",
    ):

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")

        self.bert = bert
        self.model = BERTPrediction(bert, num_features).to(self.device)

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.optim = Adam(self.model.parameters(), lr=lr)
        self.warmup_epochs = warmup_epochs
        self.optim_schedule = lr_scheduler.ExponentialLR(self.optim, gamma=decay_gamma)
        self.gradient_clippling = gradient_clipping_value
        self.criterion = nn.MSELoss(reduction="mean")
        self.experiment_name = f"{datetime.now().strftime('%b%d_%H-%M-%S')}_{experiment_name}"

        if with_cuda and torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                print("Using %d GPUs for model training" % torch.cuda.device_count())
                self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
            torch.backends.cudnn.benchmark = True

        self.loss_logger = LossLogger(self.experiment_name)

    def train(self, epoch):
        logging.info("Training model on device: %s", self.device)

        self.model.train()

        data_iter = tqdm(enumerate(self.train_loader), desc="EP_%s:%d" % ("train", epoch), total=len(self.train_loader), bar_format="{l_bar}{r_bar}")

        train_loss = 0.0
        for i, data in data_iter:
            inputs, targets = data

            # inputs shape: torch.Size([512, 5, 2])
            # targets shape: torch.Size([512, 2])
            # split based on last dimension to get x and year_seq
            x, year_seq = inputs.split(1, dim=-1)

            # Resize year_seq to remove the last dimension
            # X shouldn't be resized cause we have num_features dimension

            # TODO - once I want to do "multi step prediction" I need to use the year from y
            year_seq = year_seq.squeeze(-1)

            # move to device
            x = x.to(self.device)
            year_seq = year_seq.to(self.device)
            targets = targets.to(self.device)

            # Call the model to get the prediction
            outputs = self.model(x, year_seq)

            # takes only the first dimension of the targets cause we don't need the year, also remove it from the outputs
            y = targets[:, 0]
            outputs = outputs[:, 0]
            loss = self.criterion(y, outputs)

            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clippling)
            self.optim.step()

            train_loss += loss.item()
            post_fix = {"epoch": epoch, "iter": i, "avg_loss": train_loss / (i + 1), "current_batch_loss": loss.item()}

            if i % 10 == 0:
                data_iter.write(str(post_fix))

        train_loss = train_loss / len(data_iter)
        self.loss_logger.log_train_loss(train_loss, epoch)

        valid_loss = self.validate()
        self.loss_logger.log_valid_loss(valid_loss, epoch)

        if epoch >= self.warmup_epochs:
            self.optim_schedule.step()
        self.loss_logger.log_lr_decay(self.optim_schedule.get_lr()[0], epoch)

        print("EP%d, train_loss=%.5f, validate_loss=%.5f" % (epoch, train_loss, valid_loss))
        return train_loss, valid_loss

    def validate(self):
        logging.info("Validating model on device: %s", self.device)
        self.model.eval()

        valid_loss = 0.0
        counter = 0
        total_batches = len(self.valid_loader)
        # Initialize tqdm for progress tracking with a better description

        data_iter = tqdm(enumerate(self.valid_loader), desc="EP_valid", total=len(self.valid_loader), bar_format="{l_bar}{r_bar}")
        for i, (inputs, targets) in data_iter:
            x, year_seq = inputs.split(1, dim=-1)
            year_seq = year_seq.squeeze(-1)
            x = x.to(self.device)
            year_seq = year_seq.to(self.device)
            targets = targets.to(self.device)

            with torch.no_grad():
                outputs = self.model(x, year_seq)

                # takes only the first dimension of the targets cause we don't need the year, also remove it from the outputs
                y = targets[:, 0]
                outputs = outputs[:, 0]
                loss = self.criterion(y, outputs)

            valid_loss += loss.item()
            counter += 1

            if i % 10 == 0:
                post_fix = {"validation_iter": i, "validation_avg_loss": valid_loss / (i + 1), "validation_current_batch_loss": loss.item()}
                data_iter.write(str(post_fix))

        valid_loss /= counter
        return valid_loss

    def save(self, epoch, path):
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info(f"Created directory: {path}")

        output_path = os.path.join(path, f"checkpoint_{self.experiment_name}.tar")
        logging.info(f"Saving model checkpoint to: {output_path}")

        # Save model and optimizer state
        try:
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optim.state_dict(),
                },
                output_path,
            )
            logging.info("Model and optimizer state saved successfully.")
        except Exception as e:
            logging.error(f"Error saving model checkpoint: {e}")

        bert_path = os.path.join(path, f"checkpoint_{self.experiment_name}.bert.tar")
        logging.info(f"Saving BERT state to: {bert_path}")

        try:
            torch.save(self.bert.state_dict(), bert_path)
            self.bert.to(self.device)
            logging.info("BERT state saved successfully.")
        except Exception as e:
            logging.error(f"Error saving BERT state: {e}")

        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def load(self, path):
        input_path = os.path.join(path, f"checkpoint_{self.experiment_name}.tar")

        try:
            checkpoint = torch.load(input_path, map_location=torch.device("cpu"))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
            self.model.train()

            print("Model loaded from:" % input_path)
            return input_path
        except IOError:
            print("Error: parameter file does not exist!")

    def plot_losses(self):
        self.loss_logger.plot_losses(self.experiment_name)
