import numpy as np
import torch
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pickle


class SVMNDVIModel(torch.nn.Module):

    def __init__(self, sequence_length: int, kernel="rbf", C=1.0, epsilon=0.1):
        super(SVMNDVIModel, self).__init__()
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)  # type: ignore
        self.is_fitted = False
        self.is_trained = False

    def forward(self, ndvi: torch.Tensor, years: torch.Tensor, seasons: torch.Tensor) -> torch.Tensor:
        return self.predict(ndvi)

    def fit(self, ndvi: np.ndarray, targets: np.ndarray):
        # Scale features
        ndvi_scaled = self.scaler.fit_transform(ndvi)

        # Fit the SVM model
        self.model.fit(ndvi_scaled, targets)
        self.is_trained = True
        self.is_fitted = True

    def predict(self, ndvi: torch.Tensor) -> torch.Tensor:
        # Convert torch tensor to numpy array
        print(type(ndvi))
        ndvi_np = ndvi.cpu().numpy()

        # Reshape ndvi to 2D array if it's not already
        if ndvi_np.ndim > 2:
            ndvi_np = ndvi_np.reshape(ndvi_np.shape[0], -1)

        # Scale features
        if not self.is_fitted:
            ndvi_scaled = self.scaler.fit_transform(ndvi_np)
            self.is_fitted = True
        else:
            ndvi_scaled = self.scaler.transform(ndvi_np)

        # Predict using the SVM model
        if not self.is_trained:
            # If the model is not trained, return zeros
            predictions = np.zeros(ndvi_np.shape[0])
        else:
            predictions = self.model.predict(ndvi_scaled)

        # Convert predictions back to torch tensor
        return torch.from_numpy(predictions).float().to(ndvi.device)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
        self.is_trained = True
        self.is_fitted = True
