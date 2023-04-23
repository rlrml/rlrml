import torch
import logging
import numpy as np

from . import load
from . import build


logger = logging.getLogger(__name__)


def log_epoch_start(_trainer, epoch):
    pass


def log_batch_finish(_trainer, epoch, losses, loss, X, y_pred, y):
    # from .. import util
    # for p, p_pred in zip(y, y_pred):
    #     for pmmr, ammr in zip(p, p_pred):
    #         pmmr = util.HorribleHackScaler.unscale(float(pmmr))
    #         ammr = util.HorribleHackScaler.unscale(float(ammr))
    #         logger.info(f"{pmmr} - {ammr} = {pmmr - ammr}")
    logger.info(f"Epoch {epoch} finished with {loss:,}, gpu_free: {gpu_memory_remaining()}")


class WeightedByMSELoss(torch.nn.Module):
    def __init__(self, weight_by=None):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss(reduction="none")
        self.add_module("mse_loss", self.mse_loss)
        self._weight_by = weight_by

    def forward(self, inputs, target, weights=None):
        if weights is None:
            if self._weight_by is None:
                raise Exception(
                    "Can not call WeightedByMSELoss with no weights without setting weight_by."
                )
            weights = torch.tensor(
                [self._weight_by(actual) for actual in target], dtype=torch.float32
            )
            weights = torch.stack([weights, weights]).T

        # Ensure that input, target, and weights have the same shape
        assert inputs.shape == target.shape == weights.shape, (
            "Shapes of input, target, and weights must be the same"
        )

        # Compute element-wise MSE loss (without reducing)
        mse_loss = self.mse_loss(inputs, target)

        weights = weights.to(mse_loss.device)
        # Apply the weights
        weighted_mse_loss = mse_loss * weights

        # Mean of the weighted MSE loss
        loss = torch.mean(weighted_mse_loss)

        return loss


def create_weight_function(
        target_distance, target_weight, min_weight=1.0, max_weight=5.0,
):
    scaling_factor = 2 * float(target_weight) / target_distance

    # Define the function to calculate weights
    def calculate_weights(labels):
        # Calculate the mean absolute deviation of the labels
        labels = list(labels.cpu())
        mean = np.mean(labels, axis=0)
        mad = np.mean(np.abs(labels - mean), axis=0)

        weight = scaling_factor * mad

        # Clip the weights to the range [min_weight, max_weight]
        weight = np.clip(weight, min_weight, max_weight)

        return weight

    return calculate_weights


class ReplayModelManager:

    @classmethod
    def from_dataset(cls, dataset, model=None, *args, **kwargs):
        model = model or build.ReplayModel(dataset.header_info, dataset.playlist)
        data_loader = load.batched_packed_loader(dataset)
        return cls(model, data_loader, *args, **kwargs)

    def __init__(
            self, model, data_loader: torch.utils.data.DataLoader,
            use_cuda=None, on_epoch_start=log_epoch_start,
            on_epoch_finish=log_batch_finish, loss_function=None,
    ):
        use_cuda = torch.cuda.is_available() if use_cuda is None else use_cuda
        self._device = torch.device("cuda" if use_cuda else "cpu")
        self._model = model.to(self._device)
        self._data_loader = data_loader
        self._loss_function = loss_function or torch.nn.MSELoss()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=.001)
        self._on_epoch_start = on_epoch_start
        self._on_epoch_finish = on_epoch_finish

    def train(self, epochs=10):
        logger.info(f"Starting training for {epochs} epochs on {self._device}")
        batch_iterator = iter(self._data_loader)
        for epoch in range(epochs):

            try:
                X, y = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(self._data_loader)
                X, y = next(batch_iterator)

            self._on_epoch_start(self, epoch)
            X, y = X.to(self._device), y.to(self._device)
            y_pred = self._model(X)
            loss = self._loss_function(y_pred, y)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            self._on_epoch_finish(self, epoch, [], loss, X, y_pred, y)

    def get_total_loss(self):
        losses = []
        for batch_number, (X, y) in enumerate(self._data_loader):
            X = X.to(self._device)
            y = y.to(self._device)
            y_pred = self._model(X)
            loss = self._loss_function(y_pred, y)
            loss = float(loss)
            losses.append((loss, len(y)))
            self._on_epoch_finish(self, batch_number, losses, loss, X, y_pred, y)

        total_samples_counted = sum(count for _, count in losses)
        weighted_loss = sum(loss * count for loss, count in losses) / total_samples_counted
        return weighted_loss


def gpu_memory_remaining():
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return reserved - allocated
