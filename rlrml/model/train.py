import torch
import logging

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
