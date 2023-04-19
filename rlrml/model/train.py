import torch
import logging

from . import load
from . import build


logger = logging.getLogger(__name__)


def log_epoch_start(_trainer, epoch):
    logger.info(f"Starting epoch {epoch}")


def log_batch_finish(_trainer, epoch, loss):
    logger.info(f"Epoch finished with {loss}")


class ReplayModelTrainer:

    @classmethod
    def from_dataset(cls, dataset, *args, **kwargs):
        headers, label_count = dataset.get_shape_info()
        model = build.build_default_model(headers, label_count)
        build.get_model_size(model)
        data_loader = load.batched_packed_loader(dataset)
        return cls(model, data_loader, *args, **kwargs)

    def __init__(
            self, model, data_loader: torch.utils.data.DataLoader,
            use_cuda=None, on_epoch_start=log_epoch_start,
            on_epoch_finish=log_batch_finish
    ):
        use_cuda = torch.cuda.is_available() if use_cuda is None else use_cuda
        self._device = torch.device("cuda:0" if use_cuda else "cpu")
        self._model = model.to(self._device)
        self._data_loader = data_loader
        self._loss_function = torch.nn.MSELoss()
        self._optimizer = torch.optim.Adam(self._model.parameters())
        self._on_epoch_start = on_epoch_start
        self._on_epoch_finish = log_batch_finish

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
            logger.info(f"Batch shape: {X.shape}")
            y_pred = self._model(X)
            loss = self._loss_function(y_pred, y)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            self._on_epoch_finish(self, epoch, loss)
