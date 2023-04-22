import numpy as np
import rich
from .. import util


class TrainLiveStatsDisplay:

    def __init__(self, live, scaler=None, last_n=30):
        self._live = live
        self._losses = []
        self._scaler = scaler or util.HorribleHackScaler
        self._last_n = last_n

    def _build_table(self, epoch, y_pred, y):
        table = rich.table.Table()
        table.add_column("Epoch")
        table.add_column("Loss")
        table.add_column(f"Last {self._last_n} Loss")
        table.add_column(f"Penultimate {self._last_n} Loss")
        table.add_column("Improvement")
        table.add_column("Last 100 Loss")

        last_one_hundred = self._losses[-100:]
        last = self._losses[-self._last_n:]
        penultimate = self._losses[-2 * self._last_n: -self._last_n]
        table.add_row(
            f"{epoch}", f"{self._losses[-1]}",
            f"{np.mean(last)}",
            f"{np.mean(penultimate)}",
            f"{np.mean(penultimate) - np.mean(last)}"
            f"{np.mean(self._losses[-100:])}",
        )
        return table

    def on_epoch_finish(self, trainer, epoch, _losses, loss, X, y_pred, y):
        self._losses.append(float(loss))
        self._live.update(self._build_table(epoch, y_pred, y), refresh=True)
