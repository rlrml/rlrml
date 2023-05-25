import numpy as np
import rich
from .. import util


class TrainLiveStatsDisplay:

    def __init__(self, live, scaler=None, last_n=30, loss_transformer=np.sqrt):
        self._live = live
        self._losses = []
        self._scaler = scaler or util.HorribleHackScaler
        self._last_n = last_n
        self._loss_transformer = loss_transformer

    def _build_table(self, epoch, y_pred, y, uuids):
        table = rich.table.Table()
        table.add_column("Epoch")
        table.add_column("Loss")
        table.add_column(f"Last {self._last_n} Loss")
        table.add_column(f"Penultimate {self._last_n} Loss")
        table.add_column("Improvement")
        table.add_column("Prediction")
        table.add_column("Actual")
        table.add_column("UUID")
        table.add_column("Max Difference")

        prediction_extrema = [
            (float(preds.max()), float(preds.min()), preds, act, uuid)
            for preds, act, uuid in zip(y_pred, y, uuids)
        ]
        prediction_extrema.sort(key=lambda v: v[0] - v[1], reverse=True)

        last = self._losses[-self._last_n:]
        last_mean = np.mean(last)
        penultimate = self._losses[-2 * self._last_n: -self._last_n]
        table.add_row(
            f"{epoch:.5f}",
            f"{self._losses[-1]:.5f}",
            f"{last_mean:.5f}",
            f"{np.mean(penultimate):.5f}",
            f"{np.mean(penultimate) - last_mean:.5f}",
            f"{[int(i) for i in self._scaler.unscale(prediction_extrema[0][2])]}",
            f"{[int(i) for i in self._scaler.unscale(prediction_extrema[0][3])]}",
            f"{prediction_extrema[0][4]}",
            f"{self._scaler.unscale_no_translate(prediction_extrema[0][0] - prediction_extrema[0][1])}"
        )
        return table

    def on_epoch_finish(self, loss, epoch, y_pred, y, uuids, **kwargs):
        self._losses.append(self._scaler.unscale_no_translate(
            self._loss_transformer(float(loss))
        ))
        self._live.update(self._build_table(epoch, y_pred, y, uuids), refresh=True)
