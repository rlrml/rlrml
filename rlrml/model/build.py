import itertools

from torch import nn

from ..playlist import Playlist
from .. import util

from . import cnn


class ReplayModel(nn.Module):
    def __init__(
            self, header_info, playlist: Playlist, channel_counts=None,
            dropout=.05, lstm_width=256, lstm_depth=4, use_convolutional=False,
            evaluation_start=800, evaluation_split_width=100, **kwargs
    ):
        super().__init__()
        self._input_width = util.feature_count_for(playlist, header_info)
        self._label_count = playlist.player_count
        self._lstm_width = lstm_width
        self._evaluation_split_width = evaluation_split_width
        self._evaluation_start = evaluation_start

        next_layer_size = self._input_width
        if use_convolutional:
            channel_counts = list(itertools.chain([next_layer_size], channel_counts or [50]))
            self._cnn = cnn.TemporalReplayConvolution(channel_counts, **kwargs)
            next_layer_size = self._cnn.channel_counts[-1]
        else:
            self._cnn = nn.Identity()

        self.add_module("cnn", self._cnn)

        self._lstm = nn.LSTM(
            next_layer_size, self._lstm_width,
            batch_first=True, dropout=dropout, num_layers=lstm_depth
        )
        self._linear = nn.Linear(self._lstm_width, self._label_count)

        self.add_module("lstm", self._lstm)
        self.add_module("linear", self._linear)

    def get_lstm_out(self, X):
        cnn_out = self._cnn(X)
        lstm_out, _ = self._lstm(cnn_out)
        return lstm_out

    def forward(self, X):
        lstm_out = self.get_lstm_out(X)

        split_start = (
            lstm_out.shape[1] - 1
            if self._evaluation_split_width is None
            else self._evaluation_start
        )
        split_width = min(
            lstm_out.shape[1] if self._evaluation_split_width is None
            else self._evaluation_split_width, lstm_out.shape[1]
        )

        out_indices = list(range(split_start, lstm_out.shape[1], split_width))

        linear_outs = self._linear(lstm_out[:, out_indices, :])

        linear_out = linear_outs.mean(dim=1)

        return linear_out

    def prediction_history(self, X):
        lstm_out = self.get_lstm_out(X)

        return [self._linear(lstm_out[:, i]) for i in range(lstm_out.shape[1])]


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
