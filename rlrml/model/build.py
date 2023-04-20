import torch
from torch import nn

nn.Sequential()
def build_default_model(
        *args, **kwargs
):
    return ReplayModel(*args, **kwargs)


class ReplayModel(nn.Module):

    def __init__(
            self, headers, label_count, kernel_size=10, stride=2,
            dropout=.2, lstm_width=512
    ):
        super().__init__()
        self._input_width = len(headers)
        self._label_count = label_count
        self._kernel_size = kernel_size
        self._dropout = dropout
        self._lstm_width = lstm_width

        self._input_lstm = nn.LSTM(
            self._input_width, self._lstm_width, batch_first=True, dropout=.12, num_layers=2
        )
        # self._middle_lstm = nn.LSTM(
        #     self._lstm_width, self._lstm_width, batch_first=True, dropout=.12, num_layers=1
        # )
        # self._output_lstm = nn.LSTM(
        #     self._lstm_width, self._label_count, batch_first=True, num_layers=1
        # )

        self._linear = nn.Linear(self._lstm_width, self._label_count)

        self.add_module("input_lstm", self._input_lstm)
        # self.add_module("middle_lstm", self._middle_lstm)
        # self.add_module("output_lstm", self._output_lstm)
        self.add_module("linear", self._linear)

    def forward(self, X):
        lstm_out, _ = self._input_lstm(X)
        # second, _ = self._middle_lstm(first)
        # third, (_hidden_1, _hidden_2) = self._output_lstm(first)
        linear_out = self._linear(lstm_out[:, -1])

        return linear_out


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))


class ConvolutionalReplayModel:

    def __init__(self, headers, label_count, kernel_size=10, stride=2, dropout=.2):
        nn.Conv1d(len(headers), len(headers), kernel_size, stride=stride)
        nn.MaxPool1d(kernel_size)
        nn.BatchNorm1d()
        nn.LSTM(len(headers), 4096, batch_first=True, dropout=.2, num_layers=2)

    def convolutional_layers(self, headers):
        pass

    def convolutional_layer_from_indices(self, index_ranges, *args, **kwargs):
        for index_range in index_ranges:
            torch.nn.Conv1d()

    def build(self):
        model = torch.nn.Sequential()
