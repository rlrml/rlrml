import torch
from torch import nn


def build_default_model(
        headers, label_count, kernel_size=10, stride=2,
        dropout=.2, lstm_width=32
):
    input_width = len(headers)
    model = nn.Sequential(
        # nn.Conv1d(input_width, input_width, kernel_size, stride=stride),
        # nn.BatchNorm1d(input_width),
        # nn.MaxPool1d(kernel_size),
        # nn.BatchNorm1d(input_width),
        nn.LSTM(input_width, lstm_width, batch_first=True, dropout=.2, num_layers=3),
        # nn.LSTM(lstm_width, lstm_width, batch_first=True, dropout=.2, num_layers=1),
        nn.Linear(lstm_width, label_count)
    )
    return model


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    import ipdb; ipdb.set_trace()
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
