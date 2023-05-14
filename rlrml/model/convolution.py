import itertools
import torch
from ..playlist import Playlist
from .. import util


class TemporalReplayConvolution(torch.nn.Module):

    def __init__(
            self, header_info, playlist: Playlist, kernel_sizes=10,
            stride=1, filter_counts=30, layer_count=2, activation_fn=torch.nn.ReLU,
            pooling_fn=torch.nn.MaxPool1d
    ):
        super().__init__()
        self._playlist = playlist
        self._stride = stride
        self._header_info = header_info
        self._activation_fn = activation_fn
        self._pooling_fn = pooling_fn
        self._filter_counts = (
            itertools.cycle([filter_counts])
            if isinstance(filter_counts, int)
            else filter_counts
        )
        self._kernel_sizes = (
            itertools.cycle([kernel_sizes])
            if isinstance(kernel_sizes, int)
            else kernel_sizes
        )
        self._convolutional_widths = itertools.chain(
            [util.feature_count_for(playlist, header_info)], self._filter_counts
        )
        self._layers = [
            self._create_and_register_layers(layer_index, kernel_size, in_channels, out_channels)
            for layer_index, (kernel_size, (in_channels, out_channels)) in enumerate(zip(
                self._kernel_sizes, util.nwise(self._convolutional_widths, 2)
            ))
        ]

    def _create_and_register_layers(self, layer_index, kernel_size, in_channels, out_channels):
        convolution = self._create_convolutional_layer(in_channels, out_channels, kernel_size)
        pooling = self._create_pooling_layer(kernel_size)
        batch_normalization = torch.nn.BatchNorm1d(out_channels)
        activation = self._activation_fn()
        self.add_module(f"convolution {layer_index}", convolution)
        self.add_module(f"pooling {layer_index}", pooling)
        self.add_module(f"batch normalization {layer_index}", batch_normalization)
        self.add_module(f"activation {layer_index}", activation)
        return convolution, pooling, batch_normalization, activation

    def _create_convolutional_layer(self, in_channels, out_channels, kernel_size):
        return torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=self._stride)

    def _create_pooling_layer(self, kernel_size):
        return self._pooling_fn(kernel_size)

    def forward(self, X):
        last_output = X
        for layer_elements in self._layers:
            for layer in layer_elements:
                last_output = layer(last_output)
        return last_output
