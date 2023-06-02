import argparse
import boxcars_py
import datetime
import logging
import os

from collections import deque

from . import player_cache as pc
from . import metadata


logger = logging.getLogger(__name__)


def _constant_retry(constant):
    def get_value(exception):
        value_to_return = constant
        return value_to_return
    return get_value


def get_replay_uuids_in_directory(filepath, replay_extension="replay"):
    for root, _, files in os.walk(filepath):
        for filename in files:
            replay_id, extension = os.path.splitext(filename)
            if extension and extension[1:] == replay_extension:
                replay_path = os.path.join(root, filename)
                yield replay_id, replay_path


def player_data_present(replay_path, player_cache: pc.PlayerCache):
    meta = metadata.ReplayMeta.from_boxcar_frames_meta(
        boxcars_py.get_replay_meta(replay_path)
    )
    return all(player_cache.present_and_no_error(player) for player in meta.player_order)


def closest_date_value(pairs, target_date):
    min_difference = None
    closest_pair = None, None

    for date, value in pairs:
        if isinstance(date, datetime.datetime):
            date = date.date()
        difference = abs(target_date - date)

        if min_difference is None or difference < min_difference:
            min_difference = difference
            closest_pair = (date, value)

    return closest_pair


def symlink_replays(target_directory, replay_uuids, replay_set):
    os.makedirs(target_directory, exist_ok=True)
    for uuid in replay_uuids:
        target_path = os.path.join(target_directory, f"{uuid}.replay")
        if not os.path.exists(target_path):
            os.symlink(
                replay_set.replay_path(uuid), os.path.join(target_directory, f"{uuid}.replay")
            )


def feature_count_for(playlist, header_info):
    return (
        len(header_info['global_headers']) + (
            playlist.player_count * len(header_info['player_headers'])
        )
    )


def segment_list(input_list, k):
    if k <= 0:
        raise ValueError("Segment size must be greater than 0")

    for i in range(0, len(input_list), k):
        yield input_list[i:i + k]


def nwise(iterable, n=2):
    # Initialize a sliding window deque with max length n
    it = iter(iterable)
    d = deque((next(it, None) for _ in range(n)), maxlen=n)
    # Check if the deque has None (happens when the iterable is shorter than n)
    if None in d:
        raise ValueError("Iterable is shorter than the sliding window length")
    yield tuple(d)
    # Slide the window over the rest of the iterator
    for elem in it:
        d.append(elem)
        yield tuple(d)


class ManualLinearScaler:

    def __init__(self, data_min=0.0, data_max=1.0, target_min=-1.0, target_max=1.0):
        self._data_min = data_min
        self._data_max = data_max
        self._target_min = target_min
        self._target_max = target_max
        self._data_range = self._data_max - self._data_min
        self._target_range = self._target_max - self._target_min

    def scale_no_translate(self, value):
        return value * self._target_range / self._data_range

    def unscale_no_translate(self, value):
        return value * self._data_range / self._target_range

    def scale(self, value):
        proportion_to_max = (value - self._data_min) / (self._data_range)
        return proportion_to_max * self._target_range + self._target_min

    def unscale(self, value):
        # Apply the inverse operations in reverse order
        unscaled_value = value - self._target_min
        unscaled_value /= self._target_range
        unscaled_value *= self._data_range
        unscaled_value += self._data_min

        return unscaled_value

    def scale_column_in_place(self, matrix, column_index):
        matrix[:, column_index] -= self._data_min
        matrix[:, column_index] /= self._data_range
        matrix[:, column_index] *= self._target_range
        matrix[:, column_index] += self._target_min

    def unscale_column_in_place(self, matrix, column_index):
        matrix[:, column_index] -= self._target_min
        matrix[:, column_index] /= self._target_range
        matrix[:, column_index] *= self._data_range
        matrix[:, column_index] += self._data_min


class RatioScaler:

    def __init__(self, ratio=2.0):
        self._ratio = ratio

    def scale(self, value):
        proportion_to_max = (value - self._data_min) / (self._data_range)
        return proportion_to_max * self._target_range + self._target_min

    def unscale(self, value):
        return value / self._ratio

    def scale_column_in_place(self, matrix, column_index):
        matrix[:, column_index] *= self._ratio

    def unscale_column_in_place(self, matrix, column_index):
        matrix[:, column_index] /= self._ratio

    scale_no_translate = scale


default_position_scaler = RatioScaler(ratio=1.0 / 600.0)


class ReplayPositionRescaler:

    def __init__(self, column_headers, playlist, scaler=default_position_scaler):
        self._header_indices = [
            h[0] for h in enumerate(column_headers['global_headers']) if 'position' in h[1]
        ]
        player_indices = [
            h[0] for h in enumerate(column_headers['player_headers']) if 'position' in h[1]
        ]
        global_header_count = len(column_headers['global_headers'])
        player_header_count = len(column_headers['player_headers'])
        all_player_indices = [
            i + p * player_header_count + global_header_count
            for p in range(playlist.player_count)
            for i in player_indices
        ]
        self._header_indices += all_player_indices
        self._scaler = scaler

    def scale_position_columns(self, tensor):
        for i in self._header_indices:
            self._scaler.scale_column_in_place(tensor, i)
        return tensor


HorribleHackScaler = ManualLinearScaler(300.0, 1800.0, -3.0, 4.5)


def get_argparse_settable_variables(parser: argparse.ArgumentParser):
    destination_to_info = {}

    def set_from_attr(config_property_info, action, key, attr=None):
        if attr is None:
            attr = key
        value = getattr(action, attr)
        if (
                config_property_info.get(key) is not None and
                config_property_info.get(key) != argparse.SUPPRESS and
                value is not None
        ):
            current = config_property_info[key]
            if current != value:
                logger.warn(f"Mismatch '{attr}': {value} , existing {current}")
        elif key not in config_property_info or value is not None:
            config_property_info[key] = value

    simple_sets = ["type", "choices", "default", "dest"]
    for action in parser._actions:
        destination_to_info.setdefault(action.dest, dict())
        config_property_info = destination_to_info[action.dest]
        for key in simple_sets:
            set_from_attr(config_property_info, action, key)
        config_property_info.setdefault('consts', []).append(action.const)

    for value in destination_to_info.values():
        if value['type'] is None:
            if all(v is True or v is False for v in value['consts']):
                value['type'] = bool
                value['choices'] = [True, False]
            elif value['default'] is not None:
                value['type'] = type(value['default'])
        del value['consts']

    del destination_to_info['help']

    return destination_to_info
