import boxcars_py
import datetime
import os

from collections import deque

from . import player_cache as pc
from . import _replay_meta


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


def get_cache_answer_uuids_in_directory(filepath, player_cache: pc.PlayerCache):
    for uuid, filepath in get_replay_uuids_in_directory(filepath):
        try:
            data_present = player_data_present(filepath, player_cache)
        except Exception as e:
            print(f"Exception {e}")
            continue
        else:
            if data_present:
                yield uuid, filepath


def player_data_present(replay_path, player_cache: pc.PlayerCache):
    meta = _replay_meta.ReplayMeta.from_boxcar_frames_meta(
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
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
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


class HorribleHackScaler:

    @classmethod
    def scale_no_translate(cls, value):
        return value / 200.0

    @classmethod
    def scale(cls, label):
        return cls.scale_no_translate(label - 900.0)

    @classmethod
    def unscale(cls, label):
        return label * 200.0 + 900.0
