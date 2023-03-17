"""Load replays into memory into a format that can be used with torch."""
import itertools
import marshal
import numpy as np
import os
import torch

from boxcars_py import parse_replay
from carball_lite import game
from torch.utils.data import Dataset

from . import manifest
from . import tracker_network


def get_carball_game(replay_path):
    """Take a path to the replay and output a Game object associated with that replay.

    :param replay_path: Path to a specific replay.
    :return: A :py:class:`game.Game`.
    """
    with open(replay_path, 'rb') as f:
        buf = f.read()
    boxcars_data = parse_replay(buf)
    cb_game = game.Game()
    # This is confusingly called loaded_json, but what is expected is actually a
    # python object. In our case we are not loading from json, but directly from
    # the replay file, but this is fine.
    cb_game.initialize(loaded_json=boxcars_data)
    return cb_game


default_manifest_loader = manifest.ManifestLoader()
def default_label_lookup(rid, rfp, _):
    mmr_dict, player_metas = default_manifest_loader.lookup_labels_by_manifest_file(rid, rfp)
    for player, value in mmr_dict.items():
        if value is None:
            mmr_dict[player] = tracker_network.get_doubles_mmr_from_player_meta(player_metas[player])
    return mmr_dict


class ReplayDirectoryDataLoader(Dataset):
    """Load data from rocket league replay files in a directory."""

    def __init__(
            self, filepath, cache_directory_name="__cache",
            replay_extension="replay", cache_on_load=True,
            cache_extension="pt", labels_extension="labels",
            label_lookup=default_label_lookup, eager_labels=True,
    ):
        """Initialize the data loader."""
        self._filepath = filepath
        self._cache_directory_name = cache_directory_name
        self._replay_extension = replay_extension
        self._cache_on_load = cache_on_load
        self._eager_labels = eager_labels
        self._label_lookup = label_lookup
        self._cache_extension = cache_extension
        self._cache_directory = os.path.join(self._filepath, self._cache_directory_name)
        self._labels_filepath = os.path.join(
            self._cache_directory, f"labels.{labels_extension}"
        )
        if not os.path.exists(self._cache_directory):
            os.makedirs(self._cache_directory)
        self._labels_cache = {}
        # if not os.path.exists(self._labels_filepath):
        #     self._labels_cache = {}
        #     self._save_labels_cache()
        # else:
        #     self._labels_cache = self._load_labels_cache()

        self._replay_ids = list(self._get_replay_ids())

    def _get_replay_ids(self):
        for root, _, files in os.walk(self._filepath):
            for filename in files:
                replay_id, extension = os.path.splitext(filename)
                if extension and extension[1:] == self._replay_extension:
                    replay_path = os.path.join(root, filename)
                    if self._eager_labels:
                        label_data = self._label_lookup(replay_id, replay_path, None)
                        if any(value is None for value in label_data.values()):
                            print(f"Skipping {replay_id} because of missing label")
                            continue
                        self._labels_cache[replay_id] = label_data
                    yield replay_id, replay_path
        if self._eager_labels:
            self._save_labels_cache()

    def _cache_path_for_replay(self, replay_id):
        return os.path.join(self._cache_directory, f"{replay_id}.{self._cache_extension}")

    def _save_labels_cache(self):
        pass
        # with open(self._labels_filepath, 'wb') as f:
        #     marshal.dump(self._labels_cache, f)

    def _load_labels_cache(self):
        with open(self._labels_filepath, 'rb') as f:
            return marshal.load(f)

    def _maybe_load_from_cache(self, replay_id):
        path = self._cache_path_for_replay(replay_id)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return torch.load(f)

    def _save_replay_to_cache(self, replay_id, replay_data):
        path = self._cache_path_for_replay(replay_id)
        with open(path, 'wb') as f:
            torch.save(replay_data, f)

    def __len__(self):
        """Simply return the length of the replay ids calculated in init."""
        return len(self._replay_ids)

    def __getitem__(self, index):
        """Get the replay at the provided index."""
        replay_id, replay_path = self._replay_ids[index]

        from_cache = self._maybe_load_from_cache(replay_id)

        if from_cache is not None:
            return (from_cache, self._labels_cache[replay_id])

        carball_game = get_carball_game(replay_path)
        converter = _CarballToTensorConverter(carball_game)
        replay_data = converter.get_tensor()

        label_dict = self._label_lookup(replay_id, replay_path, carball_game)

        labels = torch.tensor([label_dict[player.name] for player in converter.player_order])

        if self._cache_on_load:
            self._save_replay_to_cache(replay_id, replay_data)
            self._labels_cache[replay_id] = labels
            self._save_labels_cache()

        return replay_data, labels


class _CarballToTensorConverter(object):

    PLAYER_COLUMNS = [
        'pos_x', 'pos_y', 'pos_z',
        'vel_x', 'vel_y', 'vel_z',
        'rot_x', 'rot_y', 'rot_z',
        'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
    ]

    # These are also available
    # 'ping', 'throttle', 'steer', 'handbrake', 'ball_cam', 'dodge_active',
    # 'boost', 'boost_active', 'jump_active', 'double_jump_active',

    BALL_COLUMNS = [
        'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z',
    ]

    # These are also available but probably not needed
    EXTRA_BALL_COLUMNS = [
        'rot_x', 'rot_y', 'rot_z', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'hit_team_no'
    ]

    @classmethod
    def from_filepath(cls, filepath):
        return cls(get_carball_game(filepath))

    def __init__(self, carball_game, include_time=True, ):
        """Initialize the converter."""
        self.carball_game = carball_game
        self.orange_team = list(
            next(team.players for team in carball_game.teams if team.is_orange)
        )
        self.blue_team = list(
            next(team.players for team in carball_game.teams if not team.is_orange)
        )
        self.player_order = self.orange_team + self.blue_team
        self.include_time = include_time

    def get_tensor(self):
        """Return a :py:class:`torch.Tensor` built from the provided carball game."""
        return torch.as_tensor(self.get_ndarray())

    def get_ndarray(self):
        """Return a numpy array from the provided carball game."""
        first_relevant_frame = self._calculate_first_relevant_frame()
        return np.stack([
            self._construct_numpy_frame(i)
            for i in self.carball_game.frames.index
            if i >= first_relevant_frame
        ])

    @classmethod
    def _get_data_frame_value_using_last_as_default(cls, column, carball_frame_index):
        if carball_frame_index < 0:
            # TODO: ..
            raise Exception("Shoudln't happen")
        if carball_frame_index in column.index:
            return column[carball_frame_index]
        # TODO: use a real logger
        print(f"Missing value at {carball_frame_index} for column {column.name}")
        cls._get_data_frame_value_using_last_as_default(column, carball_frame_index - 1)

    def _construct_numpy_frame(self, carball_frame_index):
        ball_values = (
            self._get_data_frame_value_using_last_as_default(
                self.carball_game.ball[column_name], carball_frame_index
            )
            for column_name in self.BALL_COLUMNS
        )
        player_values = (
            self._get_data_frame_value_using_last_as_default(
                player.data[column_name],
                carball_frame_index
            )
            for player in self.player_order
            for column_name in self.PLAYER_COLUMNS
        )
        return np.fromiter(
            itertools.chain(
                [self.carball_game.frames['time'][carball_frame_index]],
                ball_values,
                player_values
            ),
            dtype=float
        )

    def _calculate_first_relevant_frame(self):
        initial_time_remaining = self.carball_game.frames.seconds_remaining.iloc[0]
        for i, v in enumerate(self.carball_game.frames.seconds_remaining):
            if v > initial_time_remaining:
                raise Exception(
                    "There should never be more time remaining than what we started with"
                )
            elif v < initial_time_remaining:
                frame_index = i
                break
        else:
            raise Exception("Time remaining never decreased")

        time_at_most_one_second_in = self.carball_game.frames.time.iloc[frame_index]

        for index in range(frame_index, -1, -1):
            if time_at_most_one_second_in - self.carball_game.frames.time.iloc[index] > 1.01:
                first_relevant_index = index
                break
        else:
            raise Exception("This loop should terminate")

        first_relevant_frame = self.carball_game.frames.index[first_relevant_index]

        return first_relevant_frame
