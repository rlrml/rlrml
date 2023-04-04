"""Load replays into memory into a format that can be used with torch."""
import abc
import datetime
import itertools
import json
import logging
import numpy as np
import os
import torch

from pathlib import Path
from boxcars_py import parse_replay
from carball_lite import game
from torch.utils.data import Dataset

from . import player_cache as pc
from . import mmr
from ._replay_meta import ReplayMeta, PlatformPlayer


logger = logging.getLogger(__name__)


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


class ReplaySet(abc.ABC):
    """Abstract base class for objects representing a collection of replays."""

    @classmethod
    def cached(cls, cache_directory, *args, **kwargs):
        """Return a cached version of this the replay set."""
        return CachedReplaySet(cls(*args, **kwargs), cache_directory, **kwargs)

    @abc.abstractmethod
    def get_replay_uuids(self) -> [str]:
        """Get the replay uuids that are a part of this dataset."""
        pass

    @abc.abstractmethod
    def get_replay_tensor(self, uuid) -> (torch.Tensor, ReplayMeta):
        """Get the replay tensor and player order associated with the provided uuid."""
        pass


class CachedReplaySet(ReplaySet):
    """Wrapper for a replay set that caches the tensors that are provided by `get_replay_tensor`."""

    _player_order_extension = "replay_meta"

    def __init__(
            self, replay_set: ReplaySet, cache_directory: Path, cache_extension="pt", **kwargs
    ):
        """Initialize the cached replay set."""
        self._replay_set = replay_set
        self._cache_directory = cache_directory
        self._cache_extension = cache_extension
        if not os.path.exists(self._cache_directory):
            os.makedirs(self._cache_directory)

    def get_replay_uuids(self):
        """Get the replay uuids that are a part of this dataset."""
        self._replay_set.get_replay_uuids()

    def _cache_path_for_replay_with_extension(self, replay_id, extension):
        return os.path.join(self._cache_directory, f"{replay_id}.{extension}")

    def _get_tensor_and_meta_path(self, uuid):
        tensor_path = self._cache_path_for_replay(uuid, self._cache_extension)
        meta_path = self._cache_path_for_replay(uuid, self._meta_extension)
        return tensor_path, meta_path

    def _maybe_load_from_cache(self, uuid):
        tensor_path, meta_path = self._get_tensor_and_meta_path()
        tensor_present = os.path.exists(tensor_path)
        meta_present = os.path.exists(meta_path)
        if tensor_present and meta_present:
            with open(tensor_path, 'rb') as f:
                tensor = torch.load(f)
            with open(meta_path, 'rb') as f:
                meta = ReplayMeta.from_dict(json.loads(f.read()))
        elif tensor_present != meta_present:
            logger.warn(
                f"{tensor_path} exists: {tensor_present}" +
                f"meta_present, {meta_path} exists: {meta_present}"
            )
            if not meta_present:
                try:
                    meta = self._replay_set.get_meta(uuid)
                except Exception as e:
                    logger.warn(f"Error trying to get only player meta {e}")
                else:
                    self._json_dump_meta(meta, meta_path)
                    return self._maybe_load_from_cache(uuid)

        return tensor, meta

    def _save_to_cache(self, replay_id, replay_data, meta):
        tensor_path, meta_path = self._get_tensor_and_meta_path()
        with open(tensor_path, 'wb') as f:
            torch.save(replay_data, f)
        self._json_dump_meta(meta, meta_path)
        return replay_data

    def _json_dump_meta(self, meta: ReplayMeta, path):
        with open(path, 'w') as f:
            f.write(json.dumps(ReplayMeta.to_dict()))

    def get_replay_tensor(self, uuid) -> (torch.Tensor, ReplayMeta):
        """Get the replay tensor and player meta associated with the provided uuid."""
        return self._maybe_load_from_cache(uuid) or self._save_to_cache(
            uuid, *self._replay_set.get_replay_tensor()
        )

    def __getattr__(self, name):
        """Defer to self._replay_set."""
        return getattr(self._replay_set, name)


class DirectoryReplaySet(ReplaySet):
    """A replay set that consists of replay files in a potentially nested directory."""

    def __init__(self, filepath, replay_extension="replay", backup_get_meta=lambda uuid: None):
        self._filepath = filepath
        self._replay_extension = replay_extension
        self._replay_id_paths = list(self._get_replay_ids())
        self._replay_path_dict = dict(self._replay_id_paths)
        self._backup_get_meta = backup_get_meta

    def _get_replay_ids(self):
        for root, _, files in os.walk(self._filepath):
            for filename in files:
                replay_id, extension = os.path.splitext(filename)
                if extension and extension[1:] == self._replay_extension:
                    replay_path = os.path.join(root, filename)
                    yield replay_id, replay_path

    def _replay_path(self, replay_id):
        return os.path.join(self._filepath, f"{replay_id}.{self._replay_extension}")

    def get_replay_uuids(self):
        return [replay_id for replay_id, _ in self._replay_id_paths]

    def get_replay_tensor(self, uuid) -> (torch.Tensor, ReplayMeta):
        """Get the replay tensor and player order associated with the provided uuid."""
        replay_path = self._replay_path_dict[uuid]
        converter = _CarballToTensorConverter(get_carball_game(replay_path))
        tensor = converter.get_tensor()

        try:
            meta = converter.get_meta()
        except Exception as e:
            meta = self._backup_get_meta(uuid)

            if meta is None:
                raise Exception(f"Could not build meta for {uuid}, carball error {e}")
            else:
                if not converter.check_meta_player_order_matches(meta):
                    raise Exception(f"The generated meta player order did not match the carball game order for {uuid}")

        return tensor, meta


def player_cache_label_lookup(
        player_cache: pc.PlayerCache,
        mmr_function=mmr.SeasonBasedPolyFitMMRCalculator.get_mmr_for_player_at_date,
        *args, **kwargs,
):
    def lookup_label_for_player(player: PlatformPlayer, game_time: datetime.datetime):
        data = player_cache.get_player_data(player)
        return mmr_function(game_time, data)

    return lookup_label_for_player


class ReplayDataset(Dataset):
    """Load data from rocket league replay files in a directory."""

    def __init__(self, replay_set: ReplaySet, lookup_label):
        """Initialize the data loader."""
        self._replay_set = replay_set
        self._replay_ids = list(replay_set.get_replay_uuids())
        self._lookup_label = lookup_label

    def __len__(self):
        """Simply return the length of the replay ids calculated in init."""
        return len(self._replay_ids)

    def __getitem__(self, index):
        """Get the replay at the provided index."""
        if index < 0 or index > len(self._replay_ids):
            raise KeyError
        replay_tensor, meta = self._replay_set.get_replay_tensor(
            self._replay_ids[index]
        )
        labels = torch.FloatTensor([
            self._lookup_label(player, meta.datetime)
            for player in meta.player_order
        ])

        return replay_tensor, labels


class _CarballToTensorConverter:

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

    def __init__(self, carball_game, include_time=True):
        """Initialize the converter."""
        self.carball_game = carball_game
        self.orange_team = list(
            next(team.players for team in carball_game.teams if team.is_orange)
        )
        self.blue_team = list(
            next(team.players for team in carball_game.teams if not team.is_orange)
        )
        self.orange_team.sort(key=lambda p: p.name)
        self.blue_team.sort(key=lambda p: p.name)
        self.player_order = self.orange_team + self.blue_team
        self.include_time = include_time

    def get_tensor_and_meta(self):
        """Return a :py:class:`torch.Tensor` and a :py:class:`ReplayMeta` from the :py:class:`game.Game`."""
        return self.get_tensor(), self.get_meta()

    def get_tensor(self):
        """Return a :py:class:`torch.Tensor` built from the provided carball game."""
        return torch.as_tensor(self.get_ndarray())

    def check_meta_player_order_matches(self, meta: ReplayMeta):
        for meta_player, carball_player in zip(meta.player_order, self.player_order):
            assert meta_player.matches_carball(carball_player)

    def get_meta(self, ensure_order_equality=True):
        meta = ReplayMeta.from_carball_game(self.carball_game)

        if ensure_order_equality:
            self.check_player_order(meta)

        return meta

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
