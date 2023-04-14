"""Load replays into memory into a format that can be used with torch."""
import abc
import datetime
import json
import logging
import numbers
import os
import torch
import boxcars_py

from pathlib import Path
from torch.utils.data import Dataset

from . import mmr
from . import manifest
from ._replay_meta import ReplayMeta, PlatformPlayer
from . import util


logger = logging.getLogger(__name__)


default_manifest_loader = manifest.ManifestLoader()


def manifest_get_meta(_uuid, replay_path, manifest_loader=default_manifest_loader) -> ReplayMeta:
    """Get a `ReplayMeta` instance from a manifest file in the same directory as the replay file."""
    data = manifest_loader.get_raw_manifest_data_from_replay_filepath(replay_path)

    if data is None:
        return

    return ReplayMeta.from_ballchasing_game(data)


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

    _meta_extension = "replay_meta"

    def __init__(
            self, replay_set: ReplaySet, cache_directory: Path, cache_extension="pt",
            backup_get_meta=manifest_get_meta, **kwargs
    ):
        """Initialize the cached replay set."""
        self._replay_set = replay_set
        self._cache_directory = cache_directory
        self._cache_extension = cache_extension
        self._backup_get_meta = backup_get_meta
        if not os.path.exists(self._cache_directory):
            os.makedirs(self._cache_directory)

    def get_replay_uuids(self):
        """Get the replay uuids that are a part of this dataset."""
        return self._replay_set.get_replay_uuids()

    def _cache_path_for_replay_with_extension(self, replay_id, extension):
        return os.path.join(self._cache_directory, f"{replay_id}.{extension}")

    def _get_tensor_and_meta_path(self, uuid):
        tensor_path = self._cache_path_for_replay_with_extension(uuid, self._cache_extension)
        meta_path = self._cache_path_for_replay_with_extension(uuid, self._meta_extension)
        return tensor_path, meta_path

    def _maybe_load_from_cache(self, uuid):
        tensor_path, meta_path = self._get_tensor_and_meta_path(uuid)
        tensor_present = os.path.exists(tensor_path)
        meta_present = os.path.exists(meta_path)
        if tensor_present and meta_present:
            with open(tensor_path, 'rb') as f:
                tensor = torch.load(f)
            with open(meta_path, 'rb') as f:
                meta = ReplayMeta.from_dict(json.loads(f.read()))
            return tensor, meta
        elif tensor_present != meta_present:
            logger.warn(
                f"{tensor_path} exists: {tensor_present} " +
                f"meta_present, {meta_path} exists: {meta_present}"
            )
            if not meta_present:
                # XXX: this is not great since its assuming directory replay set
                try:
                    replay_filepath = self._replay_set.replay_path(uuid)
                except Exception as e:
                    logger.warn(f"Error trying to get only player meta {e}")
                    return None
                else:
                    meta = self._backup_get_meta(uuid, replay_filepath)
                    if meta is not None:
                        self._json_dump_meta(meta, meta_path)
                        return self._maybe_load_from_cache(uuid)

    def _save_to_cache(self, replay_id, replay_data, meta):
        tensor_path, meta_path = self._get_tensor_and_meta_path(replay_id)
        with open(tensor_path, 'wb') as f:
            torch.save(replay_data, f)
        self._json_dump_meta(meta, meta_path)
        return replay_data, meta

    def _json_dump_meta(self, meta: ReplayMeta, path):
        with open(path, 'w') as f:
            f.write(json.dumps(meta.to_dict()))

    def get_replay_meta(self, uuid) -> ReplayMeta:
        meta_path = self._cache_path_for_replay_with_extension(uuid, self._meta_extension)
        with open(meta_path, 'rb') as f:
            return ReplayMeta.from_dict(json.loads(f.read()))

    def get_replay_tensor(self, uuid) -> (torch.Tensor, ReplayMeta):
        """Get the replay tensor and player meta associated with the provided uuid."""
        return self._maybe_load_from_cache(uuid) or self._save_to_cache(
            uuid, *self._replay_set.get_replay_tensor(uuid)
        )

    def is_cached(self, uuid) -> bool:
        tensor_path, meta_path = self._get_tensor_and_meta_path(uuid)
        return os.path.exists(tensor_path) and os.path.exists(meta_path)

    def __getattr__(self, name):
        """Defer to self._replay_set."""
        return getattr(self._replay_set, name)


class DirectoryReplaySet(ReplaySet):
    """A replay set that consists of replay files in a potentially nested directory."""

    def __init__(self, filepath, replay_extension="replay", backup_get_meta=manifest_get_meta):
        self._filepath = filepath
        self._replay_extension = replay_extension
        self._replay_id_paths = list(self._get_replay_ids())
        self._replay_path_dict = dict(self._replay_id_paths)
        self._backup_get_meta = backup_get_meta

    def _get_replay_ids(self):
        return util.get_replay_uuids_in_directory(
            self._filepath, replay_extension=self._replay_extension
        )

    def replay_path(self, replay_id):
        """Get the path of the given replay id."""
        return self._replay_path_dict[replay_id]

    def get_replay_uuids(self):
        return [replay_id for replay_id, _ in self._replay_id_paths]

    def get_replay_tensor(self, uuid) -> (torch.Tensor, ReplayMeta):
        """Get the replay tensor and player order associated with the provided uuid."""
        replay_path = self.replay_path(uuid)
        logger.info(f"Loading replay from {replay_path}")
        replay_meta, np_array = boxcars_py.get_replay_meta_and_numpy_ndarray(replay_path)
        return (
            torch.as_tensor(np_array),
            ReplayMeta.from_boxcar_frames_meta(replay_meta),
        )


class LabelValueWasNone(Exception):
    pass


class ReplaySetAssesor:

    class ReplayStatus:
        ready = None

    class PassedStatus(ReplayStatus):
        ready = True

        def __init__(self, player_labels):
            self.player_labels = player_labels

    class FailedStatus(ReplayStatus, Exception):
        ready = False

        def __init__(self, exception):
            self.exception = exception

    class TensorFail(FailedStatus):
        pass

    class MetaFail(FailedStatus):
        pass

    class LabelFail(FailedStatus):

        def __init__(self, player, *args):
            self.player = player
            return super().__init__(*args)

    def __init__(self, replay_set: ReplaySet, label_lookup, scorer=None):
        self._replay_set = replay_set
        self._label_lookup = label_lookup
        self._scorer = scorer

    def get_replay_statuses(self, load_tensor=True):
        return {
            uuid: self._get_replay_status(uuid, load_tensor=load_tensor)
            for uuid in self._replay_set.get_replay_uuids()
        }

    known_errors = [
        "ActorId(-1) not found",
        "Player team unknown",
        "Players found in frames that were not part of",
        "Replay is corrupt",
        "Could not decode replay content data at offset"
    ]

    def _get_player_labels(self, meta):
        player_labels = []
        for player in meta.player_order:
            try:
                player_label_value = self._label_lookup(player, meta.datetime.date())
            except mmr.NoMMRHistory as e:
                raise self.LabelFail(player, e)
            except Exception as e:
                raise e
            else:
                if isinstance(player_label_value, numbers.Number):
                    player_labels.append(player_label_value)
                else:
                    raise self.LabelFail(player, LabelValueWasNone())

        return player_labels

    def _should_reraise(self, e):
        try:
            exception_text = e.args[0]
        except Exception:
            pass
        else:
            for error_text in self.known_errors:
                if error_text in exception_text:
                    return False
        return True

    def _get_replay_status(self, uuid, load_tensor=True):
        logger.info(self._replay_set.replay_path(uuid))
        if (
                isinstance(self._replay_set, CachedReplaySet) and not
                load_tensor and self._replay_set.is_cached(uuid)
        ):
            meta = self._replay_set.get_replay_meta(uuid)
        else:
            try:
                _, meta = self._replay_set.get_replay_tensor(uuid)
            except Exception as e:
                logger.warn(f"Tensor load failure for {uuid}, {e}")
                if self._should_reraise(e):
                    raise e
                else:
                    return self.TensorFail(e)

        try:
            player_labels = self._get_player_labels(meta)
        except self.LabelFail as e:
            logger.warn(f"Label failure for {uuid}, {e}")
            if self._scorer is not None:
                logger.info(f"{uuid}: self._scorer.score_replay_meta(meta)")
            return e

        logger.info(f"Replay {uuid} passed.")

        if self._scorer is not None:
            logger.info(self._scorer.score_replay_meta(meta))
        return self.PassedStatus(player_labels)


def player_cache_label_lookup(
        get_player,
        mmr_function=mmr.SeasonBasedPolyFitMMRCalculator.get_mmr_for_player_at_date,
        *args, **kwargs,
):
    def lookup_label_for_player(player: PlatformPlayer, game_time: datetime.datetime):
        data = get_player(player)

        if data is None:
            raise mmr.NoMMRHistory

        return mmr_function(game_time, data, *args, **kwargs)

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
