"""Caches for game and player metadata implemented with plyvel."""
import abc
import json
import lmdb
import logging
import plyvel

from . import tracker_network
from ._replay_meta import PlatformPlayer


logger = logging.getLogger(__name__)


def _use_tracker_url_suffix_as_key(player):
    if isinstance(player, PlatformPlayer):
        return player.tracker_suffix.encode('utf-8')

    key_string = tracker_network.get_profile_suffix_for_player(player)
    if key_string is not None:
        return key_string.encode('utf-8')


class DatabaseBackend(abc.ABC):
    @abc.abstractmethod
    def get(self, key):
        pass

    @abc.abstractmethod
    def put(self, key, value):
        pass

    @abc.abstractmethod
    def iterator(self):
        pass


class PlyvelDatabaseBackend(DatabaseBackend):

    def __init__(self, filepath, dbname="player-id-"):
        self._db = plyvel.DB(filepath, create_if_missing=True)
        self._db = self._db.prefixed_db(dbname.encode('utf-8')) if dbname else self._db

    def get(self, key):
        return self._db.get(key)

    def put(self, key, value):
        self._db.put(key, value)

    def iterator(self, start_key=None):
        return self._db.iterator(start=start_key)


class LMDBDatabaseBackend(DatabaseBackend):

    def __init__(self, filepath, dbname="player-id", **kwargs):
        kwargs.setdefault("max_dbs", 10)
        kwargs.setdefault("map_size", 2 * 1024 ** 3)
        self._env = lmdb.open(filepath, **kwargs)
        self._db = self._env.open_db(dbname.encode('utf-8')) if dbname else self._env

    def get(self, key):
        with self._env.begin(db=self._db) as txn:
            return txn.get(key)

    def put(self, key, value):
        with self._env.begin(db=self._db, write=True) as txn:
            txn.put(key, value)

    def iterator(self, start_key=None):
        with self._env.begin(db=self._db) as txn:
            cursor = txn.cursor()
            if start_key is not None:
                cursor.set_key(start_key)
            for v in cursor.iternext():
                yield v


class PlayerCache:
    """Encapsulates the player cache."""

    error_key = "__error__"
    manual_override_key = "__manual_override__"

    @classmethod
    def plyvel(cls, filepath, **kwargs):
        return cls(PlyvelDatabaseBackend(filepath))

    @classmethod
    def lmdb(cls, filepath, **kwargs):
        return cls(LMDBDatabaseBackend(filepath), **kwargs)

    def __init__(self, db_backend, key_fn=_use_tracker_url_suffix_as_key):
        """Initialize the metadata cache from a replay directory."""
        self._db = db_backend
        self._key_fn = key_fn

    def insert_data_for_player(self, player, data):
        key = self._key_for_player(player)
        if key is not None:
            return self._db.put(key, json.dumps(data).encode('utf-8'))

    def _get_data_from_key(self, key) -> dict:
        result = self._db.get(key)
        if result is None:
            return None
        value = json.loads(result.decode('utf-8'))
        return value

    def iterator(self, *args, **kwargs):
        for key, value in self._db.iterator(*args, **kwargs):
            yield self._decode_key(key), self._decode_value(value)

    def __iter__(self):
        for key, value in self._db.iterator():
            yield self._decode_key(key), self._decode_value(value)

    # Methods that are unaffected by db

    def get_player_data_no_err(self, player: PlatformPlayer):
        """Get the data of the provided player catching errors if they occur."""
        try:
            return self.get_player_data(player)
        except PlayerCacheError:
            pass

    def get_player_data(self, player: PlatformPlayer):
        """Get the data of the provided player."""
        key = self._key_for_player(player)
        if key is None:
            return None
        return self._get_data_from_key(key)

    def insert_manual_override(self, player: PlatformPlayer, mmr):
        self.insert_data_for_player(player, {self.manual_override_key: mmr})

    def insert_error_for_player(self, player, error_data):
        """Insert an error for a player."""
        assert "type" in error_data
        return self.insert_data_for_player(player, {self.error_key: error_data})

    def has_error(self, player):
        """Indicate whether or not the player has a stored error."""
        try:
            return self.error_key in self.get_player_data_no_err(player)
        except Exception:
            return False

    def present_and_no_error(self, player):
        """Check whether a player is present and errorless in the db."""
        data_or_error = self.get_player_data(player) or {
            self.error_key: {'type': "Not present"}
        }
        return self.error_key not in data_or_error

    def _key_for_player(self, player) -> bytes:
        return self._key_fn(player)

    def _key(self, player_id, platform) -> bytes:
        return json.dumps({"id": player_id, "platform": platform}).encode('utf-8')

    def _decode_key(self, key_bytes: bytes):
        return key_bytes.decode('utf-8')

    def _decode_value(self, value_bytes: bytes) -> int:
        return json.loads(value_bytes)


class CachedGetPlayerData:
    """A version of the tracker network that is backed by a :py:class:`PlayerCache`."""

    def __init__(self, player_cache, get_player_data, cache_misses=True, retry_errors=("500")):
        """Initatialize the cached get."""
        self._get_player_data = get_player_data
        self._player_cache = player_cache
        self._retry_errors = retry_errors
        self._cache_misses = cache_misses

    def get_player_data(self, player_meta, force_refresh=False):
        """Get player data from cache or get_player_data."""

        if not force_refresh:
            player_data = self._player_cache.get_player_data(player_meta)
            if player_data and self._player_cache.error_key in player_data:
                if player_data[self._player_cache.error_key]['type'] in self._retry_errors:
                    pass
                else:
                    return player_data
            elif player_data:
                return player_data

        try:
            player_data = self._get_player_data(player_meta)
        except tracker_network.Non200Exception as e:
            logger.warn("Could not obtain data for {} due to {}".format(
                player_meta, e
            ))
            if self._cache_misses and e.status_code in (404, 500):
                self._player_cache.insert_error_for_player(
                    player_meta, {"type": str(e.status_code)}
                )
        else:
            meta_dict = (
                player_meta.to_dict()
                if isinstance(player_meta, PlatformPlayer)
                else player_meta
            )
            player_data["player_metadata"] = meta_dict
            self._player_cache.insert_data_for_player(player_meta, player_data)

        return self._player_cache.get_player_data_no_err(player_meta)
