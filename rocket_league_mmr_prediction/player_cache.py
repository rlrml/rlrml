"""Caches for game and player metadata implemented with plyvel."""
import json
import logging
import os
import plyvel

from . import tracker_network


logger = logging.getLogger(__name__)


def _use_tracker_url_suffix_as_key(player):
    key_string = tracker_network.get_profile_suffix_for_player(player)
    if key_string is not None:
        return key_string.encode('utf-8')


# Remove this
class PlayerNotFoundOnTrackerNetwork:
    """Sentinel value used to indicate that the player can't be found on the tracker network."""

    string = "PlayerNotFoundOnTrackerNetwork"


class PlayerCacheError(Exception):
    """A base class for player cache errors."""

    pass


class PlayerCacheStoredError(PlayerCacheError):
    """An error stored in the cache for the player."""

    def __init__(self, data):
        """Initialize the error with the data that is stored about the error in the cache."""
        self.data = data


class PlayerCacheMissError(PlayerCacheError):
    """An error that is thrown when the player is not found in the cache."""

    pass


class PlayerCache:
    """Encapsulates the player cache."""

    error_key = "__error__"

    @classmethod
    def new_with_cache_directory(cls, replay_filepath):
        """Create a new MetadataCache in the cache subdirectory of the provided filepath."""
        return cls(os.path.join(replay_filepath, "cache"))

    def __init__(self, filepath, key_fn=_use_tracker_url_suffix_as_key):
        """Initialize the metadata cache from a replay directory."""
        self._filepath = filepath
        self._db = plyvel.DB(self._filepath, create_if_missing=True)
        self._player_name_db = self._db.prefixed_db("player-name-".encode('utf-8'))
        self._player_id_db = self._db.prefixed_db("player-id-".encode('utf-8'))
        self._key_fn = key_fn

    def get_player_data_no_err(self, player):
        """Get the data of the provided player catching errors if they occur."""
        try:
            return self.get_player_data(player)
        except PlayerCacheError:
            pass

    def get_player_data(self, player):
        """Get the data of the provided player."""
        key = self._key_for_player(player)
        if key is None:
            return None
        return self._get_data_from_key(key)

    def insert_data_for_player(self, player, meta: dict):
        """Insert the provided data for given player."""
        key = self._key_for_player(player)
        if key is not None:
            return self._player_id_db.put(
                key, json.dumps(meta).encode('utf-8')
            )

    def insert_error_for_player(self, player, error_data):
        """Insert an error for a player."""
        assert "type" in error_data
        return self.insert_data_for_player(player, {self.error_key: error_data})

    def _get_data_from_key(self, key) -> dict:
        result = self._player_id_db.get(key)
        if result is None:
            raise PlayerCacheMissError()
        value = self._decode_value(result)
        if value == PlayerNotFoundOnTrackerNetwork.string:
            raise PlayerCacheStoredError({"type": "404", "__oldform__": True})
        if self.error_key in value:
            raise PlayerCacheStoredError(value[self.error_key])
        return value

    def _key_for_player(self, player) -> bytes:
        return self._key_fn(player)

    def _key(self, player_id, platform) -> bytes:
        return json.dumps({"id": player_id, "platform": platform}).encode('utf-8')

    def _decode_key(self, key_bytes: bytes):
        return key_bytes.decode('utf-8')

    def _decode_value(self, value_bytes: bytes) -> int:
        return json.loads(value_bytes)

    def iterator(self, *args, **kwargs):
        """Construct an iterator using the underlying db."""
        for key_bytes, value_bytes in self._player_id_db.iterator(*args, **kwargs):
            yield self._decode_key(key_bytes), self._decode_value(value_bytes)

    def __iter__(self):
        """Iterate over the decoded values in the cache."""
        for key_bytes, value_bytes in self._player_id_db.iterator():
            yield self._decode_key(key_bytes), self._decode_value(value_bytes)


class CachedGetPlayerData:
    """A version of the tracker network that is backed by a :py:class:`PlayerCache`."""

    def __init__(self, player_cache, get_player_data, cache_misses=True, retry_errors=("500")):
        """Initatialize the cached get."""
        self._get_player_data = get_player_data
        self._player_cache = player_cache
        self._retry_errors = retry_errors
        self._cache_misses = cache_misses

    def get_player_data(self, player_meta):
        """Get player data from cache or get_player_data."""
        try:
            return self._player_cache.get_player_data(player_meta)
        except PlayerCacheMissError:
            pass
        except PlayerCacheStoredError as e:
            if "__oldform__" in e.data:
                pass
            elif e.data["type"] in self._retry_errors:
                pass
            else:
                return None

        try:
            player_data = self._get_player_data(player_meta)
        except tracker_network.Non200Exception as e:
            logger.warn("Could not obtain an mmr value for {} due to {}".format(
                player_meta, e
            ))
            if self._cache_misses and e.status_code in (404, 500):
                self._player_cache.insert_error_for_player(
                    player_meta, {"type": str(e.status_code)}
                )
        else:
            player_data["player_metadata"] = player_meta
            self._player_cache.insert_data_for_player(player_meta, player_data)

        return self._player_cache.get_player_data_no_err(player_meta)
