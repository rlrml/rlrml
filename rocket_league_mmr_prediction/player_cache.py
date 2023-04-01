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


class PlayerNotFoundOnTrackerNetwork:
    """Sentinel value used to indicate that the player can't be found on the tracker network."""

    string = "PlayerNotFoundOnTrackerNetwork"


class PlayerCache:
    """Encapsulates the player cache."""

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

    def get_player_data(self, player):
        """Get the mmr of the provided player."""
        key = self._key_for_player(player)
        if key is None:
            return None
        return self._get_info_from_key(key)

    def insert_info_for_player(self, player, meta: dict):
        """Insert the provided data for given player."""
        key = self._key_for_player(player)
        if key is not None:
            return self._player_id_db.put(
                key, json.dumps(meta).encode('utf-8')
            )

    def _get_info_from_key(self, key) -> dict:
        result = self._player_id_db.get(key)
        if result is not None:
            return self._decode_value(result)

    def _key_for_player(self, player) -> bytes:
        return self._key_fn(player)

    def _key(self, player_id, platform) -> bytes:
        return json.dumps({"id": player_id, "platform": platform}).encode('utf-8')

    def _decode_key(self, key_bytes: bytes):
        return key_bytes.decode('utf-8')

    def _decode_value(self, value_bytes: bytes) -> int:
        value = json.loads(value_bytes)
        if value == PlayerNotFoundOnTrackerNetwork.string:
            return PlayerNotFoundOnTrackerNetwork
        return value

    def __iter__(self):
        """Iterate over the decoded values in the cache."""
        for key_bytes, value_bytes in self._player_id_db.iterator():
            yield self._decode_key(key_bytes), self._decode_value(value_bytes)


class CachedGetPlayerData:
    """A version of the tracker network that is backed by a :py:class:`PlayerCache`."""

    def __init__(self, player_cache, get_player_data, cache_misses=True, retry_tombstones=False):
        """Initatialize the cached get."""
        self._get_player_data = get_player_data
        self._player_cache = player_cache
        self._retry_tombstones = retry_tombstones
        self._cache_misses = cache_misses

    async def get_player_data(self, player_meta):
        """Get player data from cache or get_player_data."""
        existing_value = self._player_cache.get_player_data(player_meta)

        if existing_value is not None and (
                not self._retry_tombstones or existing_value != PlayerNotFoundOnTrackerNetwork
        ):
            return existing_value

        try:
            player_info = await self._get_player_data(player_meta)
        except tracker_network.Non200Exception as e:
            logger.warn("Could not obtain an mmr value for {} due to {}".format(
                player_meta, e
            ))
            if self._cache_misses and e.status_code == 404:
                self._player_cache.insert_info_for_player(
                    player_meta, PlayerNotFoundOnTrackerNetwork.string
                )
        except Exception as e:
            logger.warn("Could not obtain an mmr value for {} due to {}, error_type: {}".format(
                player_meta, e, type(e)
            ))
        else:
            if existing_value == PlayerNotFoundOnTrackerNetwork:
                logger.warn(f"Found value for {player_meta} that was not found")
            player_info["player_metadata"] = player_meta
            self._player_cache.insert_info_for_player(player_meta, player_info)

        return self._player_cache.get_player_data(player_meta)
