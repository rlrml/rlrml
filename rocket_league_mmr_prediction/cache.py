import json
import os
import plyvel


class MetadataCache(object):
    """Encapsulates the mmr label cache."""

    @classmethod
    def new_with_cache_directory(cls, replay_filepath):
        """Create a new MetadataCache in the cache subdirectory of the provided filepath."""
        return cls(os.path.join(replay_filepath, "cache"))

    def __init__(self, filepath):
        """Initialize the metadata cache from a replay directory."""
        self._filepath = filepath
        self._db = plyvel.DB(self._filepath, create_if_missing=True)
        self._player_name_db = self._db.prefixed_db("player-name-".encode('utf-8'))
        self._player_id_db = self._db.prefixed_db("player-id-".encode('utf-8'))

    def get_meta_for_player(self, player) -> int:
        """Get the mmr of the provided player."""
        self._get_meta_from_key(self._key_for_player(player))

    def insert_meta_for_player(self, player, meta: dict):
        """Insert the provided mmr for the provided player."""
        return self._player_id_db.put(self._key_for_player(player), meta)

    def _get_meta_from_key(self, key) -> dict:
        result = self._player_id_db.get(key)
        if result is not None:
            return json.loads(result)

    def _key_for_player(self, player) -> bytes:
        id_data = player["id"]
        return self._key(id_data["id"], id_data["platform"])

    def _key(self, player_id, platform) -> bytes:
        return json.dumps({"id": player_id, "platform": platform}).encode('utf-8')

    def _decode_key(self, key_bytes: bytes):
        return json.loads(key_bytes)

    def _decode_value(self, value_bytes: bytes) -> int:
        return json.loads(value_bytes)

    def __iter__(self):
        """Iterate over the decoded values in the cache."""
        for key_bytes, value_bytes in self._player_id_db.iterator():
            yield self._decode_key(key_bytes), self._decode_value(value_bytes)
