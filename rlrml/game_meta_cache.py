import plyvel
import json


class GameMetaDataCache:
    """Encapsulates the game meta data cache."""

    def __init__(self, filepath):
        """Initialize the cache."""
        self._filepath = filepath
        self._db = plyvel.DB(self._filepath)

    def get_meta_by_id(self, game_id):
        """Return the metadata associated with the provided id."""
        if isinstance(game_id, str):
            game_id = game_id.encode('utf-8')
        return json.loads(self._db.get(game_id).decode('utf-8'))

    def insert_game(self, game_data):
        """Insert the provided game."""
        key = game_data["id"].encode('utf-8')
        data_bytes = json.dumps(game_data).encode('utf-8')
        return self._db.put(key, data_bytes)
