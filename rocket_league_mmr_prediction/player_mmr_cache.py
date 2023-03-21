import os
import itertools
import backoff
import plyvel
import json

from . import tracker_network


def get_all_players_from_replay_directory(filepath):
    """Get all player dictionaries from manifest files contained in the directory at `filepath`."""
    for manifest_filepath in get_manifest_files(filepath):
        for player in get_players_from_manifest_file(manifest_filepath):
            yield player


def get_players_from_manifest_file(filepath):
    """Get all player dictionaries from the manifest file at `filepath`."""
    with open(filepath) as f:
        manifest_data = json.loads(f.read())
    for game in manifest_data.values():
        for player in itertools.chain(game["orange"]["players"], game["blue"]["players"]):
            yield player


def get_manifest_files(filepath, manifest_filename="manifest.json"):
    """Get all manifest files in `filepath`."""
    for root, _, files in os.walk(filepath):
        for filename in files:
            if filename == manifest_filename:
                yield os.path.join(root, filename)


class MMRCache(object):
    """Encapsulates the mmr label cache."""

    def __init__(self, replay_directory_filepath):
        """Initialize the mmr cache from a replay directory."""
        self._filepath = replay_directory_filepath
        self._database_filepath = os.path.join(self._filepath, "label_cache")
        self._db = plyvel.DB(self._database_filepath, create_if_missing=True)

    def get_mmr_for_player_with_platform(self, player: str, platform: str) -> int:
        """Get the mmr of the player associated with the provided name and platform."""
        return self._get_mmr_from_key(self._key(player, platform))

    def get_mmr_for_player(self, player) -> int:
        """Get the mmr of the provided player."""
        return self._get_mmr_from_key(self._key_for_player(player))

    def insert_mmr_for_player(self, player, mmr: int):
        """Insert the provided mmr for the provided player."""
        self._db.put(self._key_for_player(player), round(mmr).to_bytes(64, byteorder='big'))

    def _get_mmr_from_key(self, key) -> int:
        result = self._db.get(key)
        if result is not None:
            return int.from_bytes(result, byteorder='big')

    def _key_for_player(self, player) -> bytes:
        return self._key(player["name"], player["id"]["platform"])

    def _key(self, player, platform) -> bytes:
        return json.dumps({"player": player, "platform": platform}).encode('utf-8')

    def _decode_key(self, key_bytes: bytes):
        return json.loads(key_bytes)

    def _decode_value(self, value_bytes: bytes) -> int:
        return int.from_bytes(value_bytes, byteorder='big')

    def __iter__(self):
        """Iterate over the decoded values in the cache."""
        for key_bytes, value_bytes in self._db.iterator():
            yield self._decode_key(key_bytes), self._decode_value(value_bytes)


@backoff.on_exception(
    backoff.expo, json.decoder.JSONDecodeError, max_tries=8
)
def get_mmr_with_retries(player_meta):
    """Fetch mmr for player meta with an exponential backoff retry."""
    return tracker_network.get_doubles_mmr_from_player_meta(player_meta)


def populate_mmr_cache_from_directory_using_tracker_network(
        filepath,
        get_mmr_from_player_meta=get_mmr_with_retries
):
    """Populate the mmr cache for the provided directory."""
    mmr_cache = MMRCache(filepath)
    for player_meta in get_all_players_from_replay_directory(filepath):
        existing_value = mmr_cache.get_mmr_for_player(player_meta)
        if existing_value is None:
            try:
                mmr = get_mmr_from_player_meta(player_meta)
            except Exception as e:
                print("Could not obtain an mmr value for {} due to {}".format(
                    player_meta["name"], e)
                )
            if mmr is not None:
                mmr_cache.insert_mmr_for_player(player_meta, mmr)
            else:
                print("Could not find a value for {}".format(player_meta["name"]))
        else:
            player_name = player_meta["name"]
            print(f"Value already present for {player_name}")
