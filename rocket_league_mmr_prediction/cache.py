"""Caches for game and player metadata implemented with plyvel."""
import asyncio
import backoff
import itertools
import json
import os
import plyvel

from . import tracker_network


def _use_tracker_url_suffix_as_key(player):
    return tracker_network.get_profile_suffix_for_player(player).encode('utf-8')


class PlayerNotFoundOnTrackerNetwork:
    """Sentinel value used to indicate that the player can't be found on the tracker network."""

    string = "PlayerNotFoundOnTrackerNetwork"


class PlayerCache(object):
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

    def get_info_for_player(self, player) -> int:
        """Get the mmr of the provided player."""
        return self._get_info_from_key(self._key_for_player(player))

    def insert_info_for_player(self, player, meta: dict):
        """Insert the provided data for given player."""
        return self._player_id_db.put(
            self._key_for_player(player), json.dumps(meta).encode('utf-8')
        )

    def _get_info_from_key(self, key) -> dict:
        result = self._player_id_db.get(key)
        if result is not None:
            value = json.loads(result)
            if value == PlayerNotFoundOnTrackerNetwork.string:
                return PlayerNotFoundOnTrackerNetwork
            return value

    def _key_for_player(self, player) -> bytes:
        return self._key_fn(player)

    def _key(self, player_id, platform) -> bytes:
        return json.dumps({"id": player_id, "platform": platform}).encode('utf-8')

    def _decode_key(self, key_bytes: bytes):
        return key_bytes.decode('utf-8')

    def _decode_value(self, value_bytes: bytes) -> int:
        return json.loads(value_bytes)

    def __iter__(self):
        """Iterate over the decoded values in the cache."""
        for key_bytes, value_bytes in self._player_id_db.iterator():
            yield self._decode_key(key_bytes), self._decode_value(value_bytes)


def get_all_games_from_replay_directory(filepath):
    """Get all game dictionaries from manifest files contained in the directory at `filepath`."""
    for manifest_filepath in get_manifest_files(filepath):
        with open(manifest_filepath) as f:
            manifest_data = json.loads(f.read())
        for game in manifest_data.values():
            yield game


def get_all_players_from_replay_directory(filepath):
    """Get all player dictionaries from manifest files contained in the directory at `filepath`."""
    for game in get_all_games_from_replay_directory(filepath):
        for player in itertools.chain(game["orange"]["players"], game["blue"]["players"]):
            yield player


def get_manifest_files(filepath, manifest_filename="manifest.json"):
    """Get all manifest files in `filepath`."""
    for root, _, files in os.walk(filepath):
        for filename in files:
            if filename == manifest_filename:
                yield os.path.join(root, filename)


def _log_backoff(details):
    details = details["exception"]
    print(f"Backing off {details}")


def _use_retry_after(exception: tracker_network.Non200Exception):
    string_value = exception.response_headers.get('retry-after') or \
        exception.response_headers.get('Retry-After')
    if string_value is None:
        return 60
    return int(string_value)


def cached_get_from_tracker_network(
        player_cache: PlayerCache, tracker_network_api: tracker_network.TrackerNetwork,
        cache_misses=True,
):
    """Return a function that provides a cached fetch for player info from the tracker network."""
    # XXX: choose better exception here
    fetch_with_backoff = backoff.on_exception(
        backoff.runtime, tracker_network.Non200Exception, max_time=300,
        giveup=lambda e: e.status_code != 429,
        on_backoff=_log_backoff,
        value=_use_retry_after,
        jitter=None,
    )(tracker_network_api.get_info_for_player)

    async def cached_fetch(player_meta):
        existing_value = player_cache.get_info_for_player(player_meta)

        if existing_value is not None:
            return existing_value

        try:
            player_info = tracker_network.combine_profile_and_mmr_json(
                await fetch_with_backoff(player_meta)
            )
        except tracker_network.Non200Exception as e:
            print("Could not obtain an mmr value for {} due to {}".format(
                player_meta, e
            ))
            if cache_misses and e.status_code != 429:
                player_cache.insert_info_for_player(
                    player_meta, PlayerNotFoundOnTrackerNetwork.string
                )
        except Exception as e:
            print("Could not obtain an mmr value for {} due to {}".format(
                player_meta, e
            ))
        else:
            player_info["player_metadata"] = player_meta
            player_cache.insert_info_for_player(player_meta, player_info)

        return player_cache.get_info_for_player(player_meta)

    return cached_fetch


async def populate_player_cache_from_directory_using_tracker_network(filepath, count=50):
    """Populate the mmr cache for the provided directory using the tracker network."""
    player_cache = PlayerCache.new_with_cache_directory(filepath)
    player_data_fetch = tracker_network.TrackerNetwork()

    cached_get = cached_get_from_tracker_network(player_cache, player_data_fetch)

    player_metas = list(get_all_players_from_replay_directory(filepath))

    semaphore = asyncio.Semaphore(2)

    for player_meta in player_metas[:count]:
        async with semaphore as _:
            await cached_get(player_meta)


async def copy_games_if_metadata_available_and_conditions_met(
        source_filepath, target_filepath
):
    """Copy games from the source_filepath to the target_filepath under appropriate conditions."""
    player_cache = PlayerCache.new_with_cache_directory(target_filepath)
    player_data_fetch = tracker_network.TrackerNetwork()
    checker = CachedPlayerDataAvailabilityChecker(player_cache, player_data_fetch)

    for game in get_all_games_from_replay_directory(source_filepath):
        player_datas = await checker.get_player_data(game)
        all_were_dict = all(isinstance(result, dict) for result in player_datas)
        game_id = game["id"]
        print(f"{game_id} will be copied over: {all_were_dict}")


class CachedPlayerDataAvailabilityChecker:
    """Check that we have player mmr data for all players in the game."""

    def __init__(
            self,
            player_cache: PlayerCache,
            tracker_network_api: tracker_network.TrackerNetwork,
            max_concurrency=1
    ):
        """Initialize the availability checker."""
        self._player_cache = player_cache
        self._tracker_network = tracker_network_api
        self._cached_get = cached_get_from_tracker_network(
            self._player_cache, self._tracker_network
        )
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def check_for_player_data(self, game) -> bool:
        """Check that we have player mmr data for all players in the game."""
        results = await self.get_player_data(game)
        return all(isinstance(result, dict) for result in results)

    async def get_player_data(self, game):
        """Get player data for each player in the provided game."""
        all_players = itertools.chain(game["orange"]["players"], game["blue"]["players"])

        return await asyncio.gather(
            *[self._get_player_data(player) for player in all_players]
        )

    async def _get_player_data(self, player):
        async with self._semaphore as _:
            return await self._cached_get(player)
