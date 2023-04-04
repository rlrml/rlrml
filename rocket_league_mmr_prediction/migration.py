"""Utilities for migrating data between directories and formats."""
import asyncio
import json
import logging
import os
import itertools

from . import player_cache as pc, tracker_network
from . import mmr
from . import util


logger = logging.getLogger(__name__)


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


def copy_games_if_metadata_available_and_conditions_met(
        source_filepath, target_filepath
):
    """Copy games from the source_filepath to the target_filepath under appropriate conditions."""

    get_player_data = util.vpn_cycled_cached_player_get(target_filepath)

    included = 0
    excluded = 0
    missing_features = {}

    for game in get_all_games_from_replay_directory(source_filepath):
        game_id = game["id"]
        reason = None
        all_players = itertools.chain(game["orange"]["players"], game["blue"]["players"])
        for player_meta in all_players:
            player_data = get_player_data(player_meta)

            if not isinstance(player_data, dict):
                suffix = tracker_network.get_profile_suffix_for_player(player_meta)
                missing_features[suffix] = missing_features.setdefault(suffix, 0) + 1
                # if missing_features[suffix] > 1:
                #     logger.warn(f"Retrying {player_meta}")
                #     await checker._get_player_data(player_meta, retry_tombstones=True)
                reason = f"Missing data for player {player_meta}"
                break

            try:
                mmr.get_player_mmr_for_game(
                    player_data, game, days_after=0, days_before=10
                )
            except mmr.MMRFilteringError as e:
                reason = f"MMR filtering {e} {player_meta}"

        if reason is None:
            included += 1
        else:
            excluded += 1

        logger.debug(f"{game_id} will be copied over: {reason is None}, reason: {reason}")
        logger.debug(f"included: {included}, excluded: {excluded}")


class PlayerDataConcurrencyLimiter:
    """Check that we have player mmr data for all players in the game."""

    def __init__(self, get_player_data, max_concurrency=1):
        """Initialize the availability checker."""
        self._get_player_data = get_player_data
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def get_player_data_for_game(self, game, **kwargs):
        """Get player data for each player in the provided game."""
        all_players = itertools.chain(game["orange"]["players"], game["blue"]["players"])

        return await asyncio.gather(
            *[self._get_one_player_data(player) for player in all_players]
        )

    async def _get_one_player_data(self, player, **kwargs):
        async with self._semaphore as _:
            return (player, await self._get_player_data(player, **kwargs))
