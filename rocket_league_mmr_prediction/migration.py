"""Utilities for migrating data between directories and formats."""
import asyncio
import json
import os
import itertools

from . import player_cache as pc, tracker_network


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


async def populate_player_cache_from_directory_using_tracker_network(filepath, count=50):
    """Populate the mmr cache for the provided directory using the tracker network."""
    player_cache = pc.PlayerCache.new_with_cache_directory(filepath)
    player_data_fetch = tracker_network.TrackerNetwork()

    cached_get = pc.cached_get_from_tracker_network(player_cache, player_data_fetch)

    player_metas = list(get_all_players_from_replay_directory(filepath))

    semaphore = asyncio.Semaphore(2)

    for player_meta in player_metas[:count]:
        async with semaphore as _:
            await cached_get(player_meta)


async def copy_games_if_metadata_available_and_conditions_met(
        source_filepath, target_filepath
):
    """Copy games from the source_filepath to the target_filepath under appropriate conditions."""
    player_cache = pc.PlayerCache.new_with_cache_directory(target_filepath)
    player_data_fetch = tracker_network.TrackerNetwork()
    checker = pc.CachedPlayerDataAvailabilityChecker(player_cache, player_data_fetch)

    for game in get_all_games_from_replay_directory(source_filepath):
        player_datas = await checker.get_player_data(game)
        all_were_dict = all(isinstance(result, dict) for result in player_datas)
        game_id = game["id"]
        print(f"{game_id} will be copied over: {all_were_dict}")
