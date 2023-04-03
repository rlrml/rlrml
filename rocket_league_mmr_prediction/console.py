"""Defines command line entrypoints to the this library."""
import asyncio
import sys
import functools
import logging
import coloredlogs

from sdbus_block import networkmanager as nm

from . import load
from . import migration
from . import logger
from . import filters


def _call_with_sys_argv(function):
    @functools.wraps(function)
    def call_with_sys_argv():
        coloredlogs.install(level='INFO', logger=logger)
        logger.setLevel(logging.INFO)
        function(*sys.argv[1:])
    return call_with_sys_argv


@_call_with_sys_argv
def load_game_dataset(filepath):
    """Convert the game provided through sys.argv."""
    dataset = load.ReplayDataset(filepath, eager_labels=False)
    for i in range(len(dataset)):
        print(i)
        try:
            dataset[i]
        except Exception as e:
            print(e)


@_call_with_sys_argv
def convert_game(filepath):
    game = load.get_carball_game(filepath)
    import ipdb; ipdb.set_trace()
    converter = load._CarballToTensorConverter(game)


@_call_with_sys_argv
def load_game_at_indices(filepath, *indices):
    """Convert the game provided through sys.argv."""
    dataset = load.ReplayDataset(filepath, eager_labels=False)
    for index in indices:
        dataset[int(index)]


@_call_with_sys_argv
def fill_cache_with_tracker_rank(filepath):
    """Fill a player info cache in a directory of replays."""
    loop = asyncio.get_event_loop()
    task = migration.populate_player_cache_from_directory_using_tracker_network(filepath)
    loop.run_until_complete(task)


@_call_with_sys_argv
def _iter_cache(filepath):
    from . import util
    from . import player_cache as cache
    from . import tracker_network as tn
    import sdbus
    sdbus.set_default_bus(sdbus.sd_bus_open_system())

    old_form = []
    missing_data = 0
    present_data = 0
    player_cache = cache.PlayerCache.new_with_cache_directory(filepath)
    player_get = util.vpn_cycled_cached_player_get(filepath, player_cache=player_cache)
    for player_key, player_data in player_cache:
        if "__error__" in player_data:
            if player_data["__error__"]["type"] == "500":
                player_get({"__tracker_suffix__": player_key})
            missing_data += 1
        elif cache.PlayerNotFoundOnTrackerNetwork.string == player_data:
            missing_data += 1
            old_form.append(player_key)
        else:
            if "platform" not in player_data and "mmr" in player_data:
                print(f"Fixing {player_key}")
                combined = tn.combine_profile_and_mmr_json(player_data)
                player_cache.insert_data_for_player(
                    {"__tracker_suffix__": player_key}, combined
                )
            present_data += 1

    del player_cache

    print(f"present_data: {present_data}, missing_data: {missing_data}")

    if len(old_form):
        logger.warn(f"Non-empty old formm {old_form}")

    for player_suffix in old_form:
        player_get({"__tracker_suffix__": player_suffix})


@_call_with_sys_argv
def _copy_games(source, dest):
    import sdbus
    sdbus.set_default_bus(sdbus.sd_bus_open_system())
    migration.copy_games_if_metadata_available_and_conditions_met(source, dest)


@_call_with_sys_argv
def host_plots(filepath):
    """Run an http server that hosts plots of player mmr that in the cache."""
    from . import _http_graph_server
    _http_graph_server.make_routes(filepath)
    _http_graph_server.app.run(port=5001)


@_call_with_sys_argv
def get_player(filepath, player_key):
    """Get the provided player either from the cache or the tracker network."""
    import json
    import sdbus
    from . import util

    sdbus.set_default_bus(sdbus.sd_bus_open_system())
    player_get = util.vpn_cycled_cached_player_get(filepath)
    player = player_get({"__tracker_suffix__": player_key})
    # print(player["platform"])
    # print(len(player['mmr_history']['Ranked Doubles 2v2']))
    season_dates = filters.tighten_season_dates(filters.SEASON_DATES, move_end_date=1)
    # print(season_dates)
    segmented_history = filters.split_mmr_history_into_seasons(
        player['mmr_history']['Ranked Doubles 2v2'],
        season_dates=season_dates
    )

    print(json.dumps(
        filters.calculate_all_season_statistics(segmented_history, keep_poly=False)
    ))
