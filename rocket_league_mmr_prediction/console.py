"""Defines command line entrypoints to the this library."""
import argparse
import asyncio
import coloredlogs
import functools
import logging
import os
import sys
import xdg_base_dirs

from pathlib import Path

from . import tracker_network
from . import player_cache as pc
from . import load
from . import migration
from . import logger
from . import mmr

def load_rlrml_config(config_path=None):
    config_path = config_path or os.path.join(rlrml_config_directory(), "config.toml")
    try:
        import tomllib
        with open(config_path, 'r') as f:
            return tomllib.loads(f.read())['rlrml']
    except Exception:
        return {}


def rlrml_config_directory():
    return os.path.join(xdg_base_dirs.xdg_config_home(), "rlrml")


def rlrml_data_directory():
    return os.path.join(xdg_base_dirs.xdg_data_dirs()[0], "rlrml")


def _add_rlrml_args(parser=None):
    parser = parser or argparse.ArgumentParser()
    config = load_rlrml_config()
    rlrml_directory = rlrml_data_directory()
    defaults = {
        "player-cache": os.path.join(rlrml_directory, "player_cache"),
        "tensor-cache": os.path.join(rlrml_directory, "tensor_cache"),
    }
    defaults.update(**config)

    parser.add_argument(
        '--player-cache',
        help="The directory where the player cache can be found.",
        type=Path,
        default=defaults['player-cache']
    )
    parser.add_argument(
        '--replay-path',
        help="The directory where game files are stored.",
        type=Path,
        default=os.path.join(rlrml_directory, "replays")
    )
    parser.add_argument(
        '--tensor-cache',
        help="The directory where the tensor cache is held",
        type=Path,
        default=defaults['player-cache']
    )
    parser.add_argument(
        '--ballchasing-token', help="A ballchasing.com authorization token.", type=str,
        default=defaults.get('ballchasing-token')
    )
    return parser


def _call_with_sys_argv(function):
    @functools.wraps(function)
    def call_with_sys_argv():
        coloredlogs.install(level='INFO', logger=logger)
        logger.setLevel(logging.INFO)
        function(*sys.argv[1:])
    return call_with_sys_argv


def load_game_dataset():
    """Convert the game provided through sys.argv."""
    from . import filter
    coloredlogs.install(level='INFO', logger=logger)
    logger.setLevel(logging.INFO)
    parser = _add_rlrml_args()
    args = parser.parse_args()
    print(args)
    player_cache = pc.PlayerCache(str(args.player_cache))
    cached_player_get = pc.CachedGetPlayerData(
        player_cache, tracker_network.get_player_data_with_429_retry()
    ).get_player_data
    scorer = filter.MMREstimateQualityFilter(cached_player_get)
    assesor = load.ReplaySetAssesor(
        load.DirectoryReplaySet.cached(args.tensor_cache, args.replay_path),
        load.player_cache_label_lookup(cached_player_get),
        scorer=scorer
    )
    result = assesor.get_replay_statuses(load_tensor=False)
    import ipdb; ipdb.set_trace()


@_call_with_sys_argv
def convert_game(filepath):
    _add_rlrml_args()
    import boxcars_py
    try:
        meta, tensor = boxcars_py.get_replay_meta_and_numpy_ndarray(filepath)
    except (Exception) as e:
        print(e)
        import ipdb; ipdb.set_trace()
        pass
    import torch
    logger.info("Making tensor")
    tensor = torch.as_tensor(tensor)
    logger.info("done making tensor")
    with open("./saved_tensor.pt", 'wb') as f:
        torch.save(tensor, f)

    with open("./saved_tensor.pt", 'rb') as f:
        tensor = torch.load(f)
    from . import _replay_meta
    print(_replay_meta.ReplayMeta.from_boxcar_frames_meta(meta))


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



def ballchasing_lookup():
    import requests
    import json
    from . import manifest
    parser = _add_rlrml_args()
    parser.add_argument('uuid')
    args = parser.parse_args()
    game_data = requests.get(
        f"https://ballchasing.com/api/replays/{args.uuid}",
        headers={'Authorization': args.ballchasing_token},
    ).json()
    mmr_data = manifest.get_mmr_data_from_manifest_game(game_data)
    print(json.dumps(mmr_data))


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
    season_dates = mmr.tighten_season_dates(mmr.SEASON_DATES, move_end_date=1)
    # print(season_dates)
    segmented_history = mmr.split_mmr_history_into_seasons(
        player['mmr_history']['Ranked Doubles 2v2'],
        season_dates=season_dates
    )

    print(json.dumps(
        mmr.calculate_all_season_statistics(segmented_history, keep_poly=False)
    ))


def setup_system_bus():
    import sdbus
    sdbus.set_default_bus(sdbus.sd_bus_open_system())


def proxy():
    setup_system_bus()
    from .network import proxy
    proxy.app.run(port=5002)
