"""Defines command line entrypoints to the this library."""
import argparse
import backoff
import coloredlogs
import functools
import logging
import os
import sys
import json
import xdg_base_dirs

from pathlib import Path

from . import tracker_network
from . import player_cache as pc
from . import load
from . import manifest
from . import _http_graph_server
from . import logger
from . import vpn
from . import util
from . import filter


def _load_rlrml_config(config_path=None):
    config_path = config_path or os.path.join(_rlrml_config_directory(), "config.toml")
    try:
        import tomllib
        with open(config_path, 'r') as f:
            return tomllib.loads(f.read())['rlrml']
    except Exception:
        return {}


def _rlrml_config_directory():
    return os.path.join(xdg_base_dirs.xdg_config_home(), "rlrml")


def _rlrml_data_directory():
    return os.path.join(xdg_base_dirs.xdg_data_dirs()[0], "rlrml")


def _add_rlrml_args(parser=None):
    parser = parser or argparse.ArgumentParser()
    config = _load_rlrml_config()
    rlrml_directory = _rlrml_data_directory()
    defaults = {
        "player-cache": os.path.join(rlrml_directory, "player_cache"),
        "tensor-cache": os.path.join(rlrml_directory, "tensor_cache"),
        "replay-path": os.path.join(rlrml_directory, "replay_path")
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
        default=defaults.get('replay-path')
    )
    parser.add_argument(
        '--tensor-cache',
        help="The directory where the tensor cache is held",
        type=Path,
        default=defaults.get('tensor-cache')
    )
    parser.add_argument(
        '--playlist',
        help="The name of the playlist that is being used.",
        type=str,               # TODO: Change this to an enum
        default='Ranked Doubles 2v2'
    )
    parser.add_argument(
        '--cycle-vpn',
        help="Enable vpn cycling.",
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--ballchasing-token', help="A ballchasing.com authorization token.", type=str,
        default=defaults.get('ballchasing-token')
    )
    return parser


def _setup_system_bus():
    import sdbus
    sdbus.set_default_bus(sdbus.sd_bus_open_system())


class _RLRMLBuilder:

    @classmethod
    def with_default(cls, fn):
        @functools.wraps(fn)
        def wrapped():
            builder = cls.default()
            return fn(builder)
        return wrapped

    @classmethod
    def default(cls):
        coloredlogs.install(level='INFO', logger=logger)
        logger.setLevel(logging.INFO)
        return cls(_add_rlrml_args().parse_args())

    def __init__(self, args):
        self._args = args

    @functools.cached_property
    def vpn_cycle_status_codes(self):
        return (429, 403)

    @functools.cached_property
    def player_cache(self):
        return pc.PlayerCache(str(self._args.player_cache))

    @functools.cached_property
    def cached_get_player_data(self):
        return pc.CachedGetPlayerData(
            self.player_cache, self.network_get_player_data
        ).get_player_data

    @functools.cached_property
    def tracker_network_cloud_scraper(self):
        return tracker_network.CloudScraperTrackerNetwork()

    @functools.cached_property
    def bare_get_player_data(self):
        return self.tracker_network_cloud_scraper.get_player_data

    @functools.cached_property
    def network_get_player_data(self):
        if self._args.cycle_vpn:
            return self.vpn_cycled_get_player_data
        else:
            return tracker_network.get_player_data_with_429_retry()

    @functools.cached_property
    def vpn_cycler(self):
        _setup_system_bus()
        return vpn.VPNCycler()

    @functools.cached_property
    def vpn_cycled_get_player_data(self):
        return self.vpn_cycler.cycle_vpn_backoff(
            backoff.runtime,
            tracker_network.Non200Exception,
            giveup=lambda e: e.status_code not in self.vpn_cycle_status_codes,
            on_backoff=lambda d: self.tracker_network_cloud_scraper.refresh_scraper(),
            value=util._constant_retry(8)
        )(self.tracker_network_cloud_scraper.get_player_data)

    @functools.cached_property
    def player_mmr_estimate_scorer(self):
        return filter.MMREstimateQualityFilter(
            self.cached_get_player_data
        )

    @functools.cached_property
    def cached_directory_replay_set(self):
        return load.DirectoryReplaySet.cached(
            self._args.tensor_cache, self._args.replay_path
        )

    @functools.cached_property
    def lookup_label(self):
        # Perhaps allow settings that control whether any network is allowed?
        return load.player_cache_label_lookup(self.cached_get_player_data)

    @functools.cached_property
    def torch_dataset(self):
        return load.ReplayDataset(self.cached_directory_replay_set, self.lookup_label)

    def decorate(self, fn):
        @functools.wraps(fn)
        def wrapped():
            return fn(self)
        return wrapped


def _call_with_sys_argv(function):
    @functools.wraps(function)
    def call_with_sys_argv():
        coloredlogs.install(level='INFO', logger=logger)
        logger.setLevel(logging.INFO)
        function(*sys.argv[1:])
    return call_with_sys_argv


@_RLRMLBuilder.with_default
def load_game_dataset(builder):
    """Convert the game provided through sys.argv."""
    assesor = load.ReplaySetAssesor(
        builder.cached_directory_replay_set,
        load.player_cache_label_lookup(
            builder.cached_get_player_data
        ),
        scorer=builder.player_mmr_estimate_scorer
    )
    result = assesor.get_replay_statuses(load_tensor=False)
    result = result
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


@_RLRMLBuilder.with_default
def host_plots(builder):
    """Run an http server that hosts plots of player mmr that in the cache."""
    _http_graph_server.make_routes(builder.player_cache)
    _http_graph_server.app.run(port=5001)


def proxy():
    _setup_system_bus()
    from .network import proxy
    proxy.app.run(port=5002)


@_RLRMLBuilder.with_default
def get_cache_answer_uuids(builder):
    builder = _RLRMLBuilder.default()
    uuids = list(util.get_cache_answer_uuids_in_directory(
        builder._args.replay_path, builder.player_cache
    ))
    print(len(uuids))
    import ipdb; ipdb.set_trace()


def ballchasing_lookup():
    import requests
    parser = _add_rlrml_args()
    parser.add_argument('uuid')
    args = parser.parse_args()
    game_data = requests.get(
        f"https://ballchasing.com/api/replays/{args.uuid}",
        headers={'Authorization': args.ballchasing_token},
    ).json()
    mmr_data = manifest.get_mmr_data_from_manifest_game(game_data)
    print(json.dumps(mmr_data))


def get_player():
    """Get the provided player either from the cache or the tracker network."""
    parser = _add_rlrml_args()
    parser.add_argument('player_key')
    args = parser.parse_args()
    builder = _RLRMLBuilder(args)

    data = builder.player_cache.get_player_data(
        {"__tracker_suffix__": args.player_key}
    )
    print(json.dumps(data))
