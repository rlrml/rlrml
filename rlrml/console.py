"""Defines command line entrypoints to the this library."""
import argparse
import boxcars_py
import backoff
import coloredlogs
import datetime
import functools
import logging
import os
import requests
import torch
import json
import xdg_base_dirs

from pathlib import Path

from . import _http_graph_server
from . import score
from . import load
from . import logger
from . import manifest
from . import player_cache as pc
from . import tracker_network
from . import util
from . import vpn
from . import _replay_meta
from .assess import ReplaySetAssesor
from .model import train, build
from .playlist import Playlist


def _load_rlrml_config(config_path=None):
    config_path = config_path or os.path.join(_rlrml_config_directory(), "config.toml")
    try:
        import tomllib
        with open(config_path, 'r') as f:
            return tomllib.loads(f.read())['rlrml']
    except Exception as e:
        logger.warn(f"Hit exception trying to load rlrml config: {e}")
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
        "replay-path": os.path.join(rlrml_directory, "replay_path"),
        "playlist": Playlist("Ranked Doubles 2v2"),
        "boxcar-frames-arguments": {
            "fps": 10,
        }
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
        '--preload',
        help="Whether or not to preload the dataset",
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--tensor-cache',
        help="The directory where the tensor cache is held",
        type=Path,
        default=defaults.get('tensor-cache')
    )
    parser.add_argument(
        '--playlist',
        help="The name (or number) of the playlist that is being used.",
        default='Ranked Doubles 2v2'
    )
    parser.add_argument(
        '--cycle-vpn',
        help="Enable vpn cycling.",
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--model-path',
        help="The path from which to load a model",
        default=defaults.get('model-path')
    )
    parser.add_argument(
        '--add-proxy',
        help="Add a socks proxy uri.",
        action='append',
        dest="socks_proxy_urls",
        default=defaults.get('socks-proxy-urls', [])
    )
    parser.add_argument(
        '--lstm-width',
        type=int,
        default=512
    )
    parser.add_argument('--bcf-args', default=defaults.get("boxcar-frames-arguments"))
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
    def add_args(cls, *args):
        def decorate(fn):
            @functools.wraps(fn)
            def wrapped():
                parser = _add_rlrml_args()
                for arg in args:
                    parser.add_argument(arg)
                parsed_args = parser.parse_args()
                builder = cls(parsed_args)
                cls._setup_default_logging()
                return fn(builder)
            return wrapped
        return decorate

    @classmethod
    def with_default(cls, fn):
        @functools.wraps(fn)
        def wrapped():
            builder = cls.default()
            return fn(builder)
        return wrapped

    @classmethod
    def _setup_default_logging(cls):
        coloredlogs.install(level='INFO', logger=logger)
        logger.setLevel(logging.INFO)

    @classmethod
    def default(cls):
        cls._setup_default_logging()
        return cls(_add_rlrml_args().parse_args())

    def __init__(self, args):
        self._args = args
        logger.info(f"Runnign with {self._args}")

    @functools.cached_property
    def label_scaler(self):
        return util.HorribleHackScaler

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
        return tracker_network.CloudScraperTrackerNetwork(proxy_uris=self._args.socks_proxy_urls)

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
    def playlist(self):
        return Playlist.from_string_or_number(self._args.playlist)

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
        return score.MMREstimateScorer(
            self.cached_get_player_data
        )

    @functools.cached_property
    def cached_directory_replay_set(self):
        return load.DirectoryReplaySet.cached(
            self._args.tensor_cache, self._args.replay_path,
            boxcar_frames_arguments=self._args.bcf_args,
        )

    @functools.cached_property
    def lookup_label(self):
        def get_player_label(player, date):
            if isinstance(date, datetime.datetime):
                date = date.date()
            return self.player_mmr_estimate_scorer.score_player_mmr_estimate(
                player, date, playlist=self.playlist
            )[0]
        return get_player_label

    @functools.cached_property
    def torch_dataset(self):
        return load.ReplayDataset(
            self.cached_directory_replay_set, self.lookup_label,
            self.playlist, self.header_info, preload=self._args.preload,
            label_scaler=self.label_scaler
        )

    @functools.cached_property
    def header_info(self):
        headers_args = dict(self._args.bcf_args)
        if 'fps' in headers_args:
            del headers_args['fps']
        return boxcars_py.get_column_headers(**headers_args)

    @functools.cached_property
    def load_game_from_filepath(self):
        def _load_game_from_filepath(filepath, **kwargs):
            call_kwargs = dict(self._args.bcf_args)
            call_kwargs.update(kwargs)
            return boxcars_py.get_ndarray_with_info_from_replay_filepath(
                filepath, **call_kwargs
            )
        return _load_game_from_filepath

    # do json stuff
    def game_to_dictionary(self, filepath, **kwargs):
        meta, data = self.load_game_from_filepath(filepath, **kwargs)
        column_headers = meta['column_headers']
        meta = _replay_meta.ReplayMeta.from_boxcar_frames_meta(meta['replay_meta'])
        all_headers = list(column_headers['global_headers'])
        for index, player in enumerate(meta.player_order):
            for player_header in column_headers['player_headers']:
                #all_headers.append(f"player {index}({player.tracker_suffix}) - {player_header}")
                all_headers.append(f"player {index} - {player_header}")
        assert len(all_headers) == data.shape[1]
        #print(all_headers)
        return dict(zip(all_headers, [list(map(float, data[:,column])) for column in range(data.shape[1])]))

    def game_to_json(self, filepath, **kwargs):
        dictionary = self.game_to_dictionary(filepath, **kwargs)
        p1list, p2list = self.mmr_plot_to_json(filepath)
        #print(len(p1list))
        #print(len(p2list))
        #print(p1list[:5])
        #print(p2list[:5])
        dictionary['player 1 - mmr'] = p1list
        dictionary['player 2 - mmr'] = p2list
        return json.dumps(dictionary)

    def write_game_json_to_file(self, src_filepath, dest_filepath, **kwargs):
        with open(dest_filepath, "w") as f:
            f.write(self.game_to_json(src_filepath, **kwargs))

    def mmr_plot_to_json(self, src_filepath):
        meta, ndarray = self.load_game_from_filepath(src_filepath)
        self.model.eval()
        self.model.to(self.device)
        x = torch.stack([torch.tensor(ndarray)]).to(self.device)
        history = self.model.prediction_history(x)
        python_history = [
            [self.label_scaler.unscale(float(prediction)) for prediction in elem[0]]
            for elem in history
        ]
        p1list = [mmr[0] for mmr in python_history]
        p2list = [mmr[1] for mmr in python_history]
        #print(python_history[0])
        return p1list, p2list

    @functools.cached_property
    def loss_function(self):
        scale_target = 250
        scale_weight = 3.0
        scale_target = self.label_scaler.scale_no_translate(scale_target)
        weight_function = train.create_weight_function(scale_target, scale_weight)
        return train.WeightedByMSELoss(weight_by=weight_function)

    @functools.cached_property
    def model(self):
        model = build.ReplayModel(
            self.header_info, self.playlist, lstm_width=self._args.lstm_width,
        )
        if self._args.model_path and os.path.exists(self._args.model_path):
            model.load_state_dict(torch.load(self._args.model_path, map_location=self.device))
        model.to(self.device)
        return model

    @functools.cached_property
    def device(self):
        return torch.device('cpu')

    @functools.cached_property
    def ballchasing_requests_session(self):
        session = requests.Session()
        session.headers.update(Authorization=self._args.ballchasing_token)
        return session

    @functools.cached_property
    def uuid_to_path(self):
        return dict(util.get_replay_uuids_in_directory(
            self._args.replay_path
        ))

    def get_game_filepath_by_uuid(self, uuid):
        try:
            filepath = self.uuid_to_path[uuid]
        except KeyError:
            filepath = self.download_game_by_uuid(uuid)
            logger.info(f"Downloding game from ball chasing {filepath}")
        else:
            logger.info(f"Using found file at {filepath}")
        return filepath

    def download_game_by_uuid(self, uuid):
        response = self.ballchasing_requests_session.get(
            f"https://ballchasing.com/api/replays/{uuid}/file",
        )
        target_file = os.path.join(self._args.replay_path, "temp", f"{uuid}.replay")
        with open(target_file, 'wb') as f:
            f.write(response.content)
        return target_file

    def decorate(self, fn):
        @functools.wraps(fn)
        def wrapped():
            return fn(self)
        return wrapped


@_RLRMLBuilder.with_default
def load_game_dataset(builder: _RLRMLBuilder):
    """Convert the game provided through sys.argv."""
    assesor = ReplaySetAssesor(
        builder.cached_directory_replay_set,
        scorer=builder.player_mmr_estimate_scorer,
        playlist=builder.playlist
    )
    results = assesor.get_replay_statuses_by_rank(load_tensor=False)
    import ipdb; ipdb.set_trace()


def create_symlink_replay_directory():
    parser = _add_rlrml_args()
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('target_directory')
    args = parser.parse_args()
    builder = _RLRMLBuilder(args)

    assesor = ReplaySetAssesor(
        builder.cached_directory_replay_set,
        scorer=builder.player_mmr_estimate_scorer,
        playlist=builder.playlist
    )
    top_scoring_replays = assesor.get_top_scoring_n_replay_per_rank(args.count)
    all_uuids = [uuid for pairs in top_scoring_replays.values() for uuid, _ in pairs]
    util.symlink_replays(
        args.target_directory, all_uuids, builder.cached_directory_replay_set
    )


@_RLRMLBuilder.with_default
def host_plots(builder):
    """Run an http server that hosts plots of player mmr that in the cache."""
    _http_graph_server.make_routes(builder)
    _http_graph_server.app.run(host="0.0.0.0", port=5001)


def proxy():
    _setup_system_bus()
    from .network import proxy
    proxy.app.run(port=5002)


@_RLRMLBuilder.with_default
def get_cache_answer_uuids(builder):
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


@_RLRMLBuilder.add_args("uri")
def socks_proxy_get(builder: _RLRMLBuilder):
    for i in range(6):
        result = builder.tracker_network_cloud_scraper.get_player_data(
            {"__tracker_suffix__": "epic/colonel_panic8"}
        )
        print(result)


@_RLRMLBuilder.add_args("player_key")
def get_player(builder: _RLRMLBuilder):
    """Get the provided player either from the cache or the tracker network."""

    player = {"__tracker_suffix__": builder._args.player_key}
    data = builder.player_cache.get_player_data(
        player
    )
    print(json.dumps(data))
    import datetime
    print(builder.lookup_label(player, datetime.date.today()))


@_RLRMLBuilder.add_args('iterations')
def train_model(builder: _RLRMLBuilder):
    import rich.live
    from .model import display

    def do_train(*args, **kwargs):
        with rich.live.Live() as live:
            live_stats = display.TrainLiveStatsDisplay(live, scaler=builder.label_scaler)
            trainer = train.ReplayModelManager.from_dataset(
                builder.torch_dataset, model=builder.model,
                on_epoch_finish=live_stats.on_epoch_finish,
                loss_function=builder.loss_function
            )
            trainer.train(*args, **kwargs)
    do_train(int(builder._args.iterations))
    import ipdb; ipdb.set_trace()


@_RLRMLBuilder.add_args("uuid")
def apply_model(builder: _RLRMLBuilder):
    meta, ndarray = builder.load_game_from_filepath(
        builder.get_game_filepath_by_uuid(builder._args.uuid)
    )
    builder.model.eval()
    builder.model.to(builder.device)
    x = torch.stack([torch.tensor(ndarray)]).to(builder.device)
    output = builder.model(x)
    print("Predictions: ")
    print([builder.label_scaler.unscale(float(label)) for label in output[0]])
    meta = _replay_meta.ReplayMeta.from_boxcar_frames_meta(meta['replay_meta'])
    print("Actual: ")
    print([builder.lookup_label(player, meta.datetime.date()) for player in meta.player_order])


@_RLRMLBuilder.with_default
def calculate_mean_absolute_loss(builder: _RLRMLBuilder):
    builder.model.eval()
    results = []
    for uuid, tensor, labels in builder.torch_dataset.iter_with_uuid():
        tensor = tensor.to(builder.device)
        labels = torch.stack([labels]).to(builder.device)
        predicted = builder.model(torch.stack([tensor]))
        values = (
            uuid,
            [float(f) for f in predicted[0]],
            [float(f) for f in labels[0]],
            float(torch.nn.functional.l1_loss(predicted, labels))
        )
        print(values)
        results.append(values)

    def get_l1_loss(values):
        _, predicted, labels, loss = values
        return loss

    results.sort(key=get_l1_loss, reverse=True)
    import ipdb; ipdb.set_trace()
    with open('./loss.json', 'w') as f:
        f.write(json.dumps(results))

    import ipdb; ipdb.set_trace()


@_RLRMLBuilder.add_args("tracker_suffix", "mmr")
def manual_override(builder: _RLRMLBuilder):
    builder.player_cache.insert_manual_override(
        {"__tracker_suffix__": builder._args.tracker_suffix}, builder._args.mmr
    )


@_RLRMLBuilder.with_default
def delete_if_less_than(builder:  _RLRMLBuilder):
    deleted = 0
    fine = 0
    for uuid, tensor, labels in builder.torch_dataset.iter_with_uuid():
        game_length = tensor.shape[0]
        if game_length < 1500:
            path = builder.cached_directory_replay_set.replay_path(uuid)
            deleted += 1
            os.remove(path)
        else:
            fine += 1

    logger.info(f"fine: {fine}, deleted: {deleted}")

@_RLRMLBuilder.add_args("src_filepath", "dest_filepath")
def game_to_json(builder: _RLRMLBuilder):
    builder.write_game_json_to_file(builder._args.src_filepath, builder._args.dest_filepath, fps=30, global_feature_adders = [ "BallRigidBodyNoVelocities", "SecondsRemaining" ])
    #builder.mmr_plot_to_json(builder._args.src_filepath, builder._args.dest_filepath);
