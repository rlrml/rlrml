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
import numpy as np

from pathlib import Path

from . import _http_graph_server
from . import score
from . import load
from . import logger
from . import player_cache as pc
from . import tracker_network
from . import util
from . import loss
from . import vpn
from . import _replay_meta
from . import assess
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
        '--scale-positions',
        default=True,
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
        default=768,
    )
    parser.add_argument(
        '--lstm-depth',
        type=int,
        default=3,
    )
    parser.add_argument(
        '--lmdb',
        action='store_const',
        const='lmbd',
        dest='db_backend',
        default=defaults.get('db-backend', 'lmdb'),
    )
    parser.add_argument(
        '--level-db',
        action='store_const',
        const='leveldb',
        dest='db_backend',
    )
    parser.add_argument(
        '--loss-type',
        type=loss.LossType,
        choices=list(loss.LossType),
        default=loss.LossType.DIFFERENCE_AND_MSE_LOSS,
    )
    parser.add_argument(
        '--loss-param', '-l', action='append', nargs=2, metavar=('PARAM', 'VALUE'),
        help="Add loss function parameter", dest='loss_params', default={},
    )
    parser.add_argument(
        '--mmr-required-for-all-but',
        type=int,
        default=defaults.get('mmr-required-for-all-but', 0)
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=defaults.get('batch-size', 16)
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=defaults.get('learning-rate', .0001)
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
        self.args = args
        logger.info(f"Runnign with {self.args}")

    @functools.cached_property
    def label_scaler(self):
        return util.HorribleHackScaler

    @functools.cached_property
    def position_scaler(self):
        return util.ReplayPositionRescaler(self.header_info, self.playlist)

    @functools.cached_property
    def vpn_cycle_status_codes(self):
        return (429, 403)

    @functools.cached_property
    def player_cache(self):
        return (
            pc.PlayerCache.plyvel
            if self.args.db_backend == "leveldb"
            else pc.PlayerCache.lmdb
        )(str(self.args.player_cache))

    @functools.cached_property
    def cached_get_player_data(self):
        getter = pc.CachedGetPlayerData(
            self.player_cache, self.network_get_player_data, retry_errors=("500",)
        )
        ids_to_check = ['14d9cf07-ad46-440c-bac5-000d778449c2']

        def _hack_to_check_for_bad_data(player, *args, **kwargs):
            player_data = getter.get_player_data(player, *args, **kwargs)
            if player_data and 'platform' in player_data and (
                    player_data['platform']['platformUserId'] in ids_to_check
            ):
                old_id = player_data['platform']['platformUserId']
                player_data = getter.get_player_data(player, force_refresh=True)
                try:
                    new_id = player_data['platform']['platformUserId']
                except Exception:
                    if '__error__' not in player_data:
                        import ipdb; ipdb.set_trace()
                else:
                    logger.warn(f"Weird id {old_id} for {player}, after {new_id}")
            return player_data
        return _hack_to_check_for_bad_data

    @functools.cached_property
    def tracker_network_cloud_scraper(self):
        return tracker_network.CloudScraperTrackerNetwork(
            proxy_uris=self.args.socks_proxy_urls
        )

    @functools.cached_property
    def bare_get_player_data(self):
        return self.tracker_network_cloud_scraper.get_player_data

    @functools.cached_property
    def network_get_player_data(self):
        if self.args.cycle_vpn:
            return self.vpn_cycled_get_player_data
        else:
            return tracker_network.get_player_data_with_429_retry()

    @functools.cached_property
    def playlist(self):
        return Playlist.from_string_or_number(self.args.playlist)

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
            self.cached_get_player_data, truncate_lowest_count=self.args.mmr_required_for_all_but
        )

    @functools.cached_property
    def cached_directory_replay_set(self):
        return load.DirectoryReplaySet.cached(
            self.args.tensor_cache, self.args.replay_path,
            boxcar_frames_arguments=self.args.bcf_args,
            tensor_transformer=self.position_scaler.scale_position_columns
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
            self.playlist, self.header_info, preload=self.args.preload,
            label_scaler=self.label_scaler
        )

    @functools.cached_property
    def header_info(self):
        headers_args = dict(self.args.bcf_args)
        if 'fps' in headers_args:
            del headers_args['fps']
        return boxcars_py.get_column_headers(**headers_args)

    @functools.cached_property
    def load_game_from_filepath(self):
        def _load_game_from_filepath(filepath, **kwargs):
            call_kwargs = dict(self.args.bcf_args)
            call_kwargs.update(kwargs)
            meta, tensor = boxcars_py.get_ndarray_with_info_from_replay_filepath(
                filepath, **call_kwargs
            )
            return meta, self.position_scaler.scale_position_columns(tensor)
        return _load_game_from_filepath

    def game_to_dictionary(self, filepath, **kwargs):
        meta, data = self.load_game_from_filepath(filepath, **kwargs)
        column_headers = meta['column_headers']
        meta = _replay_meta.ReplayMeta.from_boxcar_frames_meta(meta['replay_meta'])
        all_headers = list(column_headers['global_headers'])
        for index, player in enumerate(meta.player_order):
            for player_header in column_headers['player_headers']:
                all_headers.append(f"player {index} - {player_header}")
        assert len(all_headers) == data.shape[1]
        return dict(zip(all_headers, [
            list(map(float, data[:, column])) for column in range(data.shape[1])
        ]))

    def game_to_json(self, filepath, **kwargs):
        dictionary = self.game_to_dictionary(filepath, **kwargs)
        p1list, p2list = self.mmr_plot_to_json(filepath)
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
        return p1list, p2list

    @functools.cached_property
    def loss_function(self):
        return self.args.loss_type.get_fn_from_args(**self.args.loss_params)

    @functools.cached_property
    def model(self):
        model = build.ReplayModel(
            self.header_info, self.playlist, lstm_width=self.args.lstm_width,
            lstm_depth=self.args.lstm_depth
        )
        if self.args.model_path and os.path.exists(self.args.model_path):
            logger.info(f"Loading model path from {self.args.model_path}")
            model.load_state_dict(torch.load(self.args.model_path, map_location=self.device))
        model.to(self.device)
        return model

    @functools.cached_property
    def device(self):
        return torch.device('cuda')

    @functools.cached_property
    def ballchasing_requests_session(self):
        session = requests.Session()
        session.headers.update(Authorization=self.args.ballchasing_token)
        return session

    @functools.cached_property
    def uuid_to_path(self):
        return dict(util.get_replay_uuids_in_directory(
            self.args.replay_path
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
        target_file = os.path.join(self.args.replay_path, "temp", f"{uuid}.replay")
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
    assesor = assess.ReplaySetAssesor(
        builder.cached_directory_replay_set,
        scorer=builder.player_mmr_estimate_scorer,
        playlist=builder.playlist
    )
    results = assesor.get_replay_statuses_by_rank(load_tensor=False)
    import ipdb; ipdb.set_trace()


@_RLRMLBuilder.add_args("target_directory")
def create_symlink_replay_directory(builder):
    # assess.ParallelTensorMetaLoader.load_all(builder.cached_directory_replay_set)
    assesor = assess.ReplaySetAssesor(
        builder.cached_directory_replay_set,
        scorer=builder.player_mmr_estimate_scorer,
        playlist=builder.playlist
    )
    top_scoring_replays = assesor.get_top_scoring_n_replay_per_rank(1000)
    all_uuids = [uuid for pairs in top_scoring_replays.values() for uuid, _ in pairs]
    def do_symlink():
        util.symlink_replays(
            builder.args.target_directory, all_uuids, builder.cached_directory_replay_set
        )
    import ipdb; ipdb.set_trace()
    do_symlink()


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
def get_cache_answer_uuids(builder: _RLRMLBuilder):
    uuids = list(util.get_cache_answer_uuids_in_directory(
        builder.args.replay_path, builder.player_cache
    ))
    print(len(uuids))
    import ipdb; ipdb.set_trace()


@_RLRMLBuilder.add_args("uuid")
def ballchasing_lookup(builder: _RLRMLBuilder):
    game_data = requests.get(
        f"https://ballchasing.com/api/replays/{builder.args.uuid}",
        headers={'Authorization': builder.args.ballchasing_token},
    ).json()
    meta = _replay_meta.ReplayMeta.from_ballchasing_game(game_data)
    for player in meta.player_order:
        label = builder.lookup_label(player, meta.datetime)
        print(f"{player} - {label}")


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

    player = {"__tracker_suffix__": builder.args.player_key}
    builder.cached_get_player_data(player, force_refresh=True)
    data = builder.player_cache.get_player_data(
        player
    )
    print(json.dumps(data['platform']))
    import datetime
    print(builder.lookup_label(player, datetime.date.today()))


@_RLRMLBuilder.with_default
def train_model(builder: _RLRMLBuilder):
    import rich.live
    from .model import display

    def do_train(*args, **kwargs):
        with rich.live.Live() as live:
            live_stats = display.TrainLiveStatsDisplay(live, scaler=builder.label_scaler)
            trainer = train.ReplayModelManager.from_dataset(
                builder.torch_dataset, model=builder.model,
                on_epoch_finish=live_stats.on_epoch_finish,
                loss_function=builder.loss_function, lr=builder.args.learning_rate,
                batch_size=builder.args.batch_size
            )
            trainer.train(*args, **kwargs)
    do_train(10)
    import ipdb; ipdb.set_trace()


@_RLRMLBuilder.add_args("uuid")
def apply_model(builder: _RLRMLBuilder):
    meta, ndarray = builder.load_game_from_filepath(
        builder.get_game_filepath_by_uuid(builder.args.uuid)
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
    results = []

    def unscale(values):
        return [builder.label_scaler.unscale(v) for v in values]

    builder.model.eval()

    trainer = train.ReplayModelManager.from_dataset(
        builder.torch_dataset, model=builder.model,
        loss_function=builder.loss_function, lr=builder.args.learning_rate,
        batch_size=builder.args.batch_size
    )

    results = []

    def process(training_data, y_pred, losses):
        print("Next Batch")
        unscaled_y = builder.label_scaler.unscale(training_data.y)
        unscaled_y_pred = builder.label_scaler.unscale(y_pred)

        results.extend(zip(
            training_data.uuids,
            unscaled_y.tolist(),
            unscaled_y_pred.tolist(),
            losses.tolist(),
            map(np.mean, losses.tolist()),
        ))

    trainer.process_loss(process)

    results.sort(key=lambda v: v[4])

    def save_results(filename='./loss.json'):
        with open(filename, 'w') as f:
            f.write(json.dumps(results))

    import ipdb; ipdb.set_trace()


@_RLRMLBuilder.add_args("tracker_suffix", "mmr")
def manual_override(builder: _RLRMLBuilder):
    builder.player_cache.insert_manual_override(
        {"__tracker_suffix__": builder.args.tracker_suffix}, builder.args.mmr
    )


@_RLRMLBuilder.with_default
def delete_if_less_than(builder: _RLRMLBuilder):
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


@_RLRMLBuilder.add_args("game_uuid")
def score_game(builder: _RLRMLBuilder):
    meta = builder.cached_directory_replay_set.get_replay_meta(builder.args.game_uuid)
    score = builder.player_mmr_estimate_scorer.score_replay_meta(meta, playlist=builder.playlist)
    print(meta.player_order)
    print(score)
    builder.model.eval()
    builder.cached_directory_replay_set.get_replay_tensor()
    builder.model()


@_RLRMLBuilder.add_args("src_filepath", "dest_filepath")
def game_to_json(builder: _RLRMLBuilder):
    builder.write_game_json_to_file(
        builder.args.src_filepath, builder.args.dest_filepath,
        fps=30, global_feature_adders=["BallRigidBodyNoVelocities", "SecondsRemaining"]
    )


@_RLRMLBuilder.add_args("lmdb_filepath")
def lmdb_migrate(builder: _RLRMLBuilder):
    lmdb_cache = pc.PlayerCache.lmdb(builder.args.lmdb_filepath)
    migrate_cache_raw(builder.player_cache, lmdb_cache)


def migrate_cache_raw(source_cache: pc.PlayerCache, dest_cache: pc.PlayerCache):
    for key, value in source_cache._db.iterator():
        dest_cache._db.put(key, value)
