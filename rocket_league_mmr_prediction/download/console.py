"""Defines a command line interface to the replay fetcher defined in this package."""
import argparse
import coloredlogs
import os
import pprint
import tqdm
import logging

from . import replay_downloader
from .progress import BarProgressHandler

from ..filter import MMREstimateQualityFilter
from .. import console
from .._replay_meta import ReplayMeta
from .. import player_cache as pc
from .. import tracker_network
from .. import util
from .. import _replay_meta


def run():
    """Entry point to the command line interface to ballchasing_replay_fetcher."""
    parser = console._add_rlrml_args(argparse.ArgumentParser())

    default_path = os.path.join(os.getcwd(), "replays")
    parser.add_argument(
        '--auth-token', help="A ballchasing.com authorization token.", type=str, required=True
    )
    parser.add_argument('--count', type=int, help="Number of replays to download", default=100)
    parser.add_argument(
        '--path', type=str, help="The directory in which to store replays", default=default_path
    )
    parser.add_argument(
        '--task-count', type=int, default=4,
        help="The number of concurrent tasks that should be used for each queue"
    )
    parser.add_argument(
        '--query', '-q', action='append', nargs=2, metavar=('PARAM', 'VALUE'),
        help="Add a query parameter"
    )
    parser.add_argument(
        '--logs', type=bool, default=True
    )
    parser.add_argument(
        '--all-replays-dir', type=str,
        help="The directory to use for replay filtering", default=None
    )
    args = parser.parse_args()
    all_replays_dir = args.all_replays_dir or args.path

    existing_replay_uuids = set(
        uuid for uuid, _ in util.get_replay_uuids_in_directory(all_replays_dir)
    )

    player_cache = pc.PlayerCache(str(args.player_cache))

    def replay_exists(uuid):
        return uuid in existing_replay_uuids

    async def check_for_cached_error(_, replay_meta):
        try:
            meta = _replay_meta.ReplayMeta.from_ballchasing_game(replay_meta)
            should_enqueue = (not any(bool(player_cache.has_error(player)) for player in meta.player_order))
            if not should_enqueue:
                replay_downloader.logger.warn(f"Filtering due to error player {replay_meta}")
            return (
                should_enqueue,
                replay_meta
            )
        except Exception as e:
            import ipdb; ipdb.set_trace()
            print(f"filtering error {e}")
            return True

    task_filter = replay_downloader.compose_filters(
        replay_downloader.require_at_least_one_non_null_mmr,
        replay_downloader.build_filter_existing(replay_exists),
        check_for_cached_error,
    )

    if args.logs:
        coloredlogs.install(level='INFO', logger=replay_downloader.logger)
        replay_downloader.logger.setLevel(logging.INFO)

    # player_cache = pc.PlayerCache(str(args.player_cache))

    # cached_player_get = pc.CachedGetPlayerData(
    #     player_cache, tracker_network.get_player_data_with_429_retry()
    # ).get_player_data

    # mmr_filter = MMREstimateQualityFilter(cached_player_get)

    # async def task_filter(session, replay_meta):
    #     should_continue, replay_meta = await require_at_least_one_non_null_mmr(
    #         session, replay_meta
    #     )
    #     if not should_continue:
    #         return should_continue, replay_meta

    #     return mmr_filter.meta_download_filter(
    #         ReplayMeta.from_ballchasing_game(replay_meta)
    #     ), replay_meta

    query_params = dict(args.query)

    pretty_params = pprint.PrettyPrinter().pformat(query_params)
    query_string = f"with query parameters:\n\n {pretty_params}\n"

    print(
        f"\nDownloading {args.count} replays to:\n\n{args.path}\n\n{query_string}"
    )

    with tqdm.tqdm(total=args.count) as pbar:
        replay_downloader.ReplayDownloader(
            args.ballchasing_token, download_path=args.path, download_count=args.count,
            download_task_count=args.task_count, filter_task_count=args.task_count,
            progress_handler=BarProgressHandler(pbar),
            replay_list_query_params=query_params,
            replay_filter=task_filter,
        ).start_event_loop_and_run()
