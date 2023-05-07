"""Defines a command line interface to the replay fetcher defined in this package."""
import argparse
import coloredlogs
import os
import pprint
import tqdm
import logging

from . import replay_downloader
from . import filters
from .progress import BarProgressHandler

from .. import console
from .. import util
from .. import _replay_meta


logger = logging.getLogger(__name__)


def run():
    """Entry point to the command line interface to ballchasing_replay_fetcher."""
    parser = console._add_rlrml_args(argparse.ArgumentParser())

    default_path = os.path.join(os.getcwd(), "replays")
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
    parser.add_argument(
        '--minimum-replay-score', type=float, default=0.001
    )
    parser.add_argument(
        '--min-mmr-disparity', type=int, default=None,
    )
    parser.add_argument(
        '--min-duration', type=int, default=150
    )
    args = parser.parse_args()
    builder = console._RLRMLBuilder(args)
    builder._setup_default_logging()

    all_replays_dir = args.all_replays_dir or args.path

    existing_replay_uuids = set(
        uuid for uuid, _ in util.get_replay_uuids_in_directory(all_replays_dir)
    )

    def replay_exists(uuid):
        return uuid in existing_replay_uuids

    def filter_by_replay_score(replay_meta):
        meta = _replay_meta.ReplayMeta.from_ballchasing_game(replay_meta)
        score_info = builder.player_mmr_estimate_scorer.score_replay_meta(meta)

        additional_condition = True
        if args.min_mmr_disparity is not None:
            player_mmrs = [mmr for _, mmr in score_info.estimates]
            max_mmr = max(player_mmrs)
            min_mmr = min(player_mmrs)
            additional_condition = max_mmr - min_mmr >= args.min_mmr_disparity
            logger.info(f"{max_mmr}, {min_mmr}, {additional_condition}")

        return (
            additional_condition and score_info.meta_score >= args.minimum_replay_score,
            replay_meta
        )

    async def async_filter_by_replay_score(_, replay_meta):
        return filter_by_replay_score(replay_meta)

    def filter_by_duration(replay_meta):
        return replay_meta['duration'] > args.min_duration, replay_meta

    async def async_filter_by_duration(_, replay_meta):
        return filter_by_duration(replay_meta)

    task_filter = filters.compose_filters(
        filters.async_require_at_least_one_non_null_mmr,
        filters.build_filter_existing(replay_exists),
        async_filter_by_replay_score,
        async_filter_by_duration,
    )

    if args.logs:
        coloredlogs.install(level='INFO', logger=replay_downloader.logger)
        replay_downloader.logger.setLevel(logging.INFO)

    query_params = dict(args.query)
    query_params.setdefault('playlist', builder.playlist.ballchasing_filter_string)

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
