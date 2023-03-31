"""Defines a command line interface to the replay fetcher defined in this package."""
import argparse
import os
import pprint
import tqdm
from .replay_fetcher import ReplayFetcher
from .progress import BarProgressHandler


def run():
    """Entry point to the command line interface to ballchasing_replay_fetcher."""
    parser = argparse.ArgumentParser()

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
    args = parser.parse_args()

    query_params = dict(args.query)

    pretty_params = pprint.PrettyPrinter().pformat(query_params)
    query_string = f"with query parameters:\n\n {pretty_params}\n"

    print(
        f"\nDownloading {args.count} replays to:\n\n{args.path}\n\n{query_string}"
    )

    with tqdm.tqdm(total=args.count) as pbar:
        ReplayFetcher(
            args.auth_token, download_path=args.path, download_count=args.count,
            download_task_count=args.task_count, filter_task_count=args.task_count,
            progress_handler=BarProgressHandler(pbar),
            replay_list_query_params=query_params,
        ).start_event_loop_and_run()
