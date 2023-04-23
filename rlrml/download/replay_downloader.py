"""Defines a class that can be used to fetch replays from ballchasing.com in parallel."""
import aiohttp
import asyncio
import json
import logging
import os
import urllib

from .parallel_downloader import ParallelDownloader, ParallelDownloaderConfig
from .. import manifest


logger = logging.getLogger(__name__)


async def always_download_replay_filter(session, replay_meta) -> (bool, dict):
    """Download the provided replay unconditionally, make no modifications to the metadata."""
    return True, replay_meta


async def require_at_least_one_non_null_mmr(_, replay_meta):
    """Require that at least one of the mmrs in the replay meta from ballchasing is non-null.

    This is actually important because there are replays on ballchasing.com for
    which there is no estimate on mmr for any player that will pass through any
    minimum or maximum rank filter. This filter is a quick and dirty way to
    ensure that the filter is at least semi effective.
    """
    try:
        mmr_estimates = manifest.get_mmr_data_from_manifest_game(replay_meta)
    except Exception:
        logger.warn("Exception getting mmr_estimate")
        return False, replay_meta
    return any(
        value is not None
        for value in mmr_estimates.values()
    ), replay_meta


def build_filter_existing(replay_exists):
    """Filter any tasks for replays that already exist."""
    async def filter_existing(_, replay_meta):
        return (not replay_exists(replay_meta['id'])), replay_meta
    return filter_existing


def compose_filters(*filters):
    """Compose the provided filters."""
    async def new_filter(session, replay_meta):
        for next_filter in filters:
            should_enqueue, replay_meta = await next_filter(session, replay_meta)
            if not should_enqueue:
                break
        return should_enqueue, replay_meta
    return new_filter


def compose_filters_with_reasons(*filters):
    """Compose the provided filters."""
    async def new_filter(session, replay_meta):
        for (reason, next_filter) in filters:
            should_enqueue, replay_meta = await next_filter(session, replay_meta)
            if not should_enqueue:
                logger.warn(f"{replay_meta['id']} filtered because {reason}")
                break
        return should_enqueue, replay_meta
    return new_filter


def use_replay_id(replay_fetcher, replay_metadata):
    """Use the replay's id as the filename for the replay."""
    return os.path.join(
        replay_fetcher.download_path, "{}.replay".format(replay_metadata['id'])
    )


class ReplayDownloader(ParallelDownloaderConfig):
    """A configuration for :py:class:`ParallelDownloader`` to download replays from ballchasing.com.

    :param auth_token: An auth_token for ballchasing.com
    """

    def __init__(
            self, auth_token,
            replay_list_query_params=None,
            filepath_setter=use_replay_id,
            replay_filter=require_at_least_one_non_null_mmr,
            ballchasing_base_uri="https://ballchasing.com/api/",
            save_tasks_to_manifest=False,
            **kwargs
    ):
        """Initialize the replay fetcher."""
        super().__init__(**kwargs)
        self.auth_token = auth_token
        self.replay_list_query_params = replay_list_query_params or {"season": "f9"}
        self.replay_filter = replay_filter
        self.ballchasing_base_uri = ballchasing_base_uri
        self.filepath_setter = filepath_setter
        self.save_tasks_to_manifest = save_tasks_to_manifest
        self._enqueued_uuids = set()

    @property
    def _replay_list_request_query_params_string(self):
        return "&".join([
            "{}={}".format(key, value)
            for key, value
            in self.replay_list_query_params.items()
        ])

    @property
    def _replay_list_request_uri(self):
        return "{}replays".format(self.ballchasing_base_uri)

    def _replay_download_uri(self, replay_id):
        return "{}replays/{}/file".format(self.ballchasing_base_uri, replay_id)

    async def fetch_replays(self, *args, **kwargs):
        """Start fetching replays asynchronously."""
        async with aiohttp.ClientSession(
                headers={'Authorization': self.auth_token}, *args, **kwargs
        ) as session:
            replay_metadata = await ParallelDownloader(self, session).run()
            if self.save_tasks_to_manifest:
                replay_dict = dict((r["id"], r) for r in replay_metadata)
                with open(os.path.join(self.download_path, "manifest.json"), "w") as f:
                    f.write(json.dumps(replay_dict))

    def start_event_loop_and_run(self, *args, **kwargs):
        """Initialize a :py:class:`asyncio.BaseEventLoop` call :py:meth:`fetch_replays`."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            asyncio.Task(self.fetch_replays(*args, **kwargs))
        )

    # ParalellDownloaderConfig Implementation:

    def _readd_original_query_parameters(self, url):
        parsed = urllib.parse.urlparse(url)
        parsed_params = urllib.parse.parse_qs(parsed.query)
        new_params = dict(self.replay_list_query_params)
        new_params['after'] = parsed_params['after'][0]
        logger.info(new_params)
        logger.info(url)
        return parsed._replace(query='').geturl(), new_params

    async def get_tasks_and_next_request_from_response(self, session, response_obj):
        """Get the next request task and items to be enqueued from a response."""
        if not response_obj:
            return session.get(
                self._replay_list_request_uri,
                params=self.replay_list_query_params,
            ), []
        else:
            response = await response_obj.json()
            uri, params = self._readd_original_query_parameters(response["next"])
            return (
                session.get(uri, params=params),
                response["list"]
            )

    def get_filepath(self, task_meta):
        """Return the filepath that should be used to store the download."""
        return self.filepath_setter(self, task_meta)

    def get_request_for_task(self, session, task_meta):
        """Return the request that should be used to download."""
        return session.get(self._replay_download_uri(task_meta["id"]))

    async def get_filter_task(self, session, task_meta):
        """Return a coroutine to filter the provided task."""
        uuid = task_meta['id']
        if uuid in self._enqueued_uuids:
            logger.warn(f"Attempted enqueue of already enqueued uuid {uuid}")
            return False, task_meta
        should_enqueue, task_meta = await self.replay_filter(session, task_meta)
        if should_enqueue:
            self._enqueued_uuids.add(uuid)

        return should_enqueue, task_meta
