"""Defines a mechanism by which files can be downloaded in parallel."""
import aiohttp
import aiofiles
import asyncio
import abc
import backoff
import os
from .progress import DoNothingProgressHandler


class ParallelDownloaderConfig(abc.ABC):
    """A configuration for :py:class:`ParallelDownloader`."""

    def __init__(
            self, filter_queue_max_size=10, download_count=50, download_path=".",
            download_task_count=4, filter_task_count=4, backoff_retries=8,
            progress_handler=DoNothingProgressHandler()
    ):
        """Initialize basic paramters applicable to all ParallelDownloaderConfigs."""
        self.filter_queue_max_size = filter_queue_max_size
        self.download_path = download_path
        self.download_count = download_count
        self.download_task_count = download_task_count
        self.filter_task_count = filter_task_count
        self.progress_handler = progress_handler
        self.backoff_retires = backoff_retries

    @abc.abstractmethod
    async def get_tasks_and_next_request_from_response(self, session, response):
        """Get the next request task and items to be enqueued from a response."""
        pass

    @abc.abstractmethod
    def get_filepath(self, task_meta):
        """Return the filepath that should be used to store the download."""
        pass

    @abc.abstractmethod
    def get_request_for_task(self, session, task_meta):
        """Return the request task that should be used to download."""
        pass

    @abc.abstractmethod
    def get_filter_task(self, session, task_meta):
        """Return a coroutine to filter the provided task."""
        pass

    def regular_callback(self, invocation):
        """Report updates. Called regularly throughout the download process."""
        pass

    def item_downloaded(self, task_meta):
        """Report updates due to item download."""
        pass


class ParallelDownloader:
    """Orchestrate the downloading of files that are listed at a series of URIs."""

    def __init__(self, config, session):
        """Initialize with a config and a session."""
        self._config = config
        self._session = session
        self._next_request = None
        self._downloaded_items_metadata = []
        self._filter_queue = asyncio.Queue(
            maxsize=self._config.filter_queue_max_size
        )
        self._download_queue = asyncio.Queue()
        self.backoff_count = 0

    async def run(self):
        """Execute this :py:class:`ParallelDownloader`."""
        if not os.path.exists(self._config.download_path):
            os.makedirs(self._config.download_path)
        loop = asyncio.get_running_loop()
        queue_fill_task = loop.create_task(self._keep_filter_queue_filled())
        await asyncio.wait([
            self._update_progress()
        ] + [
            asyncio.create_task(self._download_until_count_reached())
            for _ in range(self._config.download_task_count)
        ] + [
            asyncio.create_task(self._filter_enequeued_items())
            for _ in range(self._config.filter_task_count)
        ])
        queue_fill_task.cancel()
        loop.create_task(self._next_request).cancel()
        return self._downloaded_items_metadata

    @property
    def number_of_items_downloaded(self):
        """Return a count of the number of items downloaded so far."""
        return len(self._downloaded_items_metadata)

    @property
    def _number_of_items_left_to_download(self):
        return self._config.download_count - self.number_of_items_downloaded

    @property
    def _number_of_items_left_to_enqueue(self):
        return self._number_of_items_left_to_download - self._download_queue.qsize()

    async def _keep_filter_queue_filled(self):
        while self._number_of_items_left_to_enqueue > 0:
            try:
                await self._get_next_item_list_page()
            except Exception as e:
                print("Hit exception keeping queue filled {}".format(e))
                raise e

    async def _download_until_count_reached(self):
        while self._number_of_items_left_to_download > 0:
            try:
                await self._dequeue_and_download_item()
            except Exception as e:
                print("Hit exception downloading {}".format(e))
                raise e

    async def _dequeue_and_download_item(self):
        task = await self._download_queue.get()
        filepath = self._config.get_filepath(task)

        @backoff.on_exception(
            backoff.expo, aiohttp.ClientResponseError,
            max_tries=self._config.backoff_retires,
            on_backoff=self._increment_backoff
        )
        async def download_item():
            # TODO: Check to ensure that file does not already exist
            async with self._config.get_request_for_task(self._session, task) as response:
                response.raise_for_status()
                async for data in response.content.iter_chunked(1024 * 16):
                    async with aiofiles.open(filepath, "ba") as f:
                        await f.write(data)

        await download_item()
        self._config.progress_handler.item_downloaded(task, self)
        self._downloaded_items_metadata.append(task)

    def _increment_backoff(self, *args, **kwargs):
        self.backoff_count += 1

    async def _filter_enequeued_items(self):
        while self._number_of_items_left_to_enqueue > 0:
            try:
                await self._filter_and_enqueue_to_download_if_appropriate()
            except Exception as e:
                print("Hit exception filtering {}".format(e))
                raise e

    async def _filter_and_enqueue_to_download_if_appropriate(self):
        task_meta = await self._filter_queue.get()
        (
            should_download, task
        ) = await self._config.get_filter_task(self._session, task_meta)

        self._config.progress_handler.item_filtered(task_meta, should_download, self)
        if should_download:
            await self._download_queue.put(task)

    async def _get_next_item_list_page(self, retries=0):
        if not self._next_request:
            self._next_request, _ = \
                await self._config.get_tasks_and_next_request_from_response(
                    self._session, None
                )
        async with self._next_request as response:
            self._next_request, tasks = \
                await self._config.get_tasks_and_next_request_from_response(
                    self._session, response
                )
        self._config.progress_handler.item_list_updated(tasks, self)
        for task_meta in tasks:
            await self._filter_queue.put(task_meta)

    async def _update_progress(self):
        while self._number_of_items_left_to_download > 0:
            self._config.progress_handler.downloader_ping(self)
            await asyncio.sleep(2)
