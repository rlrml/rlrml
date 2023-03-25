"""Define an interface for replay download handling progress and useful implementations."""
import abc


class DownloadProgressHandler(abc.ABC):
    """Interface for the handling of progress updates from a download manager."""

    @abc.abstractmethod
    def item_downloaded(self, item, downloader):
        """Handle the completed download of the provided item."""
        pass

    @abc.abstractmethod
    def item_list_updated(self, items, downloader):
        """Handle an update to the items in the filter queue."""
        pass

    @abc.abstractmethod
    def item_filtered(self, item_metadata, included, downloader):
        """Handle a item being either being selected or not for downloading."""
        pass

    @abc.abstractmethod
    def downloader_ping(self, downloader):
        """Handle being called at a regular interval by the download manager."""
        pass


class DoNothingProgressHandler(DownloadProgressHandler):
    """Do nothing on progress updates."""

    def item_downloaded(self, item, downloader):
        """Handle the completed download of the provided item."""
        pass

    def item_list_updated(self, items, downloader):
        """Handle an update to the items in the filter queue."""
        pass

    def item_filtered(self, item_metadata, included, downloader):
        """Handle a item being either being selected or not for downloading."""
        pass

    def downloader_ping(self, downloader):
        """Handle being called at a regular interval by the download manager."""
        pass


class BarProgressHandler(DownloadProgressHandler):
    """Present a progress bar using tqdm."""

    def __init__(self, pbar):
        """Store the tqdm manager."""
        self._pbar = pbar

    def _update_description(self, downloader):
        description_text = "filter_q: {}, download_q: {}, backoff_count: {}".format(
            downloader._filter_queue.qsize(),
            downloader._download_queue.qsize(),
            downloader.backoff_count
        )
        self._pbar.set_description_str(description_text)

    def item_downloaded(self, item, downloader):
        """Handle the completed download of the provided item."""
        self._pbar.update(downloader.number_of_items_downloaded - self._pbar.n)

    def item_list_updated(self, items, downloader):
        """Handle an update to the items in the filter queue."""
        pass

    def item_filtered(self, item_metadata, included, downloader):
        """Handle a item being either being selected or not for downloading."""
        pass

    def downloader_ping(self, downloader):
        """Handle being called at a regular interval by the download manager."""
        self._update_description(downloader)
