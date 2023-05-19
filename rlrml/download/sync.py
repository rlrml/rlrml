import backoff
import os
import requests
from .. import util
import urllib


class SynchronousReplayDownloader:

    def __init__(
            self, auth_token, download_path,
            replay_list_query_params=None,
            replay_filter=lambda x: True,
            ballchasing_base_uri="https://ballchasing.com/api/",
            all_replays_directory=None,
            symlink_if_known=True,
            **kwargs
    ):
        self._auth_token = auth_token
        self._replay_list_query_params = replay_list_query_params or {"season": "f9"}
        self._replay_filter = replay_filter
        self._ballchasing_base_uri = ballchasing_base_uri
        self._download_path = download_path
        self._session = requests.Session()
        self._session.headers = {'Authorization': self._auth_token}
        self._all_replays_directory = all_replays_directory
        self._symlink_if_known = symlink_if_known
        if all_replays_directory:
            self._uuid_to_path = util.get_replay_uuids_in_directory(all_replays_directory)
        else:
            self._uuid_to_path = {}

    @property
    def _replay_list_request_uri(self):
        return "{}replays".format(self._ballchasing_base_uri)

    def _replay_download_uri(self, replay_id):
        return "{}replays/{}/file".format(self._ballchasing_base_uri, replay_id)

    def _readd_original_query_parameters(self, url):
        parsed = urllib.parse.urlparse(url)
        parsed_params = urllib.parse.parse_qs(parsed.query)
        new_params = dict(self._replay_list_query_params)
        new_params['after'] = parsed_params['after'][0]
        return parsed._replace(query='').geturl(), new_params

    def _get_next_replay_page(self, next_uri):
        uri, params = self._readd_original_query_parameters(next_uri)
        return self._session.get(uri, params=params)

    def download_replays(self, count):
        os.makedirs(self._download_path, exist_ok=True)
        downloaded_count = 0
        response = self._session.get(
            self._replay_list_request_uri, params=self._replay_list_query_params
        ).json()
        while downloaded_count < count:
            next_uri, download_count = self._process_page_response(response)
            downloaded_count += download_count
            response = self._get_next_page_response(next_uri).json()

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.Timeout, requests.exceptions.ConnectionError)
    )
    def _get_next_page_response(self, next_uri):
        uri, params = self._readd_original_query_parameters(next_uri)
        return self._session.get(uri, params=params, timeout=6)

    def _process_page_response(self, response):
        count = 0
        for replay_meta in response['list']:
            if self._process_replay(replay_meta):
                count += 1
        return response['next'], count

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.Timeout, requests.exceptions.ConnectionError)
    )
    def _process_replay(self, replay_meta):
        target_filepath = self._get_filepath(replay_meta)
        uuid = replay_meta['id']
        if uuid in self._uuid_to_path:
            if self._symlink_if_known:
                source_filepath = self._uuid_to_path[uuid]
                if source_filepath != target_filepath:
                    os.symlink(source_filepath, target_filepath)
            return
        if self._replay_filter(replay_meta):
            uri = self._replay_download_uri(uuid)
            response = self._session.get(uri, timeout=6)
            with open(target_filepath, 'wb') as f:
                f.write(response.content)

    def _get_filepath(self, task_meta):
        """Return the filepath that should be used to store the download."""
        uuid = task_meta['id']
        return os.path.join(self._download_path, f"{uuid}.replay")
