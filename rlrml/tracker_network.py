"""Utilities for getting data from TrackerNetwork's public API."""
import aiocurl
import backoff
import cloudscraper
import json
import logging
import requests

from io import BytesIO
from email.parser import BytesParser

from . import _replay_meta


logger = logging.getLogger(__name__)


default_tracker_uri = "https://api.tracker.gg"


class Non200Exception(Exception):
    """Exception raised when a tracker network http request gives a non-200 response."""

    def __init__(self, status_code, response_headers=None):
        """Initialize the Non200Exception, settings response headers."""
        self.status_code = status_code
        self.response_headers = response_headers or {}


def get_mmr_history_uri_by_id(tracker_player_id, base_uri=default_tracker_uri):
    """Get the uri at which mmr history for the given player can be found."""
    return f"{base_uri}/api/v1/rocket-league/player-history/mmr/{tracker_player_id}"


def get_profile_uri_for_player(player, base_uri=default_tracker_uri):
    """Get the uri that should be used to retrieve mmr info from the given player dictionary."""
    return f"{base_uri}/api/v2/rocket-league/standard/profile/{get_profile_suffix_for_player(player)}"


def get_profile_suffix_for_player(player):
    """Get the suffix to use to generate a players tracker network profile uri."""
    if isinstance(player, _replay_meta.PlatformPlayer):
        return player.tracker_suffix

    if "__tracker_suffix__" in player:
        return player["__tracker_suffix__"]

    platform = player["id"]["platform"]

    if platform == "steam":
        return f"steam/{player['id']['id']}"

    try:
        player_name = player["name"]
    except KeyError:
        return None

    space_replaced = player_name.replace(" ", "%20")

    if platform.startswith("p"):
        platform = "psn"

    if platform == "xbox":
        platform = "xbl"
    elif platform.startswith("p"):
        platform = "psn"

    return f"{platform}/{space_replaced}"


_tracker_playlist_id_to_name = {
    "10": "Ranked Duel 1v1",
    "11": "Ranked Doubles 2v2",
    "13": "Ranked Standard 3v3",
}


def _simplify_stats(stats):
    return {
        stat_name: data["value"]
        for stat_name, data in stats.items()
    }


def _simplify_mmr_history(items):
    return [(item["collectDate"], item["rating"]) for item in items]


def combine_profile_and_mmr_json(data):
    """Combine and filter the json obtained for a player from the profile and mmr endpoints."""
    profile = data["profile"]["data"]
    platform_info = profile["platformInfo"]
    metadata = profile["metadata"]

    segments = {
        segment.get("metadata").get("name"): segment
        for segment in profile["segments"]
    }

    mmr_data = data["mmr"]["data"]

    return {
        "platform": platform_info,
        "tracker_api_id": metadata["playerId"],
        "last_updated": metadata["lastUpdated"]["value"],
        "stats": _simplify_stats(segments["Lifetime"]["stats"]),
        "playlists": {
            segment_name: _simplify_stats(segment_data["stats"])
            for segment_name, segment_data in segments.items()
            if segment_name != "overview"
        },
        "mmr_history": {
            playlist_name: _simplify_mmr_history(mmr_data[tracker_playlist_id])
            for tracker_playlist_id, playlist_name in _tracker_playlist_id_to_name.items()
            if tracker_playlist_id in mmr_data
        }
    }


class CloudScraperTrackerNetwork:
    """Use the cloudscraper library to perform requests to the tracker network."""

    def __init__(self, base_uri="https://api.tracker.gg", proxy_uris=(None,)):
        """Initialize this class."""
        if len(proxy_uris) < 1:
            proxy_uris = (None,)
        self._scrapers = [self._build_scraper(uri) for uri in proxy_uris]
        self._proxy_uris = proxy_uris
        self._scraper_index = 0
        self._base_uri = base_uri

    def _build_scraper(self, proxy_uri=None):
        scraper = cloudscraper.create_scraper(delay=1, browser="chrome")
        if proxy_uri is not None:
            scraper.proxies = {
                "http": proxy_uri,
                "https": proxy_uri,
            }
        scraper.headers.update({
            'referer': "https://rocketleague.tracker.network",
            'origin': "https://rocketleague.tracker.network",
        })
        return scraper

    def _next_scraper(self):
        scraper = self._scrapers[self._scraper_index]
        self._scraper_index = (self._scraper_index + 1) % len(self._scrapers)
        return scraper

    @property
    def _scraper(self):
        return self._next_scraper()

    def refresh_scraper(self, offset=0):
        """Make a new scraper."""
        index = (self._scraper_index + offset) % len(self._scrapers)
        self._scrapers[index] = self._build_scraper(self._proxy_uris[index])

    def __get(self, uri):
        logger.debug(f"Cloud scraper request {uri}")
        resp = self._next_scraper().get(uri, timeout=6)
        if resp.status_code != 200:
            raise Non200Exception(resp.status_code, resp.headers)
        return resp.json()

    def _get(self, uri):
        try:
            return self.__get(uri)
        except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout):
            logger.warn("Trying refreshing scraper after a timeout")
            self.refresh_scraper(offset=-1)
            return self.__get(uri)

    def get_player_data(self, player):
        """Combine info from the main player page and the mmr history player page."""
        uri = get_profile_uri_for_player(player)
        profile_result = self._get(uri)
        tracker_api_id = profile_result["data"]["metadata"]["playerId"]

        mmr_history_uri = get_mmr_history_uri_by_id(tracker_api_id)
        mmr_result = self._get(mmr_history_uri)

        return combine_profile_and_mmr_json({
            "profile": profile_result,
            "mmr": mmr_result,
        })


def _log_backoff(details):
    exception = details["exception"]
    logger.info(f"Backing off {exception}")


def _use_retry_after(exception: Non200Exception):
    string_value = exception.response_headers.get('retry-after') or \
        exception.response_headers.get('Retry-After')
    if string_value is None:
        return 60
    return int(string_value)


def get_player_data_with_429_retry(get_player_data=CloudScraperTrackerNetwork().get_player_data):
    """Wrap the provided getter with 429 backoff that uses the retry-after header."""
    return backoff.on_exception(
        backoff.runtime, Non200Exception, max_time=300,
        giveup=lambda e: e.status_code != 429,
        on_backoff=_log_backoff,
        value=_use_retry_after,
        jitter=None,
    )(get_player_data)
