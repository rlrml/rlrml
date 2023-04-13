import backoff
import os
import boxcars_py

from . import vpn
from . import tracker_network
from . import player_cache as pc
from . import _replay_meta


def _constant_retry(constant):
    def get_value(exception):
        value_to_return = constant
        return value_to_return
    return get_value


def vpn_cycled_cached_player_get(
        filepath, player_cache=None, status_codes=(429, 403), *args, **kwargs
):
    """Apply vpn cycling and caching to `get_player_data`."""
    player_cache = player_cache or pc.PlayerCache.new_with_cache_directory(filepath)
    scraper_tn = tracker_network.CloudScraperTrackerNetwork()
    get_player_data = kwargs.pop(
        "get_player_data", scraper_tn.get_player_data
    )
    vpn_cycler = vpn.VPNCycler(*args, **kwargs)
    kwargs.setdefault("value", _constant_retry(8))

    return pc.CachedGetPlayerData(
        player_cache,
        vpn_cycler.cycle_vpn_backoff(
            backoff.runtime,
            tracker_network.Non200Exception,
            giveup=lambda e: e.status_code not in status_codes,
            on_backoff=lambda d: scraper_tn.refresh_scraper(),
            *args, **kwargs
        )(get_player_data)
    ).get_player_data


def get_replay_uuids_in_directory(filepath, replay_extension="replay"):
    for root, _, files in os.walk(filepath):
        for filename in files:
            replay_id, extension = os.path.splitext(filename)
            if extension and extension[1:] == replay_extension:
                replay_path = os.path.join(root, filename)
                yield replay_id, replay_path


def get_cache_answer_uuids_in_directory(filepath, player_cache: pc.PlayerCache):
    for uuid, filepath in get_replay_uuids_in_directory(filepath):
        try:
            data_present = player_data_present(filepath, player_cache)
        except Exception as e:
            print(f"Exception {e}")
            continue
        else:
            if data_present:
                yield uuid, filepath


def player_data_present(replay_path, player_cache: pc.PlayerCache):
    meta = _replay_meta.ReplayMeta.from_boxcar_frames_meta(
        boxcars_py.get_replay_meta(replay_path)
    )
    return all(player_cache.present_and_no_error(player) for player in meta.player_order)
