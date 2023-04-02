import backoff

from . import vpn
from . import tracker_network
from . import player_cache as pc


def _constant_retry(constant):
    def get_value(exception):
        value_to_return = constant
        return value_to_return
    return get_value


def vpn_cycled_cached_player_get(filepath, player_cache=None, *args, **kwargs):
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
            giveup=lambda e: e.status_code not in (429, 403),
            on_backoff=lambda d: scraper_tn.refresh_scraper(),
            *args, **kwargs
        )(get_player_data)
    ).get_player_data
