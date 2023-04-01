import backoff

from . import vpn
from . import tracker_network
from . import player_cache as pc


def vpn_cycled_cached_player_get(filepath, *args, **kwargs):
    """Apply vpn cycling and caching to `get_player_data`."""
    player_cache = pc.PlayerCache.new_with_cache_directory(filepath)
    get_player_data = kwargs.pop(
        "get_player_data", tracker_network.TrackerNetwork().get_player_data
    )
    vpn_cycler = vpn.VPNCycler(*args, **kwargs)

    return pc.CachedGetPlayerData(
        player_cache,
        vpn_cycler.cycle_vpn_backoff(
            backoff.runtime,
            tracker_network.Non200Exception,
            giveup=lambda e: e.status_code != 429,
            *args, **kwargs
        )(get_player_data)
    ).get_player_data
