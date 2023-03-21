"""Defines command line entrypoints to the this library."""
import sys
import functools
from . import load
from . import player_mmr_cache


def convert_replay():
    """Convert the game provided through sys.argv."""
    from . import tracker_network
    from . import player_mmr_cache
    player_mmr_cache.populate_mmr_cache_from_directory_using_tracker_network(
        sys.argv[1]
    )
    # data_set = load.ReplayDirectoryDataLoader(sys.argv[1])
    # for i in data_set:
    #     import ipdb; ipdb.set_trace()
    #     pass
    # numpy_array = load._CarballToNumpyConverter(
    #     load.get_carball_game(sys.argv[1])
    # ).get_numpy_array()
    print("done")


def _call_with_sys_argv(function):
    @functools.wraps(function)
    def call_with_sys_argv():
        function(sys.argv[1])
    return call_with_sys_argv


@_call_with_sys_argv
def fill_cache_with_manifest_rank(filepath):
    """Fill the mmr cache for filepath with ranks obtained from manifest files."""
    player_mmr_cache.populate_mmr_cache_from_directory_using_manifest_rank(filepath)
    mmr_cache = player_mmr_cache.MMRCache(filepath)
    print(f"{len(list(mmr_cache))} entries in mmr cache")


fill_cache_with_tracker_rank = _call_with_sys_argv(
    player_mmr_cache.populate_mmr_cache_from_directory_using_tracker_network
)
