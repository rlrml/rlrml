"""Defines command line entrypoints to the this library."""
import sys
import functools
from . import load
from . import cache
from . import tracker_network


def convert_replay():
    """Convert the game provided through sys.argv."""
    from . import player_mmr_cache
    player_mmr_cache.populate_mmr_cache_from_directory_using_tracker_network(
        *sys.argv[1:]
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
        function(*sys.argv[1:])
    return call_with_sys_argv


# @_call_with_sys_argv
# def fill_cache_with_manifest_rank(filepath, *args):
#     """Fill the mmr cache for filepath with ranks obtained from manifest files."""
#     player_mmr_cache.populate_mmr_cache_from_directory_using_manifest_rank(filepath)
#     mmr_cache = player_mmr_cache.MMRCache(filepath)
#     print(f"{len(list(mmr_cache))} entries in mmr cache")


@_call_with_sys_argv
def fill_cache_with_tracker_rank(filepath):
    """Fill a player info cache in a directory of replays."""
    import asyncio
    loop = asyncio.get_event_loop()
    task = cache.populate_player_cache_from_directory_using_tracker_network(filepath)
    loop.run_until_complete(task)


@_call_with_sys_argv
def _test_aiocurl(_suffix):
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_until_complete(_run_tracker_curl())


async def _run_tracker_curl():
    import json
    result = await tracker_network.TrackerNetworkCurler().get_info_for_player(
        {"id": {"platform": "epic"}, "name": "colonel_panic8"}
    )
    with open('./sample_player_json.json', 'w') as f:
        f.write(json.dumps(result))
    # result = await tracker_network.TrackerNetworkCurler().get_info_for_players([
    #     {"id": {"platform": "epic"}, "name": "colonel_panic8"},
    #     {"id": {"platform": "epic"}, "name": "elonsroadster"},
    #     {"id": {"platform": "epic"}, "name": "calemacar"}
    # ])
    return result


@_call_with_sys_argv
def test_result_filtering(filepath):
    """Test filtering of data."""
    import json
    with open(filepath, 'r') as f:
        obj = json.loads(f.read())

    print(json.dumps(tracker_network.combine_profile_and_mmr_json(obj)))


@_call_with_sys_argv
def _iter_cache(filepath):
    import json
    for player_key, player_data in cache.PlayerCache.new_with_cache_directory(filepath):
        print(json.dumps(player_data))


@_call_with_sys_argv
def copy_games(source, dest):
    import asyncio
    task = cache.copy_games_if_metadata_available_and_conditions_met(source, dest)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(task)
