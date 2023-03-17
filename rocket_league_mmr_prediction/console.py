"""Defines command line entrypoints to the this library."""
from . import load


def test_request():
    import httpx
    client = httpx.Client(http2=True)
    headers = {'user-agent': 'testing', 'accept': "*/*", 'authority': "api.tracker.gg"}
    import ipdb; ipdb.set_trace()
    print(client.get("https://api.tracker.gg/api/v1/rocket-league/player-history/mmr/15972280", headers=headers))


def convert_replay():
    """Convert the game provided through sys.argv."""
    import sys
    data_set = load.ReplayDirectoryDataLoader(sys.argv[1])
    for i in data_set:
        import ipdb; ipdb.set_trace()
        pass
    # numpy_array = load._CarballToNumpyConverter(
    #     load.get_carball_game(sys.argv[1])
    # ).get_numpy_array()
    print("done")
