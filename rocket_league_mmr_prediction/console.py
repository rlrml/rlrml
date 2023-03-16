"""Defines command line entrypoints to the this library."""
from . import prepare


def convert_replay():
    """Convert the game provided through sys.argv."""
    import sys
    numpy_array = prepare._CarballToNumpyConverter(
        prepare.get_carball_game(sys.argv[1])
    ).get_numpy_array()
    import ipdb; ipdb.set_trace()
    print("done")
