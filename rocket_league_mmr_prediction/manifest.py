"""Utilities for getting data out of manifest.json files produced by replay downloader."""
import os
import json

rank_tier_map = {
    21: [1726.5, 1760.5, 1809, 1842.5],
    20: [1577.5, 1612, 1647.5, 1691],
    19: [1446, 1473.5, 1515, 1545.5],
    18: [1324, 1351, 1381.5, 1411],
    17: [1204, 1225.5, 1261.5, 1291],
    16: [1084, 1105.5, 1143, 1171],
    15: [999, 1016, 1039, 1057],
    14: [919, 936, 959, 976],
    13: [839, 856, 879, 896],
    12: [776, 788, 807, 821],
    11: [716.5, 728, 746.5, 760.5],
    10: [656, 668, 687, 700],
    9: [596.5, 608, 627, 644.5],
    8: [536.5, 548.5, 567, 579],
    7: [476.5, 488.5, 507.5, 524.5],
    6: [413, 428.5, 446.5, 465.5],
    5: [353, 370, 388.5, 405.5],
    4: [297.5, 308.5, 327.5, 344.5]
}


class ManifestLoader(object):
    """Load data from manifest files."""

    def __init__(self):
        self.filepath_cache = {}

    def _get_filepath_data(self, manifest_path):
        if manifest_path not in self.filepath_cache:
            with open(manifest_path) as f:
                manifest_data = json.loads(f.read())

            self.filepath_cache[manifest_path] = manifest_data

        return self.filepath_cache[manifest_path]

    def get_raw_manifest_data_from_replay_filepath(self, replay_path):
        actual_path = os.readlink(replay_path) if os.path.islink(replay_path) else replay_path
        directory, filename = os.path.split(actual_path)
        replay_id, _ = os.path.splitext(filename)
        return self.get_raw_manifest_data(replay_id, os.path.join(directory, "manifest.json"))

    def get_raw_manifest_data(self, replay_id, manifest_filepath):
        data = self._get_filepath_data(manifest_filepath)
        return data.get(replay_id)

    def lookup_labels_by_manifest_file(self, replay_id, replay_filepath):
        directory = os.path.dirname(replay_filepath)
        manifest_filepath = os.path.join(directory, "manifest.json")

        return get_mmr_data_from_manifest_game(manifest_game), self._get_player_meta_dict(manifest_game)

    @staticmethod
    def _get_player_meta_dict(manifest_game):
        return dict((player["name"], player) for player in (
            manifest_game["orange"]["players"] + manifest_game["blue"]["players"]
        ))


def get_mmr_data_from_manifest_game(manifest_game):
    """Get an mmr number from the rank/division for each player in a manifest game."""
    return dict(
        (player["name"], get_mmr_from_manifest_player(player))
        for player in (manifest_game["orange"]["players"] + manifest_game["blue"]["players"])
    )


def _rank_tier_and_division_to_mmr(rank_tier, division):
    if rank_tier > 21:
        return 1900.0
    if rank_tier < 4:
        return 250.0

    assert division > 0 and division < 5

    return rank_tier_map[rank_tier][division - 1]


def get_mmr_from_manifest_player(player):
    """Get the an mmr number from the rank and division in the provided player meta data."""
    try:
        return _rank_tier_and_division_to_mmr(player["rank"]["tier"], player["rank"]["division"])
    except KeyError:
        pass
