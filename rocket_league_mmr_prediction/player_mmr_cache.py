import os
import itertools
import backoff
import plyvel
import json


def get_all_players_from_replay_directory(filepath):
    for manifest_filepath in get_manifest_files(filepath):
        for player in get_players_from_manifest_file(manifest_filepath):
            yield player


def get_players_from_manifest_file(filepath):
    with open(filepath) as f:
        manifest_data = json.loads(f.read())
    for game in manifest_data.values():
        for player in itertools.chain(game["orange"]["players"], game["blue"]["players"]):
            yield player


def get_manifest_files(filepath):
    for root, _, files in os.walk(filepath):
        for filename in files:
            if filename == "manifest.json":
                yield os.path.join(root, filename)
