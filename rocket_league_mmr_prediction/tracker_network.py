import functools
import subprocess
import json


@functools.cache
def tracker_curl_for_mmr(uri):
    get_2v2_mmr_from_api_data(tracker_curl(uri))


def tracker_curl(uri):
    return json.loads(
        subprocess.check_output(['curl', uri, '--user-agent', 'testing'])
    )


def get_2v2_mmr_from_api_data(player_data):
    for segment in player_data["data"]["segments"]:
        if segment["metadata"]["name"] == "Ranked Doubles 2v2":
            return float(segment["stats"]["rating"]["value"])


def get_info_uri_for_player(player):
    platform = player["id"]["platform"]
    player_name = player["name"]
    space_replaced = player_name.replace(" ", "%20")
    base = "https://api.tracker.gg/api/v2/rocket-league/standard/profile/"
    if platform == "epic":
        return f"{base}epic/{space_replaced}"
    elif platform == "steam":
        return f"{base}steam/{player['id']['id']}"
    elif platform == "xbox":
        return f"{base}xbl/{space_replaced}"
    elif platform.startswith("p"):
        return f"{base}psn/{space_replaced}"


def get_doubles_mmr_from_player_meta(player_meta):
    return get_2v2_mmr_from_api_data(tracker_curl(get_info_uri_for_player(player_meta)))


