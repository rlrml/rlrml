import functools
import subprocess
import json


@functools.cache
def get_2v2_mmr_from_tracker_curl(uri):
    """Get the 2v2 mmr field from the json retrieved from the provided uri."""
    return get_2v2_mmr_from_api_data(tracker_curl(uri))


def tracker_curl(uri):
    """Curl using a subshell, setting a user agent, and loading the result as json."""
    return json.loads(
        subprocess.check_output(['curl', uri, '--user-agent', 'testing'])
    )


def get_2v2_mmr_from_api_data(player_data):
    """Extract the 2v2 mmr from data retrieved from the tracker api."""
    if 'errors' in player_data:
        return None
    try:
        for segment in player_data["data"]["segments"]:
            if segment["metadata"]["name"] == "Ranked Doubles 2v2":
                return float(segment["stats"]["rating"]["value"])
    except:
        import ipdb; ipdb.set_trace()
        print("Exception on json returned from tracekr api")


def get_mmr_history_uri_by_id(tracker_player_id):
    """Get the uri at which mmr history for the given player can be found."""
    return f"https://api.tracker.gg/api/v1/rocket-league/player-history/mmr/{tracker_player_id}"


def get_info_uri_for_player(player):
    """Get the uri that should be used to retrieve mmr info from the given player dictionary."""
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
    """Get a 2v2 mmr from the provided player meta data dictionary."""
    return get_2v2_mmr_from_api_data(tracker_curl(get_info_uri_for_player(player_meta)))


def get_relevant_info(profile_json):
    data = profile_json["data"]
    platform_info = data["platformInfo"]
    metadata = data["metadata"]
    return {
        "platform": platform_info["platformSlug"],
        "tracker_api_id": metadata["playerId"]
        
    }


def get_relevant_playlist_info(playlist):
    stats = playlist["stats"]
    return {
        "matches_played": stats["matches_played"]["value"],
        
    }
