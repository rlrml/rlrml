import base64
import logging
from io import BytesIO

from flask import Flask, request
from matplotlib.figure import Figure

from . import player_cache as pc
from . import filters
from .tracker_network import CloudScraperTrackerNetwork


app = Flask(__name__)
logger = logging.getLogger(__name__)


def _img_from_fig(fig: Figure):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"


def make_routes(filepath):

    cache = pc.PlayerCache.new_with_cache_directory(filepath)

    cached_get = pc.CachedGetPlayerData(
        cache, CloudScraperTrackerNetwork().get_player_data
    ).get_player_data

    @app.route("/at/<platform>/<player_name>")
    def starting_at(platform, player_name):
        return at(f"{platform}/{player_name}", double_dot=True)

    @app.route("/at/<player_key>")
    def at(player_key, double_dot=False):
        if request.args.get("fetch", default=False):
            try:
                cached_get({"__tracker_suffix__": player_key})
            except Exception as e:
                logger.warn(f"Exception when doing cached_get {e}")
                pass

        count = request.args.get("count", default=1)
        elements = []
        for (player_key, player_data) in cache.iterator(start=player_key.encode('utf-8')):
            print(f"fhecking {player_key}")
            if len(elements) >= count:
                break

            fig = Figure()

            try:
                filters.plot_mmr(player_data, player_key, fig)
            except Exception as e:
                logger.warn(f"Exception attempting to plot {player_key} {e}")
                continue

            player_info = player_data["platform"]
            elements.append(f"<div>{player_info}<br>{_img_from_fig(fig)}</div>")

        last_player_key = player_key

        dd_str = "../" if double_dot else ""
        elements.append(f'<div><a href="{dd_str}../at/{last_player_key}">next</a></div>')

        return "\n<br><br>".join(elements)

    @app.route("/")
    def _hello():
        images = []
        for i, (player_key, player_data) in enumerate(cache):
            if len(images) > 9:
                break

            try:
                player_data['mmr_history']['Ranked Doubles 2v2']
            except:
                continue

            fig = Figure()

            try:
                filters.plot_mmr(player_data, player_key, fig)
            except:
                continue

            player_info = player_data["platform"]
            images.append(f"<div>{player_info}{_img_from_fig(fig)}")

        # Embed the result in the html output.

        return "<br><br>\n".join(images)
