import base64
import logging
from io import BytesIO

from flask import Flask, request, redirect
from markupsafe import escape
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
    """Build the routes to serve mmr graphs with the provided cache filepath."""
    cache = pc.PlayerCache.new_with_cache_directory(filepath)

    @app.route("/at/<platform>/<player_name>")
    def starting_at(platform, player_name):
        return at(f"{platform}/{player_name}", double_dot=True)

    @app.route("/at/<player_key>")
    def at(target_player_key, double_dot=False):
        try:
            pc.CachedGetPlayerData(
                cache, CloudScraperTrackerNetwork().get_player_data
            ).get_player_data({"__tracker_suffix__": target_player_key})
        except Exception as e:
            logger.warn(f"Exception when doing cached_get {e}")

        count = int(request.args.get("count", default=3))
        elements = []
        for (player_key, player_data) in cache.iterator(start=target_player_key.encode('utf-8')):
            if len(elements) >= count:
                # Cycle the player key until we have one that is actualy new.
                if player_key == target_player_key:
                    continue
                else:
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
        logger.warn(f"{target_player_key}, {last_player_key}")
        elements.append(
            f'<div><a href="{dd_str}../at/{escape(last_player_key)}">next</a></div>'
        )

        return "\n<br><br>".join(elements)

    @app.route("/")
    def root():
        return redirect("/at/0")
