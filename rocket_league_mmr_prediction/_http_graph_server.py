import base64
from io import BytesIO

from flask import Flask
from matplotlib.figure import Figure

from . import player_cache as pc
from . import filters


app = Flask(__name__)


def make_routes(filepath):

    cache = pc.PlayerCache.new_with_cache_directory(filepath)

    @app.route("/")
    def _hello():
        fig = Figure()

        ax = fig.subplots()

        player_key, player_data = next(iter(cache))
        filters.plot_mmr(player_data, player_key, ax)

        # Save it to a temporary buffer.
        buf = BytesIO()
        fig.savefig(buf, format="png")

        # Embed the result in the html output.
        data = base64.b64encode(buf.getbuffer()).decode("ascii")

        return f"<img src='data:image/png;base64,{data}'/>"
