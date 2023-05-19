import base64
import logging
import torch

from io import BytesIO
from flask import Flask, request, redirect
from markupsafe import escape
from matplotlib.figure import Figure

from . import mmr
from . import player_cache as pc
from . import plot
from .tracker_network import CloudScraperTrackerNetwork
from . import _replay_meta


app = Flask(__name__)
logger = logging.getLogger(__name__)


def _img_from_fig(fig: Figure):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"


def make_routes(builder):
    """Build the routes to serve mmr graphs with the provided cache filepath."""
    cache = builder.player_cache

    @app.route("/predict/<uuid>")
    def prediction_graph(uuid):
        meta, ndarray = builder.load_game_from_filepath(
            builder.get_game_filepath_by_uuid(uuid)
        )
        builder.model.eval()
        x = torch.stack([torch.tensor(ndarray)]).to(builder.device)
        history = builder.model.prediction_history(x)
        python_history = [
            [builder.label_scaler.unscale(float(prediction)) for prediction in elem[0]]
            for elem in history
        ]
        predictions = [builder.label_scaler.unscale(float(p)) for p in builder.model(x)[0]]
        meta = _replay_meta.ReplayMeta.from_boxcar_frames_meta(meta['replay_meta'])
        figure = plot.GameMMRPredictionPlotGenerator(
            python_history,
            [
                (player, builder.lookup_label(player, meta.datetime.date()))
                for player in meta.player_order
            ],
            predictions
        ).generate()
        elements = []
        elements.append(f"<div>{_img_from_fig(figure)}</div>")
        return "\n<br><br>".join(elements)

    @app.route("/at/<platform>/<player_name>")
    def starting_at(platform, player_name):
        return at(f"{platform}/{player_name}", double_dot=True)

    @app.route("/at/<target_player_key>")
    def at(target_player_key, double_dot=False):
        try:
            pc.CachedGetPlayerData(
                cache, CloudScraperTrackerNetwork().get_player_data
            ).get_player_data({"__tracker_suffix__": target_player_key})
        except Exception as e:
            logger.warn(f"Exception when doing cached_get {e}")
            raise e

        count = int(request.args.get("count", default=3))
        elements = []
        for (player_key, player_data) in cache.iterator(start_key=target_player_key.encode('utf-8')):
            if len(elements) >= count:
                # Cycle the player key until we have one that is actualy new.
                if player_key == target_player_key:
                    continue
                else:
                    break

            try:
                mmr_history = player_data['mmr_history']['Ranked Doubles 2v2']
                season_dates = mmr.tighten_season_dates(
                    mmr.SEASON_DATES, move_end_date=2
                )
                mmr_by_season = mmr.split_mmr_history_into_seasons(
                    mmr_history, season_dates=season_dates
                )
                calc = mmr.SeasonBasedPolyFitMMRCalculator(
                    mmr_by_season, season_dates=season_dates,
                )
            except KeyError:
                continue
            else:
                fig = plot.MMRHistoryPlotGenerator(
                    mmr_by_season, additional_plotters=(
                        # plot.make_plot_poly_fit(2),
                        plot.make_calc_plot(calc),
                    )
                ).generate()

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
