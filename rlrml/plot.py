import math
import datetime
import itertools
import numpy as np

from matplotlib.figure import Figure
from . import mmr


def _skip_in_between_seasons(fn):
    def f(self, season, dates, mmrs):
        if int(season) != season:
            return
        return fn(self, season, dates, mmrs)
    return f


def make_plot_poly_fit(n):
    """Make a function that plot the polynomial of degree n."""
    @_skip_in_between_seasons
    def plot_poly_fit(self, season, dates, mmrs):
        # Only plot for actual seasons, no in betweens

        first_date = dates[0]
        last_date = dates[-1]
        total_days = (last_date - first_date).days
        day_deltas = [(date - first_date).days for date in dates]

        try:
            poly = np.polynomial.Polynomial.fit(day_deltas, mmrs, n)
        except np.linalg.LinAlgError:
            return

        x, y = poly.linspace(total_days + 1)

        x_days = [first_date + datetime.timedelta(days=xv) for xv in x]

        self._plt.plot(x_days, y, color="red")

    return plot_poly_fit


@_skip_in_between_seasons
def kelly_approach(self, season, dates, mmrs):
    """Plot the lines associated with the kelly mmr function."""
    self._plt.hlines(
        y=np.mean(mmrs), xmin=dates[0], xmax=dates[-1], color='red'
    )

    self._plt.plot([dates[0], dates[-1]], [mmrs[0], mmrs[-1]])

    self._plt.hlines(
        y=mmr.kelly_mmr_function(mmrs), color='yellow',
        xmin=dates[0], xmax=dates[-1],
    )


def make_calc_plot(calc):
    """Make an additional_plotter for the given mmr calculator."""
    @_skip_in_between_seasons
    def plot_calc_for_season(self, season, dates, mmrs):
        if season > 9:
            return
        start_date, end_date = self._season_dates_dict[season]

        total_days = (end_date - start_date).days

        x = []
        y = []
        for days in range(total_days + 1):
            plot_date = start_date + datetime.timedelta(days=days)
            mmr = calc(plot_date)

            if mmr is not None and mmr != 0:
                x.append(plot_date)
                y.append(mmr)

        if len(y) > 5:
            self._plt.plot(x, y, color='green')

    return plot_calc_for_season


class MMRHistoryPlotGenerator:
    """Generate MMR plots from tracker network data."""

    @classmethod
    def from_player_data(cls, player_data, **kwargs):
        """Build a plot generator with some sensible defaults with player_data."""
        mmr_history = player_data['mmr_history']['Ranked Doubles 2v2']
        season_dates = kwargs.setdefault(
            'season_dates', mmr.tighten_season_dates(
                mmr.SEASON_DATES, move_end_date=2
            )
        )
        mmr_by_season = mmr.split_mmr_history_into_seasons(
            mmr_history, season_dates=season_dates
        )
        return cls(mmr_by_season, **kwargs)

    def __init__(
            self, mmr_by_season, figure=None,
            playlist_name='Ranked Doubles 2v2',
            season_dates=mmr.SEASON_DATES,
            mmr_colors=("blue",), additional_plotters=(
                make_plot_poly_fit(2),
            )
    ):
        """Init the plot generator with player data."""
        self._figure = figure or Figure()
        self._plt = self._figure.subplots()
        self._playlist_name = playlist_name
        self._mmr_by_season = mmr_by_season
        self._all_mmrs = list(
            mmr for _, mmr in
            itertools.chain(*[data for _, data in self._mmr_by_season])
        )
        self._ymin = min(self._all_mmrs)
        self._ymax = max(self._all_mmrs)
        self._season_dates_dict = dict(season_dates)
        self._mmr_colors = mmr_colors
        self._additional_plotters = additional_plotters

    def _plot_season_lines(self):
        min_season = math.ceil(
            min([season_number for season_number, _ in self._mmr_by_season])
        )
        max_season = math.floor(
            max([season_number for season_number, _ in self._mmr_by_season])
        )

        season_starts = []
        season_ends = []
        for season in range(min_season, max_season + 1):
            season_start, season_end = self._season_dates_dict[season]
            season_starts.append(season_start)
            season_ends.append(season_end)

        self._plt.vlines(
            season_starts, colors='blue', ymin=self._ymin, ymax=self._ymax
        )
        self._plt.vlines(
            season_ends, colors='red', ymin=self._ymin, ymax=self._ymax
        )

    def _plot_mmr(self):
        last_mmr, last_date = None, None
        for c, (season, season_mmr) in enumerate(self._mmr_by_season):
            mmrs = [mmr for _, mmr in season_mmr]
            dates = [d for d, _ in season_mmr]

            if last_mmr is not None:
                self._plt.plot([last_date, dates[0]], [last_mmr, mmrs[0]], color='red')

            self._plt.plot(dates, mmrs, color=self._mmr_colors[c % len(self._mmr_colors)])

            for plotter in self._additional_plotters:
                plotter(self, season, dates, mmrs)

            last_mmr, last_date = mmrs[-1], dates[-1]

    def _finalize(self):
        self._plt.set_xticklabels(self._plt.get_xticklabels(), rotation=45)
        self._figure.tight_layout()

    def generate(self):
        """Generate the MMR graph."""
        self._plot_season_lines()
        self._plot_mmr()
        self._finalize()
        return self._figure


class GameMMRPredictionPlotGenerator:
    def __init__(
            self, prediction_history, players_with_mmr, predictions=(), figure=None,
            player_colors=('orange', 'blue')
    ):
        self._figure = figure or Figure(figsize=(10, 6), dpi=200)
        self._prediction_history = np.array(prediction_history)
        self._players_with_mmr = players_with_mmr
        self._plt = self._figure.subplots()
        self._player_colors = player_colors
        self._predictions = predictions or itertools.cycle([None])

    def generate(self):
        for index, (player_color, (player, player_mmr), prediction) in enumerate(zip(
                itertools.cycle(self._player_colors),
                self._players_with_mmr,
                self._predictions,
        )):
            self._plt.plot(self._prediction_history[:, index], color=player_color)
            self._plt.hlines(player_mmr, 0, len(self._prediction_history), color=player_color)
            self._figure.tight_layout()
            if prediction is not None:
                self._plt.hlines(
                    prediction, 0, len(self._prediction_history),
                    color=player_color, linestyles='dashed'
                )
        return self._figure
