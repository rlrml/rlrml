"""Utilities for filtering games based on their metadata."""
import datetime
import numpy as np


logger = logging.getLogger(__name__)


SEASON_TEXT_DATES = enumerate([
    ("2020-09-23", "2020-12-09"),
    ("2020-12-09", "2021-04-07"),
    ("2021-04-07", "2021-08-11"),
    ("2021-08-11", "2021-11-17"),
    ("2021-11-17", "2022-03-09"),
    ("2022-03-09", "2022-06-15"),
    ("2022-06-15", "2022-09-07"),
    ("2022-09-07", "2022-12-07"),
    ("2022-12-07", "2023-03-08"),
    ("2023-03-08", "2023-06-07"),
], start=1)


SEASON_DATES = [
    (season_number, tuple(map(lambda v: datetime.date.fromisoformat(v), value)))
    for season_number, value in SEASON_TEXT_DATES
]


def tighten_season_dates(season_dates, move_end_date=1, move_start_date=1):
    """Adjust the provided season start and end dates."""
    return [
        (season_number, (
            season_start + datetime.timedelta(days=move_start_date),
            season_end - datetime.timedelta(days=move_end_date)
        ))
        for season_number, (season_start, season_end) in season_dates
    ]


def get_season_for_date(date, season_dates=SEASON_DATES, strict=False):
    """Get the season that the given date occured in."""
    for season_number, (season_start, season_end) in season_dates:
        if date <= season_end and (not strict or season_start <= date):
            return season_number

    return season_number


def get_game_date(game_data):
    """Get a python date corresponding to the date where the game was played."""
    try:
        return datetime.datetime.fromisoformat(game_data["date"]).date()
    except ValueError:
        try:
            return datetime.date.fromisoformat(game_data["date"][:10])
        except Exception:
            return


class _MMRHistorySplitter:

    @classmethod
    def from_tracker_data(cls, mmr_history, **kwargs):
        mmr_history = [
            (datetime.datetime.fromisoformat(date_string), mmr)
            for date_string, mmr in mmr_history
        ]
        mmr_history.sort(key=lambda v: v[0])
        return cls(mmr_history, **kwargs)

    def __init__(self, mmr_history, season_dates=SEASON_DATES):
        self._mmr_history = mmr_history
        self._season_dates = season_dates
        (
            self._season_number, self._start_date,
            self._end_date, self._season_started
        ) = (
            None, None, None, None
        )
        self._seasons_exhausted = False
        self._seasons_iterator = iter(self._season_dates)
        self._segmented_history = []
        self._current_segment = []

    def _increment_season(self):
        try:
            self._season_number, (
                self._start_date,
                self._end_date
            ) = next(self._seasons_iterator)
            self._season_started = False
        except StopIteration:
            self._seasons_exhausted = True

    def _finish_segment(self):
        segment_number = (
            self._season_number + .5 if self._seasons_exhausted
            else self._season_number if self._season_started
            else self._season_number - .5
        )
        if self._current_segment:
            self._segmented_history.append((segment_number, self._current_segment))
            self._current_segment = []

    def _handle_item(self, item):
        date, mmr = item
        date = date.date()
        after_start = self._start_date <= date
        before_end = date <= self._end_date
        in_bounds = after_start and before_end

        if self._seasons_exhausted or (
            self._season_started and in_bounds
        ) or (
            not after_start and not self._season_started
        ):
            return self._current_segment.append(item)

        if in_bounds:
            self._finish_segment()
            self._season_started = True
        elif not before_end:
            self._finish_segment()
            self._increment_season()

        return self._handle_item(item)

    def get_history(self):
        self._increment_season()
        for item in self._mmr_history:
            self._handle_item(item)

        # This final finish segment handles either making the last season (if
        # nothing was ever out of bounds of the last season), or adds a final
        # season after the last bound.
        self._finish_segment()

        return self._segmented_history


def split_mmr_history_into_seasons(mmr_history, season_dates=SEASON_DATES):
    """Split the given tracker network MMR history into per season history."""
    return _MMRHistorySplitter.from_tracker_data(
        mmr_history, season_dates=season_dates
    ).get_history()


def _calculate_basic_season_statistics(season_data, keep_poly=True, approx_increasing_allowance=.15):
    mmrs = [mmr for _, mmr in season_data]
    dates = [d for d, _ in season_data]

    values = {}
    values['max'] = max(mmrs)
    values['min'] = min(mmrs)
    values['start'] = mmrs[0]
    values['end'] = mmrs[-1]
    values['mean'] = np.mean(mmrs)

    first_date = dates[0]
    day_deltas = [(date - first_date).days for date in dates]

    try:
        poly = np.polynomial.Polynomial.fit(day_deltas, mmrs, 2)
    except np.linalg.LinAlgError:
        pass
    else:
        roots = poly.deriv().roots()
        relevant_roots = [
            root
            for root in roots
            if day_deltas[0] < root < day_deltas[-1]
        ]

        potential_mins_and_maxes = [day_deltas[0], day_deltas[-1]] + relevant_roots

        points = [(poly(x), x) for x in potential_mins_and_maxes]

        values['poly_finish'] = poly(day_deltas[-1])
        values['poly_start'] = poly(day_deltas[0])

        values['poly_max'], maximizer = max(points, key=lambda t: t[0])
        values['poly_min'], minimizer = min(points, key=lambda t: t[0])
        values['poly_increase'] = values['poly_max'] - values['poly_min']
        values['increasing'] = bool(minimizer == day_deltas[0] and maximizer == day_deltas[-1])
        values['decreasing'] = bool(minimizer == day_deltas[-1] and maximizer == day_deltas[0])
        raw_allowance = approx_increasing_allowance * values['poly_increase']
        values['~increasing'] = bool(
            (values['poly_max'] - raw_allowance) <= values['poly_finish'] and
            (values['poly_min'] + raw_allowance) >= values['poly_start']
        )
        values['poly_maximizer'] = maximizer
        values['poly_minimizer'] = minimizer
        if keep_poly:
            values['poly'] = poly

    values['point_count'] = len(mmrs)

    return values


def calculate_all_season_statistics(mmr_history_by_season, keep_poly=True):
    """Calculate statistics for each season and some global statistics from seasonal mmr history."""
    season_statistics = [
        (int(season_number), _calculate_basic_season_statistics(
            season_data, keep_poly=keep_poly
        ))
        for season_number, season_data in mmr_history_by_season
        if float(season_number).is_integer()
    ]

    previous_poly_max = 0
    previous_poly_maxes = {}
    for season_number, season_stats in season_statistics:
        previous_poly_maxes[season_number] = previous_poly_max
        if season_stats.get('point_count', 0) > 10:
            this_poly_max = season_stats.get('poly_max', 0)
            if this_poly_max > previous_poly_max:
                previous_poly_max = this_poly_max

    global_stats = {
        "global_poly_max": previous_poly_max,
        "previous_poly_maxes": previous_poly_maxes,
    }

    return {
        "season": season_statistics,
        "global": global_stats,
    }


class SeasonBasedPolyFitMMRCalculator:
    """Calculate mmr using the polyratic fit of a players mmr within the relevant season."""

    @classmethod
    def from_player_data(cls, player_data, playlist_name='Ranked Doubles 2v2', **kwargs):
        """Extract the relevant values from player_data to initialize this class."""
        mmr_history = player_data['mmr_history'][playlist_name]
        season_dates = kwargs.setdefault(
            'season_dates', tighten_season_dates(SEASON_DATES)
        )
        mmr_history_by_season = split_mmr_history_into_seasons(
            mmr_history, season_dates=season_dates
        )
        return cls(mmr_history_by_season, **kwargs)

    def __init__(
            self, mmr_history_by_season, season_dates=SEASON_DATES, season_dp_threshold=20,
            dynamic_max_poly_max_gap=lambda _: 125, min_max_proximity_threshold=75,
    ):
        """Init with the relevant mmr data and calculate statistics."""
        self._season_dates = season_dates
        self._mmr_history_by_season = mmr_history_by_season
        self._mmr_history_dict = dict(self._mmr_history_by_season)
        self._stats = calculate_all_season_statistics(
            self._mmr_history_by_season
        )
        self._season_stats = dict(self._stats["season"])
        self._dynamic_max_poly_max_gap = dynamic_max_poly_max_gap
        self._min_max_proximity_threshold = min_max_proximity_threshold

    def get_mmr(self, game_date):
        """Calculate mmr using the polyratic fit of a players mmr within the relevant season."""
        season_number = get_season_for_date(game_date)
        game_season_stats = self._season_stats.get(season_number)

        if game_season_stats is None:
            # TODO: try to use previous season?
            return

        poly_game_day = (game_date - self._mmr_history_dict[season_number][0]).date()
        poly_estimate = game_season_stats['poly'](poly_game_day)
        season_poly_max = game_season_stats['poly_max']
        season_poly_min = game_season_stats['poly_min']
        season_poly_finish = game_season_stats['poly_finish']

        previous_global_poly_max = \
            self._stats["global"]["previous_poly_maxes"].get(season_number, 0)

        for last_season_number in range(season_number - 1, -1, -1):
            last_season_stats = self._season_stats.get(last_season_number)
            if last_season_stats is not None:
                break

        previous_poly_max = last_season_stats.get('poly_max', previous_global_poly_max)

        relevant_poly_max = min(previous_poly_max, season_poly_max)

        if game_season_stats["~increasing"]:
            estimate = max(relevant_poly_max, poly_estimate)
        elif season_poly_max - season_poly_min < self._min_max_proximity_threshold:
            estimate = np.mean([season_poly_max, season_poly_min, previous_poly_max])
        elif poly_game_day >= game_season_stats['poly_maximizer']:
            estimate = np.mean([season_poly_max, season_poly_finish])
        else:
            estimate = poly_estimate

        max_difference = self._dynamic_max_poly_max_gap(previous_poly_max)

        if previous_poly_max - estimate > max_difference:
            logger.warn("Large poly max to estimate difference")
            return previous_poly_max - max_difference

        return estimate

    __call__ = get_mmr



class MMRFilteringError(Exception):
    """An error relating to the filtering of games and players to do with MMR."""

    pass


class NoMMRHistory(MMRFilteringError):
    """No MMR History was found for the player."""

    pass


class MMRMinMaxDiscrepancyTooLarge(MMRFilteringError):
    """Exception that is raised when max mmr exceeds min mmr by too much."""

    def __init__(self, min_mmr, max_mmr):
        """Initialize and record the minimum and maximum MMR."""
        self.min_mmr = min_mmr
        self.max_mmr = max_mmr


def kelly_mmr_function(mmrs):
    """Calculate MMR by looking at the season slope and the standard deviation."""
    slope = mmrs[-1] - mmrs[0]
    # for i in range(1, len(mmrs)):
    #    slope += mmrs[i] - mmrs[i-1]
    ratio = (slope - np.std(mmrs)) / (max(mmrs) - min(mmrs))

    return np.mean(mmrs) + (abs(ratio) * (mmrs[-1] - mmrs[0])) / 2


def _minimum_total_wins_for_mmr(mmr_to_use):
    """Calculate the number of wins required to use a given mmr."""
    a = 0.002538088
    b = -2.038699
    c = 480.0544
    return min(c + b * mmr_to_use + a * pow(mmr_to_use, 2), 1500)
