"""Utilities for filtering games based on their metadata."""
import enum
import datetime
import logging
import numpy as np

from . import playlist


logger = logging.getLogger(__name__)


normal_rank_tier_ranges = [
    (float('-inf'), 167),
    (170.0, 229.0),
    (233.0, 294.0),
    (289.0, 349.0),
    (352.0, 412.0),
    (410.0, 474.0),
    (475.0, 526.0),
    (535.0, 585.0),
    (593.0, 645.0),
    (654.0, 702.0),
    (713.0, 764.0),
    (774.0, 825.0),
    (835.0, 900.0),
    (915.0, 980.0),
    (995.0, 1060.0),
    (1075.0, 1179.0),
    (1195.0, 1299.0),
    (1315.0, 1419.0),
    (1435.0, 1557.0),
    (1574.0, 1702.0),
    (1715.0, 1859.0),
    (1862.0, float('inf'))
]


solo_rank_tier_ranges = [
    (float('-inf'), 148),
    (148.0, 212.0),
    (213.0, 274.0),
    (275.0, 334.0),
    (335.0, 394.0),
    (395.0, 454.0),
    (455.0, 514.0),
    (515.0, 574.0),
    (575.0, 634.0),
    (635.0, 694.0),
    (695.0, 751.0),
    (755.0, 807.0),
    (815.0, 863.0),
    (875.0, 923.0),
    (935.0, 984.0),
    (995.0, 1044.0),
    (1055.0, 1112.0),
    (1106.0, 1173.0),
    (1175.0, 1227.0),
    (1228.0, 1294.0),
    (1286.0, 1353.0),
    (1343, float('inf'))
]


class Rank(enum.StrEnum):
    BRONZE = "Bronze"
    SILVER = "Silver"
    GOLD = "Gold"
    PLATINUM = "Platinum"
    DIAMOND = "Diamond"
    CHAMPION = "Champion"
    GRAND_CHAMPION = "Grand Champion"
    SUPERSONIC_LEGEND = "Supersonic Legend"


rank_number_to_name = dict(enumerate(list(Rank)))


class MMRToRank:

    def __init__(self, rank_tier_ranges, round_up=False):
        self._rank_tier_ranges = rank_tier_ranges
        self._round_up = round_up

    def get_rank_tier(self, mmr):
        last_upper_bound = float('-inf')
        for tier_number, (lower_bound, upper_bound) in enumerate(normal_rank_tier_ranges):
            if lower_bound <= mmr <= upper_bound:
                return tier_number
            elif last_upper_bound <= mmr <= lower_bound:
                return float(tier_number) - .5
            last_upper_bound = upper_bound
        # Shouldn't be necessary
        return tier_number

    def get_rank_name(self, mmr):
        return self.get_rank_name_and_tier(mmr)[0]

    def get_rank_name_and_tier(self, mmr, round_up=False):
        to_int_fn = np.ceil if round_up else np.floor
        tier_number = int(to_int_fn(self.get_rank_tier(mmr)))
        class_name = rank_number_to_name[int(np.floor(tier_number / 3))]
        class_tier = (tier_number % 3) + 1
        return class_name, class_tier

    def get_rank_tier_name(self, mmr):
        class_name, class_tier = self.get_rank_name_and_tier(mmr)
        if class_name == "Supersonic Legend":
            return class_name
        return f"{class_name} {int(class_tier)}"


playlist_to_converter = {
    playlist.Playlist.DUEL: MMRToRank(solo_rank_tier_ranges),
    playlist.Playlist.DOUBLES: MMRToRank(normal_rank_tier_ranges),
    playlist.Playlist.STANDARD: MMRToRank(normal_rank_tier_ranges),
}


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


TIGHTENED_SEASON_DATES = tighten_season_dates(SEASON_DATES)


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


def _calculate_basic_season_statistics(
        season_data, keep_poly=True, approx_increasing_allowance=.15,
):
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

    values['point_count'] = len(mmrs)

    if values['point_count'] < 5:
        return values

    try:
        poly = np.polynomial.Polynomial.fit(day_deltas, mmrs, 3)
    except np.linalg.LinAlgError as e:
        logger.info(f"calculate stats fitting error {e}")
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
    def get_mmr_for_player_at_date(cls, game_date, *args, **kwargs):
        """Get the for the provided player at the provided game date."""
        return cls.from_player_data(*args, **kwargs).get_mmr(game_date)

    @classmethod
    def from_player_data(cls, player_data, playlist_name='Ranked Doubles 2v2', **kwargs):
        """Extract the relevant values from player_data to initialize this class."""
        mmr_history = player_data.get('mmr_history', {}).get(playlist_name, [])
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
            stats=None
    ):
        """Init with the relevant mmr data and calculate statistics."""
        self._season_dates = season_dates
        self._mmr_history_by_season = mmr_history_by_season
        self._mmr_history_dict = dict(self._mmr_history_by_season)
        self._stats = stats or calculate_all_season_statistics(
            self._mmr_history_by_season
        )
        self._season_stats = dict(self._stats["season"])
        self._dynamic_max_poly_max_gap = dynamic_max_poly_max_gap
        self._min_max_proximity_threshold = min_max_proximity_threshold
        self._season_dp_threshold = season_dp_threshold

    def get_mmr(self, game_date):
        """Calculate mmr using the polyratic fit of a players mmr within the relevant season."""
        season_number = get_season_for_date(game_date, season_dates=self._season_dates)
        game_season_stats = self._season_stats.get(season_number)

        if game_season_stats is None:
            # TODO: try to use previous season?
            return

        # if game_season_stats['point_count'] < self._season_dp_threshold:
        #     return np.mean([game_season_stats['max'], game_season_stats['min']])

        poly_game_day = (game_date - self._mmr_history_dict[season_number][0][0].date()).days

        if 'poly' not in game_season_stats:
            return

        poly_estimate = game_season_stats['poly'](poly_game_day)
        season_poly_max = game_season_stats['poly_max']
        season_poly_min = game_season_stats['poly_min']

        previous_global_poly_max = \
            self._stats["global"]["previous_poly_maxes"].get(season_number, 0)

        for last_season_number in range(season_number - 1, -1, -1):
            last_season_stats = self._season_stats.get(last_season_number)
            if last_season_stats is not None:
                break

        last_season_stats = last_season_stats or {}

        if last_season_stats:
            previous_poly_max = last_season_stats.get('poly_max', previous_global_poly_max)
        else:
            previous_poly_max = previous_global_poly_max

        relevant_poly_max = min(previous_poly_max, season_poly_max)

        if game_season_stats["~increasing"]:
            estimate = max(relevant_poly_max, poly_estimate)
        elif season_poly_max - season_poly_min < self._min_max_proximity_threshold:
            contributors = [season_poly_max, season_poly_min]
            if previous_poly_max > season_poly_min:
                contributors.append(previous_poly_max)
            estimate = np.mean(contributors)
        else:
            estimate = poly_estimate

        max_difference = self._dynamic_max_poly_max_gap(previous_poly_max)

        if (
                last_season_stats.get('~increasing', False) and
                last_season_stats['point_count'] >= self._season_dp_threshold and
                last_season_stats['poly_finish'] > estimate
        ):
            estimate = min(last_season_stats['poly_finish'], game_season_stats['max'])

        if previous_poly_max - estimate > max_difference:
            estimate = previous_poly_max - max_difference

        last_mean = last_season_stats.get('mean', 0)
        if estimate < last_mean:
            estimate = last_mean

        min_season_finish = min(
            game_season_stats['poly_finish'], last_season_stats.get('poly_finish', 0)
        )
        if estimate < min_season_finish:
            estimate = min_season_finish

        if estimate > game_season_stats['max']:
            return game_season_stats['max']

        if estimate < game_season_stats['min']:
            return game_season_stats['min']

        if self._mmr_history_dict[season_number]:
            last_date, last_mmr = self._mmr_history_dict[season_number][-1]
            if game_date > last_date.date():
                return last_mmr

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
