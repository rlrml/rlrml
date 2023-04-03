"""Utilities for filtering games based on their metadata."""
import datetime
from matplotlib.figure import Figure
import numpy as np


SEASON_TEXT_DATES = [
    (1, ("2020-09-23", "2020-12-09")),
    (2, ("2020-12-09", "2021-04-07")),
    (3, ("2021-04-07", "2021-08-11")),
    (4, ("2021-08-11", "2021-11-17")),
    (5, ("2021-11-17", "2022-03-09")),
    (6, ("2022-03-09", "2022-06-15")),
    (7, ("2022-06-15", "2022-09-07")),
    (8, ("2022-09-07", "2022-12-07")),
    (9, ("2022-12-07", "2023-03-08")),
    (10, ("2023-03-08", "2023-06-07")),
]


SEASON_DATES = [
    (season_number, tuple(map(lambda v: datetime.date.fromisoformat(v), value)))
    for season_number, value in SEASON_TEXT_DATES
]


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


def _get_start_date_and_end_date(game_data, days_before, days_after):
    try:
        game_date = datetime.datetime.fromisoformat(game_data["date"]).date()
    except ValueError:
        game_date = datetime.date.fromisoformat(game_data["date"][:10])

    start_date = None
    if days_before is not None:
        start_date = game_date - datetime.timedelta(days=days_before)

    end_date = None
    if days_after is not None:
        end_date = game_date + datetime.timedelta(days=days_after)

    return start_date, end_date


def get_player_mmr_for_game(
        player_data, game_data, days_before=None, days_after=None, max_distance=230
):
    """Get an mmr that is appropriate for to use for the provided game."""
    start_date, end_date = _get_start_date_and_end_date(
        game_data, days_before=days_before, days_after=days_after
    )
    mmr_data = mmr_between_dates(player_data, start_date=start_date, end_date=end_date)
    mmrs = [mmr for _, mmr in mmr_data]

    if len(mmrs) == 0:
        mmr_data = mmr_between_dates(player_data)
        mmrs = [mmr for _, mmr in mmr_data]

    max_mmr = max(mmrs)
    min_mmr = min(mmrs)

    average_mmr = sum(mmrs) / len(mmrs)
    midpoint_mmr = (max_mmr + min_mmr) / 2

    if max_mmr - min_mmr > max_distance:
        raise MMRMinMaxDiscrepancyTooLarge(min_mmr, max_mmr)

    return (average_mmr + midpoint_mmr) / 2


def _minimum_total_wins_for_mmr(mmr_to_use):
    """Calculate the number of wins required to use a given mmr."""
    a = 0.002538088
    b = -2.038699
    c = 480.0544
    return min(c + b * mmr_to_use + a * pow(mmr_to_use, 2), 1500)


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
        self._season_number, self._start_date, self._end_date, self._season_started = (
            None, None, None, None
        )
        self._seasons_exhausted = False
        self._seasons_iterator = iter(self._season_dates)
        self._segmented_history = []
        self._current_segment = []

    def _increment_season(self):
        try:
            self._season_number, (self._start_date, self._end_date) = next(self._seasons_iterator)
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


def tighten_season_dates(season_dates, move_end_date=1, move_start_date=1):
    """Adjust the provided season start and end dates."""
    return [
        (season_number, (
            season_start + datetime.timedelta(days=move_start_date),
            season_end - datetime.timedelta(days=move_end_date)
        ))
        for season_number, (season_start, season_end) in season_dates
    ]


def mmr_between_dates(
        mmr_history, start_date=None, end_date=None,
):
    """Get mmrs from player_data for the relevant playlist between the optional date bounds."""

    def in_bounds(the_date):
        all_good = True
        if start_date:
            all_good = all_good and start_date < the_date
        if end_date:
            all_good = all_good and the_date < end_date
        return all_good

    return [
        (date, mmr)
        for date, mmr in mmr_history
        if in_bounds(datetime.datetime.fromisoformat(date).date())
    ]
