"""Utilities for filtering games based on their metadata."""
import datetime


class MMRFilteringError(Exception):
    pass


class NoMMRHistory(MMRFilteringError):
    pass

class MMRMinMaxDiscrepancyTooLarge(MMRFilteringError):
    """Exception that is raised when max mmr exceeds min mmr by too much."""

    def __init__(self, min_mmr, max_mmr):
        self.min_mmr = min_mmr
        self.max_mmr = max_mmr


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


def require_mmr_within_range_of_used_mmr(
        mmr_to_use, player_data, game_data, playlist="Ranked Doubles 2v2",
        days_before=None, days_after=None, delta=120, expand_to_get_value=True
):
    """Check whether a players mmr was within an amount of a target."""
    game_date = datetime.datetime.fromisoformat(game_data["date"]).date()

    start_date = None
    if days_before is not None:
        start_date = game_date - datetime.timedelta(days=days_before)

    end_date = None
    if days_after is not None:
        end_date = game_date + datetime.timedelta(days=days_after)

    date_mmrs = mmr_between_dates(
        player_data, playlist=playlist, start_date=start_date, end_date=end_date
    )

    mmrs = [mmr for _, mmr in date_mmrs]

    max_mmr = max(mmrs)
    min_mmr = min(mmrs)

    return (max_mmr - mmr_to_use < delta) and (mmr_to_use - min_mmr < delta)


def mmr_between_dates(
        player_data, playlist="Ranked Doubles 2v2", start_date=None, end_date=None,
):
    """Get mmrs from player_data for the relevant playlist between the optional date bounds."""
    try:
        mmr_history = player_data["mmr_history"][playlist]
    except KeyError:
        raise NoMMRHistory()

    def in_bounds(the_date):
        all_good = True
        if start_date:
            all_good = all_good and start_date <= the_date
        if end_date:
            all_good = all_good and the_date <= end_date
        return all_good

    return [
        (datetime.datetime.fromisoformat(date_string).date(), mmr)
        for date_string, mmr in mmr_history
        if in_bounds(datetime.datetime.fromisoformat(date_string).date())
    ]
