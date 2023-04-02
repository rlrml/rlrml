"""Utilities for filtering games based on their metadata."""
import datetime
from datetime import datetime
import matplotlib.pyplot as plt
import os
import numpy as np

#datetime.date.isoformat for comparing -> maps
SEASON_DATES = {
    1:  ("2020-09-23", "2020-12-09"),
    2:  ("2020-12-09", "2021-04-07"),
    3:  ("2021-04-07", "2021-8-11"),
    4:  ("2021-08-11", "2021-11-17"),
    5:  ("2021-11-17", "2022-03-09"),
    6:  ("2022-03-09", "2022-06-15"),
    7:  ("2022-06-15", "2022-09-07"),
    8:  ("2022-09-07", "2022-12-07"),
    9:  ("2022-12-07", "2023-03-08"),
    10: ("2023-03-08", "2023-06-07"),
}


class MMRFilteringError(Exception):
    """An error relating to the filtering of games and players to do with MMR."""

    pass

class NotInTrackerNetwork(Exception):
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

'''
player_data = {
    platform,
    tracker_api_id,
    last_updated,
    stats,
    playlists,
    mmr_history,
    player_metadata
}
player_data['mmr_history'] = {
    Ranked Duel 1v1,
    Ranked Doubles 2v2,
    Ranked Standard 3v3
}
'''
def test_mmr_no_plot(player_data, player_key=None, seasons_avg=[8,9]):
    if not isinstance(player_data, dict):
        raise NotInTrackerNetwork

    try:
        mmr_history = player_data['mmr_history']['Ranked Doubles 2v2']
    except:
        raise NoMMRHistory()

    mmrs = np.array([mmr for _, mmr in mmr_history])
    dates = np.array([datetime.strptime(date.split('T')[0], '%Y-%m-%d') for date, _ in mmr_history])
    mmrs = mmrs[np.argsort(dates)]
    dates = np.sort(dates)

    seasons = np.array([datetime.strptime(date[0], '%Y-%m-%d') for date in SEASON_DATES.values()])
    seasons_avg -= 1
    seasons = [seasons_avg]
    seasons = seasons[(seasons > min(dates)) & (seasons < max(dates))]

    mmr_by_season = []
    #mmr_s_dates = []
    for i in range(len(seasons)-1):
        idx1 = np.where(dates > seasons[i])[0][0]
        idx2 = np.where(dates > seasons[i+1])[0][0]
        mmr_by_season.append(player_mmr_function(mmrs[idx1:idx2]))
        #mmr_s_dates.append(dates[idx1:idx2])
    mmr_by_season.append(player_mmr_function(mmrs[np.where(dates > seasons[-1])[0][0]:]))
    #mmr_s_dates.append(dates[np.where(dates > seasons[-1])[0][0]:])

    #mmr1, slope = player_mmr_function(mmr_by_season[-1])
    #mmr2, slope = player_mmr_function(mmr_by_season[-2])

    return np.mean(mmr_by_season)

def test_mmr(player_data, player_key):
    if not isinstance(player_data, dict):
        raise NotInTrackerNetwork

    try:
        mmr_history = player_data['mmr_history']['Ranked Doubles 2v2']
    except:
        raise NoMMRHistory()

    mmrs = np.array([mmr for _, mmr in mmr_history])
    dates = np.array([datetime.strptime(date.split('T')[0], '%Y-%m-%d') for date, _ in mmr_history])
    mmrs = mmrs[np.argsort(dates)]
    dates = np.sort(dates)

    seasons = np.array([datetime.strptime(date[0], '%Y-%m-%d') for date in SEASON_DATES.values()])
    seasons = seasons[(seasons > min(dates)) & (seasons < max(dates))]

    #mmr = player_mmr_function(mmrs, seasons)

    mmr_by_season = []
    mmr_s_dates = []
    for i in range(len(seasons)-1):
        idx1 = np.where(dates > seasons[i])[0][0]
        idx2 = np.where(dates > seasons[i+1])[0][0]
        mmr_by_season.append(mmrs[idx1:idx2])
        mmr_s_dates.append(dates[idx1:idx2])
    mmr_by_season.append(mmrs[np.where(dates > seasons[-1])[0][0]:])
    mmr_s_dates.append(dates[np.where(dates > seasons[-1])[0][0]:])

    plt.clf()
    for i in range(len(mmr_by_season)):
        plt.plot(mmr_s_dates[i], mmr_by_season[i], color='blue')
        mmr, slope = player_mmr_function(mmr_by_season[i])
        plt.plot([mmr_s_dates[i][0], mmr_s_dates[i][-1]], [mmr_by_season[i][0], mmr_by_season[i][0] + slope], color='orange')
        plt.hlines(y=np.mean(mmr_by_season[i]), xmin=min(mmr_s_dates[i]), xmax=max(mmr_s_dates[i]), color='red')
        plt.hlines(y=mmr, xmin=min(mmr_s_dates[i]), xmax=max(mmr_s_dates[i]), color='green')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.vlines(seasons, ymin=min(mmrs), ymax=max(mmrs), colors='red')
    plt.title(player_key)
    plt.savefig('./plots/' + str(player_key) + '.png', bbox_inches='tight')

def player_mmr_function(mmrs):
    slope = mmrs[-1] - mmrs[0]
    #for i in range(1, len(mmrs)):
    #    slope += mmrs[i] - mmrs[i-1]
    ratio = (slope - np.std(mmrs)) / (max(mmrs) - min(mmrs))

    return np.mean(mmrs) + (abs(ratio) * (mmrs[-1] - mmrs[0]))/2, slope

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
