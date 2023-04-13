import datetime
import numpy as np

from . import _replay_meta
from . import mmr


def scaled_sigmoid(x, base=3.7, denominator=20.0):
    return 2 * ((1.0 / (1.0 + pow(base, (-x / denominator)))) - .5)


def mean_squared_errorish_score(scores):
    return 1 - (sum([pow(1.0 - score, 2) for score in scores]) / len(scores))


def default_estimate_calculator(date, mmr_history_by_season):
    return mmr.SeasonBasedPolyFitMMRCalculator(
        mmr_history_by_season,
        season_dates=mmr.TIGHTENED_SEASON_DATES,
    ).get_mmr(date)


class MetaScoreTooLow(Exception):

    pass


class MMREstimateQualityFilter:

    def __init__(
            self, get_player_data, season_dates=mmr.TIGHTENED_SEASON_DATES,
            estimate_calculator=None, score_game_count=scaled_sigmoid,
            meta_score=np.prod
    ):
        self._get_player_data = get_player_data
        self._season_dates = season_dates
        self._estimate_caculator = estimate_calculator or default_estimate_calculator
        self._score_game_count = score_game_count
        self._meta_score = meta_score

    def score_replay_meta(self, meta: _replay_meta.ReplayMeta, abort_score=0.0):
        game_date = meta.datetime.date()
        estimates = []
        scores = []
        for player in meta.player_order:
            estimate, score = self.score_player_mmr_estimate(player, game_date)
            estimates.append((player.tracker_suffix, estimate))
            scores.append(score)
            # if estimate is None or self._meta_score(scores) < abort_score:
            #     raise MetaScoreTooLow()

        meta_score = self._meta_score(scores)
        return meta_score, estimates, scores

    def score_player_mmr_estimate(
            self, player: _replay_meta.PlatformPlayer, date: datetime.date,
            playlist='Ranked Doubles 2v2'
    ):
        player_data = self._get_player_data(player)

        if player_data is None:
            return 0, 0.0

        season_at_date = mmr.get_season_for_date(date, season_dates=self._season_dates)

        try:
            playlist_mmr_history = player_data['mmr_history'][playlist]
        except Exception:
            playlist_mmr_history = []

        history_by_season = mmr.split_mmr_history_into_seasons(
            playlist_mmr_history,
            season_dates=self._season_dates
        )

        stats = mmr.calculate_all_season_statistics(history_by_season)
        history_estimate, score = self._calculate_season_history_mmr_estimate(
            date, history_by_season, stats
        )

        all_mmrs = [mmr for _, mmr in playlist_mmr_history]

        if not all_mmrs:
            return (0.0, 0.0)

        all_time_max = max(all_mmrs)
        if history_estimate is None or score < .15:
            if len(playlist_mmr_history) > 0 and player_data['stats']['wins'] > 150:
                all_history_median = np.median([mmr for _, mmr in playlist_mmr_history])
                if all_time_max - all_history_median < 200:
                    return all_history_median, .15

        return (history_estimate, score)

    def meta_download_filter(self, replay_meta):
        try:
            score, _, _ = self.score_replay_meta(replay_meta)
            return score > 0
        except MetaScoreTooLow:
            return False

    def _calculate_season_history_mmr_estimate(self, date, history_by_season, stats):
        calculator = mmr.SeasonBasedPolyFitMMRCalculator(
            history_by_season,
            season_dates=self._season_dates,
            stats=stats,
        )
        season_at_date = mmr.get_season_for_date(date, season_dates=self._season_dates)
        season_to_data_points = dict(history_by_season)
        data_points_count = len(season_to_data_points.get(season_at_date, []))
        score = self._score_game_count(data_points_count)
        return calculator.get_mmr(date), score
