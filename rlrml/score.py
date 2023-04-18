import collections
import datetime
import itertools
import logging
import numpy as np

from . import _replay_meta
from . import mmr
from . import util
from .playlist import Playlist


logger = logging.getLogger(__name__)


MetaScoreInfo = collections.namedtuple("MetaScoreInfo", "meta_score estimates scores")


def scaled_sigmoid(x, base=3.7, denominator=20.0):
    return 2 * ((1.0 / (1.0 + pow(base, (-x / denominator)))) - .5)


class MMREstimateScorer:

    def __init__(
            self, get_player_data, season_dates=mmr.TIGHTENED_SEASON_DATES,
            score_game_count=scaled_sigmoid, meta_score=np.prod,
            minimum_games_for_mmr=lambda mmr: 0
    ):
        self._get_player_data = get_player_data
        self._season_dates = season_dates
        self._score_game_count = score_game_count
        self._meta_score = meta_score
        self._minimum_games_for_mmr = minimum_games_for_mmr

    def score_replay_meta(
            self, meta: _replay_meta.ReplayMeta, abort_score=0.0,
            playlist=Playlist('Ranked Doubles 2v2')
    ):
        game_date = meta.datetime.date()
        estimates = []
        scores = []
        for player in meta.player_order:
            estimate, score = self.score_player_mmr_estimate(player, game_date, playlist=playlist)
            estimates.append((player, estimate))
            scores.append(score)
            meta_score = self._meta_score(scores)
            # if estimate is None or self._meta_score(scores) <= abort_score:
            #     return meta_score, estimates, scores

        meta_score = self._meta_score(scores)
        return MetaScoreInfo(meta_score, estimates, scores)

    def score_player_mmr_estimate(
            self, player: _replay_meta.PlatformPlayer, date: datetime.date,
            playlist=Playlist('Ranked Doubles 2v2')
    ):
        player_data = self._get_player_data(player)

        if player_data is None or '__error__' in player_data:
            return 0, 0.0

        try:
            playlist_mmr_history = player_data['mmr_history'][playlist]
        except Exception:
            playlist_mmr_history = []

        if not playlist_mmr_history:
            logger.warning(f"{player} had no mmr history")
            return (0.0, 0.0)

        history_by_season = mmr.split_mmr_history_into_seasons(
            playlist_mmr_history,
            season_dates=self._season_dates
        )

        stats = mmr.calculate_all_season_statistics(history_by_season)
        history_estimate, score = self._calculate_season_history_mmr_estimate(
            date, history_by_season, stats
        )

        if history_estimate and history_estimate > 0:
            return history_estimate, score

        all_pairs = itertools.chain(*[data for _, data in history_by_season])
        # This is the mmr at the closest date that we have to the game date.
        _, closest_value = util.closest_date_value(
            all_pairs, date
        )

        all_mmrs = [mmr for _, mmr in playlist_mmr_history]

        if history_estimate is None or score < .15:
            if len(playlist_mmr_history) > 0:
                all_history_median = np.median(all_mmrs)
                estimate = max(all_history_median, closest_value or 0)
                # Make sure that the player has at least a reasonable number of
                # games before using this estimate.
                if player_data['stats']['wins'] > self._minimum_games_for_mmr(estimate):
                    return estimate, .15
                else:
                    logger.warning(f"Skipping {player} because they don't have enough wins")

        return (None, 0.0)

    def meta_download_filter(self, replay_meta, remove_below=0.0):
        score, _, _ = self.score_replay_meta(replay_meta)
        return score > remove_below

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
