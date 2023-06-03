import collections
import datetime
import itertools
import logging
import numpy as np

from . import player_cache as pc
from . import metadata
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
            minimum_games_for_mmr=lambda mmr: 0,
            mmr_disparity_requires_victory_threshold=200,
            truncate_lowest_count=0,
    ):
        self._get_player_data = get_player_data
        self._season_dates = season_dates
        self._score_game_count = score_game_count
        self._meta_score = meta_score
        self._minimum_games_for_mmr = minimum_games_for_mmr
        self._mmr_disparity_requries_victory_threshold = (
            mmr_disparity_requires_victory_threshold
        )
        self._truncate_lowest_count = truncate_lowest_count

    def score_replay_meta(
            self, meta: metadata.ReplayMeta, abort_score=0.0,
            playlist=Playlist('Ranked Doubles 2v2')
    ):
        game_date = meta.datetime.date()
        estimates = []
        scores = []

        team_to_mmr_total = {
            0: 0,
            1: 0,
        }

        for player in meta.player_order:
            estimate, score = self.score_player_mmr_estimate(
                player, game_date, playlist=playlist
            )
            estimates.append((player, estimate))
            scores.append(score)

        player_lookup = dict(((player.tracker_suffix, e) for player, e in estimates))
        mean_mmr = np.mean([e for _, e in estimates if e is not None])
        for team_index, team in enumerate((meta.team_zero, meta.team_one)):
            for player in team:
                team_to_mmr_total[team_index] += player_lookup[player.tracker_suffix] or mean_mmr

        meta_scores = sorted(scores)[:-self._truncate_lowest_count]
        meta_score = self._meta_score(meta_scores)

        team_zero_mmr_advantage = team_to_mmr_total[0] - team_to_mmr_total[1]
        if abs(team_zero_mmr_advantage) >= (
                self._mmr_disparity_requries_victory_threshold * len(meta.team_zero)
        ):
            # Ensure that the team that won was the team with the higher mmr
            if "Team0Score" in meta.headers and "Team1Score" in meta.headers:
                team_zero_score_advantage = (
                    meta.headers["Team0Score"] - meta.headers["Team1Score"]
                )
                if np.sign(team_zero_score_advantage) != np.sign(team_zero_mmr_advantage):
                    meta_score = 0

        return MetaScoreInfo(meta_score, estimates, scores)

    def score_player_mmr_estimate(
            self, player: metadata.PlatformPlayer, date: datetime.date,
            playlist=Playlist('Ranked Doubles 2v2')
    ):
        player_data = self._get_player_data(player)

        if player_data is None or '__error__' in player_data:
            return None, 0.0

        if pc.PlayerCache.manual_override_key in player_data:
            override_value = player_data[pc.PlayerCache.manual_override_key]
            if not override_value:
                return None, 0.0
            return float(override_value), 1.0

        try:
            playlist_mmr_history = player_data['mmr_history'][playlist]
        except Exception:
            playlist_mmr_history = []

        if not playlist_mmr_history:
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
