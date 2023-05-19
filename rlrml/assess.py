import asyncio
import multiprocessing
import numpy as np
import logging

from concurrent.futures import ThreadPoolExecutor
from .playlist import Playlist
from . import load
from . import mmr


logger = logging.getLogger(__name__)


def filter_meta_score_info_below(at_or_below):
    def filter_by_score(status):
        return status.ready and status.score_info.meta_score > at_or_below
    return filter_by_score


class ReplaySetAssesor:

    class ReplayStatus:
        ready = None

    class ScoreInfoStatus(ReplayStatus):

        def __init__(self, score_info):
            self.score_info = score_info

        @property
        def ready(self):
            return not any(mmr is None or mmr == 0 for _, mmr in self.score_info.estimates)

    class FailedStatus(ReplayStatus, Exception):
        ready = False

        def __init__(self, exception):
            self.exception = exception

    class TensorFail(FailedStatus):
        pass

    class MetaFail(FailedStatus):
        pass

    class PlaylistFail(FailedStatus):
        pass

    def __init__(
            self, replay_set: load.ReplaySet, scorer, playlist=Playlist.DOUBLES,
            ignore_known_errors=True, always_load_tensor=False, ipdb_on_exception=False,
    ):
        self._replay_set = replay_set
        self._scorer = scorer
        self._playlist = Playlist(playlist)
        self._ignore_known_errors = ignore_known_errors
        self._always_load_tensor = always_load_tensor
        self._ipdb_on_exception = ipdb_on_exception

    def get_replay_statuses(self):
        return {
            uuid: self._get_replay_status(uuid)
            for uuid in self._replay_set.get_replay_uuids()
        }

    def get_replay_statuses_by_rank(self):
        replay_statuses = self.get_replay_statuses()
        results = {"Failed": {}}
        for rank in mmr.rank_number_to_name.values():
            results[rank] = {}
        for uuid, status in replay_statuses.items():
            if status.ready:
                mmrs = [
                    mmr for _, mmr in status.score_info.estimates
                ]
                rank = mmr.playlist_to_converter[self._playlist].get_rank_name(
                    np.mean(mmrs)
                )
                results[rank][uuid] = status
            else:
                results["Failed"][uuid] = status
        return results

    def get_top_scoring_n_replay_per_rank(
            self, count_per_rank, filter_function=filter_meta_score_info_below(0)
    ):
        replay_statuses = self.get_replay_statuses_by_rank()
        top_replays = {}
        for rank, uuid_to_status in replay_statuses.items():
            if not isinstance(rank, mmr.Rank):
                continue
            replay_pairs = [
                pair for pair in uuid_to_status.items()
                if filter_function(pair[1])
            ]
            replay_pairs.sort(key=lambda pair: pair[1].score_info.meta_score, reverse=True)
            if len(replay_pairs) < count_per_rank:
                logger.warning(
                    f"Could only produce {len(replay_pairs)} "
                    f"of the {count_per_rank} requested for {rank}"
                )
            top_replays[rank] = replay_pairs[:count_per_rank]
        return top_replays

    known_errors = [
        "ActorId(-1) not found",
        "Player team unknown",
        "Players found in frames that were not part of",
        "Replay is corrupt",
        "Could not decode replay content data at offset",
        "Could not decode replay header data",
        "Could not find actor for",
        "Car actor for player"
    ]

    def _should_reraise(self, e):
        try:
            exception_text = e.args[0]
        except Exception:
            pass
        else:
            for error_text in self.known_errors:
                if self._ignore_known_errors and error_text in exception_text:
                    return False
        return True

    def _get_replay_status(self, uuid, require_headers=True):
        meta = None
        if (
                isinstance(self._replay_set, load.CachedReplaySet) and not self._always_load_tensor
        ):
            meta = self._replay_set.get_replay_meta(uuid)
            if require_headers and meta is not None and not meta.headers:
                self._replay_set.bust_cache(uuid)
                meta = None

        if meta is None:
            try:
                _, meta = self._replay_set.get_replay_tensor(uuid)
            except Exception as e:
                logger.warn(f"Tensor load failure for {uuid}, {e}")
                if self._should_reraise(e):
                    if self._ipdb_on_exception:
                        import ipdb; ipdb.set_trace()
                    raise e
                else:
                    return self.TensorFail(e)

        if meta.playlist != self._playlist:
            return self.PlaylistFail(Exception("Wrong playlist"))

        score_info = self._scorer.score_replay_meta(meta, playlist=self._playlist)
        score, estimates, scores = score_info

        return self.ScoreInfoStatus(score_info)

    def _check_labels(self, meta):
        try:
            return self._get_player_labels(meta)
        except self.LabelFail as e:
            logger.warn(f"Label failure for {meta}, {e}")
            return e


def get_passed_stats(statuses_by_rank):
    return {
        rank: len(statuses)
        for rank, statuses in statuses_by_rank.items()
    }


class ParallelTensorMetaLoader:

    @classmethod
    def load_all(cls, replay_set):
        return asyncio.run(cls._load_all(replay_set))

    @classmethod
    async def _load_all(cls, replay_set):
        return await cls(replay_set).load_all_tensors()

    def __init__(self, replay_set, executor=None, loop=None):
        logger.info(f"Cpu count: {multiprocessing.cpu_count()}")
        self._executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
        self._replay_set = replay_set
        self._loop = loop or asyncio.get_running_loop()

    def load_replay_tensor(self, uuid):
        return self._loop.run_in_executor(
            self._executor, self._replay_set.get_replay_tensor, uuid
        )

    async def call_load_replay_tensor(self, uuid):
        try:
            logger.info(f"Loading {uuid}")
            await self.load_replay_tensor(uuid)
            logger.info(f"Done {uuid}")
        except Exception as e:
            logger.warn(f"Exception loading tensor {e}")
        return

    async def load_all_tensors(self):
        uuids = self._replay_set.get_replay_uuids()
        await asyncio.gather(*[self.call_load_replay_tensor(uuid) for uuid in uuids])
        return
