import logging

from .. import manifest


logger = logging.getLogger(__name__)


async def async_require_at_least_one_non_null_mmr(_, replay_meta):
    return require_at_least_one_non_null_mmr(replay_meta)


def require_at_least_one_non_null_mmr(replay_meta):
    """Require that at least one of the mmrs in the replay meta from ballchasing is non-null.

    This is actually important because there are replays on ballchasing.com for
    which there is no estimate on mmr for any player that will pass through any
    minimum or maximum rank filter. This filter is a quick and dirty way to
    ensure that the filter is at least semi effective.
    """
    try:
        mmr_estimates = manifest.get_mmr_data_from_manifest_game(replay_meta)
    except Exception:
        logger.warn("Exception getting mmr_estimate")
        return False, replay_meta
    return any(
        value is not None
        for value in mmr_estimates.values()
    ), replay_meta


def build_filter_existing(replay_exists):
    """Filter any tasks for replays that already exist."""
    async def filter_existing(_, replay_meta):
        return (not replay_exists(replay_meta['id'])), replay_meta
    return filter_existing


def compose_filters(*filters):
    """Compose the provided filters."""
    async def new_filter(session, replay_meta):
        for next_filter in filters:
            should_enqueue, replay_meta = await next_filter(session, replay_meta)
            if not should_enqueue:
                break
        return should_enqueue, replay_meta
    return new_filter


def compose_filters_with_reasons(*filters):
    """Compose the provided filters."""
    async def new_filter(session, replay_meta):
        for (reason, next_filter) in filters:
            should_enqueue, replay_meta = await next_filter(session, replay_meta)
            if not should_enqueue:
                logger.warn(f"{replay_meta['id']} filtered because {reason}")
                break
        return should_enqueue, replay_meta
    return new_filter
