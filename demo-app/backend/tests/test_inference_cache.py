from __future__ import annotations

from app.inference import LRUFrameCache


def test_lru_cache_hit_miss_and_eviction() -> None:
    cache = LRUFrameCache(max_entries=2)

    assert cache.get(("p1", 0)) is None

    cache.set(("p1", 0), object())
    cache.set(("p1", 1), object())
    assert cache.get(("p1", 0)) is not None

    cache.set(("p1", 2), object())
    assert cache.get(("p1", 1)) is None
    assert cache.get(("p1", 0)) is not None
    assert cache.get(("p1", 2)) is not None
