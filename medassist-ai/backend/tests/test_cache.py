import fakeredis
import pytest
from unittest.mock import patch

from app.core.cache import Cache, cache_key_from_text
from app.core.observability import CACHE_HITS, CACHE_MISSES


@pytest.fixture
def fake_redis_client():
    return fakeredis.FakeRedis(decode_responses=False)


@pytest.fixture
def cache_with_fake_redis(fake_redis_client):
    with patch("app.core.cache.get_redis_client", return_value=fake_redis_client):
        yield Cache("test_namespace")


def test_set_then_get_returns_the_value(cache_with_fake_redis):
    cache_with_fake_redis.set("key1", {"a": 1}, ttl_seconds=60)
    assert cache_with_fake_redis.get("key1") == {"a": 1}


def test_get_missing_key_returns_none(cache_with_fake_redis):
    assert cache_with_fake_redis.get("nonexistent") is None


def test_delete_removes_the_key(cache_with_fake_redis):
    cache_with_fake_redis.set("key1", "value", ttl_seconds=60)
    cache_with_fake_redis.delete("key1")
    assert cache_with_fake_redis.get("key1") is None


def test_namespaces_are_isolated(fake_redis_client):
    with patch("app.core.cache.get_redis_client", return_value=fake_redis_client):
        cache_a = Cache("namespace_a")
        cache_b = Cache("namespace_b")
        cache_a.set("same_key", "value_a", ttl_seconds=60)
        assert cache_b.get("same_key") is None


def test_ttl_expiry_is_set_on_the_underlying_key(cache_with_fake_redis, fake_redis_client):
    cache_with_fake_redis.set("key1", "value", ttl_seconds=60)
    ttl = fake_redis_client.ttl(cache_with_fake_redis._key("key1"))
    assert 0 < ttl <= 60


def test_get_returns_none_and_does_not_raise_when_redis_unavailable():
    with patch("app.core.cache.get_redis_client", return_value=None):
        cache = Cache("test_namespace")
        assert cache.get("anything") is None


def test_set_is_a_silent_no_op_when_redis_unavailable():
    with patch("app.core.cache.get_redis_client", return_value=None):
        cache = Cache("test_namespace")
        cache.set("key1", "value", ttl_seconds=60)  # must not raise
        assert cache.get("key1") is None


def test_cache_hit_and_miss_counters_increment(cache_with_fake_redis):
    before_misses = CACHE_MISSES.labels("test_namespace")._value.get()
    cache_with_fake_redis.get("not_set_yet")
    after_misses = CACHE_MISSES.labels("test_namespace")._value.get()
    assert after_misses == before_misses + 1

    cache_with_fake_redis.set("key1", "value", ttl_seconds=60)
    before_hits = CACHE_HITS.labels("test_namespace")._value.get()
    cache_with_fake_redis.get("key1")
    after_hits = CACHE_HITS.labels("test_namespace")._value.get()
    assert after_hits == before_hits + 1


def test_clear_namespace_removes_only_that_namespace(fake_redis_client):
    with patch("app.core.cache.get_redis_client", return_value=fake_redis_client):
        cache_a = Cache("namespace_a")
        cache_b = Cache("namespace_b")
        cache_a.set("k1", "v1", ttl_seconds=60)
        cache_a.set("k2", "v2", ttl_seconds=60)
        cache_b.set("k1", "v1", ttl_seconds=60)

        deleted = cache_a.clear_namespace()

        assert deleted == 2
        assert cache_a.get("k1") is None
        assert cache_b.get("k1") == "v1"  # untouched


def test_cache_key_from_text_is_deterministic():
    key1 = cache_key_from_text("question", "category", "5")
    key2 = cache_key_from_text("question", "category", "5")
    assert key1 == key2


def test_cache_key_from_text_differs_for_different_inputs():
    key1 = cache_key_from_text("question A", "category", "5")
    key2 = cache_key_from_text("question B", "category", "5")
    assert key1 != key2
