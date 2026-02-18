"""Tests for artifact caching."""

from datetime import datetime

import polars as pl
import pytest

from signalflow.config.artifact_cache import ArtifactCache
from signalflow.config.dag import Node
from signalflow.core import Signals


@pytest.fixture
def sample_node():
    return Node(
        id="test_detector",
        type="signals/detector",
        name="sma_cross",
        config={"fast": 10, "slow": 20},
    )


@pytest.fixture
def sample_inputs():
    return {
        "ohlcv": pl.DataFrame(
            {
                "pair": ["BTCUSDT"] * 3,
                "timestamp": [datetime(2024, 1, i) for i in range(1, 4)],
                "close": [100.0, 101.0, 102.0],
            }
        )
    }


@pytest.fixture
def sample_outputs():
    return {
        "signals": Signals(
            pl.DataFrame(
                {
                    "pair": ["BTCUSDT"],
                    "timestamp": [datetime(2024, 1, 2)],
                    "signal_type": [1],
                }
            )
        )
    }


class TestArtifactCacheMemory:
    """Tests for memory cache mode."""

    def test_cache_miss(self, sample_node, sample_inputs):
        cache = ArtifactCache(cache_mode="memory")
        result = cache.get(sample_node, sample_inputs)
        assert result is None

    def test_cache_hit(self, sample_node, sample_inputs, sample_outputs):
        cache = ArtifactCache(cache_mode="memory")
        cache.put(sample_node, sample_inputs, sample_outputs)
        result = cache.get(sample_node, sample_inputs)
        assert result is not None
        assert "signals" in result

    def test_cache_key_consistency(self, sample_node, sample_inputs):
        cache = ArtifactCache(cache_mode="memory")
        key1 = cache.cache_key(sample_node, sample_inputs)
        key2 = cache.cache_key(sample_node, sample_inputs)
        assert key1 == key2

    def test_different_config_different_key(self, sample_inputs):
        cache = ArtifactCache(cache_mode="memory")
        node1 = Node(id="det", type="signals/detector", config={"fast": 10})
        node2 = Node(id="det", type="signals/detector", config={"fast": 20})
        key1 = cache.cache_key(node1, sample_inputs)
        key2 = cache.cache_key(node2, sample_inputs)
        assert key1 != key2

    def test_different_inputs_different_key(self, sample_node):
        cache = ArtifactCache(cache_mode="memory")
        inputs1 = {
            "ohlcv": pl.DataFrame(
                {
                    "pair": ["BTCUSDT"],
                    "timestamp": [datetime(2024, 1, 1)],
                    "close": [100.0],
                }
            )
        }
        inputs2 = {
            "ohlcv": pl.DataFrame(
                {
                    "pair": ["ETHUSDT"],  # Different pair
                    "timestamp": [datetime(2024, 1, 1)],
                    "close": [100.0],
                }
            )
        }
        key1 = cache.cache_key(sample_node, inputs1)
        key2 = cache.cache_key(sample_node, inputs2)
        assert key1 != key2

    def test_clear(self, sample_node, sample_inputs, sample_outputs):
        cache = ArtifactCache(cache_mode="memory")
        cache.put(sample_node, sample_inputs, sample_outputs)
        assert cache.get(sample_node, sample_inputs) is not None
        cache.clear()
        assert cache.get(sample_node, sample_inputs) is None

    def test_stats(self, sample_node, sample_inputs, sample_outputs):
        cache = ArtifactCache(cache_mode="memory")
        stats = cache.stats
        assert stats["mode"] == "memory"
        assert stats["entries"] == 0

        cache.put(sample_node, sample_inputs, sample_outputs)
        stats = cache.stats
        assert stats["entries"] == 1


class TestArtifactCacheDisk:
    """Tests for disk cache mode."""

    def test_cache_hit_disk(self, tmp_path, sample_node, sample_inputs, sample_outputs):
        cache = ArtifactCache(cache_mode="disk", cache_dir=tmp_path)
        cache.put(sample_node, sample_inputs, sample_outputs)

        # New cache instance should find the cached data
        cache2 = ArtifactCache(cache_mode="disk", cache_dir=tmp_path)
        result = cache2.get(sample_node, sample_inputs)
        assert result is not None
        assert "signals" in result

    def test_dataframe_roundtrip(self, tmp_path, sample_node, sample_inputs):
        outputs = {"features": pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})}
        cache = ArtifactCache(cache_mode="disk", cache_dir=tmp_path)
        cache.put(sample_node, sample_inputs, outputs)

        result = cache.get(sample_node, sample_inputs)
        assert result is not None
        assert result["features"].equals(outputs["features"])

    def test_signals_roundtrip(self, tmp_path, sample_node, sample_inputs, sample_outputs):
        cache = ArtifactCache(cache_mode="disk", cache_dir=tmp_path)
        cache.put(sample_node, sample_inputs, sample_outputs)

        result = cache.get(sample_node, sample_inputs)
        assert result is not None
        assert isinstance(result["signals"], Signals)
        original_df = sample_outputs["signals"].value
        result_df = result["signals"].value
        assert result_df.shape == original_df.shape

    def test_dict_roundtrip(self, tmp_path, sample_node, sample_inputs):
        outputs = {"metrics": {"accuracy": 0.95, "f1": 0.88}}
        cache = ArtifactCache(cache_mode="disk", cache_dir=tmp_path)
        cache.put(sample_node, sample_inputs, outputs)

        result = cache.get(sample_node, sample_inputs)
        assert result is not None
        assert result["metrics"] == outputs["metrics"]

    def test_clear_disk(self, tmp_path, sample_node, sample_inputs, sample_outputs):
        cache = ArtifactCache(cache_mode="disk", cache_dir=tmp_path)
        cache.put(sample_node, sample_inputs, sample_outputs)
        assert cache.get(sample_node, sample_inputs) is not None

        cache.clear()
        assert cache.get(sample_node, sample_inputs) is None

    def test_stats_disk(self, tmp_path, sample_node, sample_inputs, sample_outputs):
        cache = ArtifactCache(cache_mode="disk", cache_dir=tmp_path)
        stats = cache.stats
        assert stats["mode"] == "disk"
        assert stats["entries"] == 0

        cache.put(sample_node, sample_inputs, sample_outputs)
        stats = cache.stats
        assert stats["entries"] == 1


class TestArtifactCacheNone:
    """Tests for no-cache mode."""

    def test_always_miss(self, sample_node, sample_inputs, sample_outputs):
        cache = ArtifactCache(cache_mode="none")
        cache.put(sample_node, sample_inputs, sample_outputs)
        result = cache.get(sample_node, sample_inputs)
        assert result is None

    def test_stats_none(self):
        cache = ArtifactCache(cache_mode="none")
        stats = cache.stats
        assert stats["mode"] == "none"


class TestCacheKeyGeneration:
    """Tests for cache key generation."""

    def test_key_length(self, sample_node, sample_inputs):
        cache = ArtifactCache(cache_mode="memory")
        key = cache.cache_key(sample_node, sample_inputs)
        assert len(key) == 16  # MD5 hex digest truncated to 16 chars

    def test_key_is_hex(self, sample_node, sample_inputs):
        cache = ArtifactCache(cache_mode="memory")
        key = cache.cache_key(sample_node, sample_inputs)
        # Should be valid hex
        int(key, 16)

    def test_key_deterministic(self, sample_node, sample_inputs):
        """Same inputs should produce same key across cache instances."""
        cache1 = ArtifactCache(cache_mode="memory")
        cache2 = ArtifactCache(cache_mode="memory")
        key1 = cache1.cache_key(sample_node, sample_inputs)
        key2 = cache2.cache_key(sample_node, sample_inputs)
        assert key1 == key2


class TestDataSignature:
    """Tests for data signature generation."""

    def test_dataframe_signature(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        sig = ArtifactCache._data_signature(df)
        assert len(sig) == 8
        assert isinstance(sig, str)

    def test_signals_signature(self):
        signals = Signals(
            pl.DataFrame(
                {
                    "pair": ["BTCUSDT"],
                    "timestamp": [datetime(2024, 1, 1)],
                    "signal_type": [1],
                }
            )
        )
        sig = ArtifactCache._data_signature(signals)
        assert len(sig) == 8

    def test_same_data_same_signature(self):
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"a": [1, 2, 3]})
        sig1 = ArtifactCache._data_signature(df1)
        sig2 = ArtifactCache._data_signature(df2)
        assert sig1 == sig2

    def test_different_data_different_signature(self):
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"a": [4, 5, 6]})
        sig1 = ArtifactCache._data_signature(df1)
        sig2 = ArtifactCache._data_signature(df2)
        assert sig1 != sig2
