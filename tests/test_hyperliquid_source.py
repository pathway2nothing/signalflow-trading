"""Tests for Hyperliquid data source - HyperliquidClient, HyperliquidFuturesLoader."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from signalflow.core.enums import SfComponentType
from signalflow.core.registry import default_registry
from signalflow.data.source.hyperliquid import (
    HyperliquidClient,
    HyperliquidFuturesLoader,
    _HYPERLIQUID_INTERVAL_MAP,
)
from signalflow.data.source._helpers import (
    normalize_hyperliquid_pair,
    to_hyperliquid_coin,
    dt_to_ms_utc,
)

START = datetime(2024, 1, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hyperliquid_klines_response(timestamps_ms: list[int], base_price: float = 100.0) -> list:
    """Build a Hyperliquid-style candleSnapshot response."""
    candles = []
    for i, close_ts in enumerate(sorted(timestamps_ms)):
        # t = open time, T = close time
        open_ts = close_ts - 60_000  # 1 minute before
        candles.append({
            "t": open_ts,
            "T": close_ts,
            "o": str(base_price + i),
            "h": str(base_price + i + 1),
            "l": str(base_price + i - 1),
            "c": str(base_price + i + 0.5),
            "v": str(100.0 + i),
            "n": 10 + i,
            "s": "BTC",
            "i": "1m",
        })
    return candles


def _make_hyperliquid_meta_response(coins: list[str]) -> dict:
    """Build Hyperliquid meta response."""
    universe = [{"name": coin} for coin in coins]
    return {"universe": universe}


def _mock_response(body, status: int = 200):
    """Create a mock aiohttp response context manager."""
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=body)
    resp.text = AsyncMock(return_value=json.dumps(body))
    resp.headers = {}

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


# ---------------------------------------------------------------------------
# Pair normalization tests
# ---------------------------------------------------------------------------


class TestHyperliquidPairNormalization:
    def test_btc_to_btcusd(self) -> None:
        assert normalize_hyperliquid_pair("BTC") == "BTCUSD"

    def test_eth_to_ethusd(self) -> None:
        assert normalize_hyperliquid_pair("ETH") == "ETHUSD"

    def test_sol_to_solusd(self) -> None:
        assert normalize_hyperliquid_pair("sol") == "SOLUSD"

    def test_btcusd_to_btc(self) -> None:
        assert to_hyperliquid_coin("BTCUSD") == "BTC"

    def test_ethusdt_to_eth(self) -> None:
        assert to_hyperliquid_coin("ETHUSDT") == "ETH"

    def test_btcusdc_to_btc(self) -> None:
        assert to_hyperliquid_coin("BTCUSDC") == "BTC"


# ---------------------------------------------------------------------------
# HyperliquidClient tests
# ---------------------------------------------------------------------------


class TestHyperliquidClient:
    def test_raises_without_context_manager(self) -> None:
        client = HyperliquidClient()
        with pytest.raises(RuntimeError, match="async context manager"):
            asyncio.run(client.get_klines("BTC"))

    def test_get_pairs_raises_without_context_manager(self) -> None:
        client = HyperliquidClient()
        with pytest.raises(RuntimeError, match="async context manager"):
            asyncio.run(client.get_pairs())

    def test_unsupported_timeframe(self) -> None:
        async def _run():
            async with HyperliquidClient() as client:
                with pytest.raises(ValueError, match="Unsupported timeframe"):
                    await client.get_klines("BTC", timeframe="6h")

        asyncio.run(_run())

    def test_get_klines_parses_response(self) -> None:
        ts1 = dt_to_ms_utc(START + timedelta(minutes=1))
        ts2 = dt_to_ms_utc(START + timedelta(minutes=2))
        ts3 = dt_to_ms_utc(START + timedelta(minutes=3))

        body = _make_hyperliquid_klines_response([ts1, ts2, ts3])
        mock_resp = _mock_response(body)

        async def _run():
            async with HyperliquidClient() as client:
                mock_session = AsyncMock()
                mock_session.post = MagicMock(return_value=mock_resp)
                client._session = mock_session

                klines = await client.get_klines("BTC", timeframe="1m")

                assert len(klines) == 3
                # Should be ascending order
                assert klines[0]["timestamp"] < klines[1]["timestamp"] < klines[2]["timestamp"]
                # Check structure
                for k in klines:
                    assert set(k.keys()) == {"timestamp", "open", "high", "low", "close", "volume", "trades"}
                    assert isinstance(k["timestamp"], datetime)
                    assert isinstance(k["open"], float)
                    assert isinstance(k["trades"], int)

        asyncio.run(_run())

    def test_get_klines_uses_close_time(self) -> None:
        close_time = dt_to_ms_utc(START + timedelta(minutes=1))
        body = _make_hyperliquid_klines_response([close_time])
        mock_resp = _mock_response(body)

        async def _run():
            async with HyperliquidClient() as client:
                mock_session = AsyncMock()
                mock_session.post = MagicMock(return_value=mock_resp)
                client._session = mock_session

                klines = await client.get_klines("BTC", timeframe="1m")

                # timestamp should be close time (T field)
                assert klines[0]["timestamp"] == START + timedelta(minutes=1)

        asyncio.run(_run())

    def test_get_klines_api_error(self) -> None:
        body = {"error": "Invalid coin"}
        mock_resp = _mock_response(body)

        async def _run():
            async with HyperliquidClient(max_retries=1) as client:
                mock_session = AsyncMock()
                mock_session.post = MagicMock(return_value=mock_resp)
                client._session = mock_session

                with pytest.raises(RuntimeError, match="Hyperliquid API error"):
                    await client.get_klines("INVALID", timeframe="1m")

        asyncio.run(_run())

    def test_get_klines_range_empty(self) -> None:
        async def _run():
            async with HyperliquidClient() as client:
                result = await client.get_klines_range(
                    "BTC",
                    "1m",
                    start_time=START,
                    end_time=START - timedelta(hours=1),
                )
                assert result == []

        asyncio.run(_run())


class TestHyperliquidClientGetPairs:
    def test_get_pairs_parses_response(self) -> None:
        body = _make_hyperliquid_meta_response(["BTC", "ETH", "SOL", "ARB"])
        mock_resp = _mock_response(body)

        async def _run():
            async with HyperliquidClient() as client:
                mock_session = AsyncMock()
                mock_session.post = MagicMock(return_value=mock_resp)
                client._session = mock_session

                coins = await client.get_pairs()
                assert coins == ["ARB", "BTC", "ETH", "SOL"]  # sorted

        asyncio.run(_run())


class TestHyperliquidTimeframeMapping:
    def test_supported_timeframes(self) -> None:
        expected = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "1d"}
        assert set(_HYPERLIQUID_INTERVAL_MAP.keys()) == expected


# ---------------------------------------------------------------------------
# Component registration tests
# ---------------------------------------------------------------------------


class TestHyperliquidRegistration:
    def test_hyperliquid_client_registered(self) -> None:
        cls = default_registry.get(SfComponentType.RAW_DATA_SOURCE, "hyperliquid")
        assert cls is HyperliquidClient

    def test_hyperliquid_futures_loader_registered(self) -> None:
        cls = default_registry.get(SfComponentType.RAW_DATA_LOADER, "hyperliquid/futures")
        assert cls is HyperliquidFuturesLoader


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


class TestHyperliquidLoaderGetPairs:
    @patch.object(HyperliquidClient, "get_pairs")
    def test_futures_loader_get_pairs(self, mock_get_pairs: AsyncMock) -> None:
        mock_get_pairs.return_value = ["BTC", "ETH", "SOL"]

        async def _run():
            loader = HyperliquidFuturesLoader()
            coins = await loader.get_pairs()
            assert coins == ["BTC", "ETH", "SOL"]

        asyncio.run(_run())
        mock_get_pairs.assert_called_once()
