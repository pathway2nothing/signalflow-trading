"""Tests for Bybit data source - BybitClient, BybitSpotLoader, BybitFuturesLoader."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from signalflow.core.enums import SfComponentType
from signalflow.core.registry import default_registry
from signalflow.data.source.bybit import (
    BybitClient,
    BybitFuturesLoader,
    BybitSpotLoader,
    _BYBIT_INTERVAL_MAP,
)
from signalflow.data.source._helpers import dt_to_ms_utc, ms_to_dt_utc_naive

START = datetime(2024, 1, 1)
PAIR = "BTCUSDT"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bybit_response(timestamps_ms: list[int], base_price: float = 100.0) -> dict:
    """Build a Bybit-style JSON response body (descending order)."""
    rows = []
    for i, ts in enumerate(sorted(timestamps_ms, reverse=True)):
        p = base_price + i
        rows.append(
            [
                str(ts),  # open time ms
                str(p),  # open
                str(p + 1),  # high
                str(p - 1),  # low
                str(p + 0.5),  # close
                str(100.0 + i),  # volume
                str(50000.0 + i),  # turnover
            ]
        )
    return {"retCode": 0, "retMsg": "OK", "result": {"list": rows}}


def _mock_response(body: dict, status: int = 200):
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
# BybitClient tests
# ---------------------------------------------------------------------------


class TestBybitClient:
    def test_raises_without_context_manager(self) -> None:
        client = BybitClient()
        with pytest.raises(RuntimeError, match="async context manager"):
            asyncio.run(client.get_klines(PAIR))

    def test_get_pairs_raises_without_context_manager(self) -> None:
        client = BybitClient()
        with pytest.raises(RuntimeError, match="async context manager"):
            asyncio.run(client.get_pairs())

    def test_unsupported_timeframe(self) -> None:
        async def _run():
            async with BybitClient() as client:
                with pytest.raises(ValueError, match="Unsupported timeframe"):
                    await client.get_klines(PAIR, timeframe="2m")

        asyncio.run(_run())

    def test_get_klines_parses_response(self) -> None:
        ts1 = dt_to_ms_utc(START)
        ts2 = dt_to_ms_utc(START + timedelta(minutes=1))
        ts3 = dt_to_ms_utc(START + timedelta(minutes=2))

        body = _make_bybit_response([ts1, ts2, ts3])
        mock_resp = _mock_response(body)

        async def _run():
            async with BybitClient() as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                klines = await client.get_klines(PAIR, timeframe="1m")

                assert len(klines) == 3
                # Should be ascending order
                assert klines[0]["timestamp"] < klines[1]["timestamp"] < klines[2]["timestamp"]
                # Check structure
                for k in klines:
                    assert set(k.keys()) == {"timestamp", "open", "high", "low", "close", "volume", "trades"}
                    assert isinstance(k["timestamp"], datetime)
                    assert isinstance(k["open"], float)
                    assert k["trades"] == 0

        asyncio.run(_run())

    def test_get_klines_reverses_descending(self) -> None:
        t1 = START
        t2 = START + timedelta(minutes=1)
        t3 = START + timedelta(minutes=2)

        body = _make_bybit_response([dt_to_ms_utc(t1), dt_to_ms_utc(t2), dt_to_ms_utc(t3)])
        mock_resp = _mock_response(body)

        async def _run():
            async with BybitClient() as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                klines = await client.get_klines(PAIR, timeframe="1m")

                # timestamps are shifted to close time (open + 1 tf)
                assert klines[0]["timestamp"] == t1 + timedelta(minutes=1)
                assert klines[2]["timestamp"] == t3 + timedelta(minutes=1)

        asyncio.run(_run())

    def test_get_klines_api_error(self) -> None:
        body = {"retCode": 10001, "retMsg": "Invalid symbol"}
        mock_resp = _mock_response(body)

        async def _run():
            async with BybitClient(max_retries=1) as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                with pytest.raises(RuntimeError, match="Bybit API error"):
                    await client.get_klines(PAIR, timeframe="1m")

        asyncio.run(_run())

    def test_get_klines_range_empty(self) -> None:
        async def _run():
            async with BybitClient() as client:
                result = await client.get_klines_range(
                    PAIR,
                    "spot",
                    "1m",
                    start_time=START,
                    end_time=START - timedelta(hours=1),
                )
                assert result == []

        asyncio.run(_run())

    def test_get_klines_range_unsupported_timeframe(self) -> None:
        async def _run():
            async with BybitClient() as client:
                with pytest.raises(ValueError, match="Unsupported timeframe"):
                    await client.get_klines_range(
                        PAIR,
                        "spot",
                        "2m",
                        start_time=START,
                        end_time=START + timedelta(hours=1),
                    )

        asyncio.run(_run())


class TestBybitTimeframeMapping:
    def test_all_signalflow_timeframes_mapped(self) -> None:
        expected = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"}
        assert set(_BYBIT_INTERVAL_MAP.keys()) == expected

    def test_minute_intervals(self) -> None:
        assert _BYBIT_INTERVAL_MAP["1m"] == "1"
        assert _BYBIT_INTERVAL_MAP["5m"] == "5"
        assert _BYBIT_INTERVAL_MAP["30m"] == "30"

    def test_hour_intervals(self) -> None:
        assert _BYBIT_INTERVAL_MAP["1h"] == "60"
        assert _BYBIT_INTERVAL_MAP["4h"] == "240"

    def test_day_interval(self) -> None:
        assert _BYBIT_INTERVAL_MAP["1d"] == "D"


# ---------------------------------------------------------------------------
# Component registration tests
# ---------------------------------------------------------------------------


class TestBybitRegistration:
    def test_bybit_client_registered(self) -> None:
        cls = default_registry.get(SfComponentType.RAW_DATA_SOURCE, "bybit")
        assert cls is BybitClient

    def test_bybit_spot_loader_registered(self) -> None:
        cls = default_registry.get(SfComponentType.RAW_DATA_LOADER, "bybit/spot")
        assert cls is BybitSpotLoader

    def test_bybit_futures_loader_registered(self) -> None:
        cls = default_registry.get(SfComponentType.RAW_DATA_LOADER, "bybit/futures")
        assert cls is BybitFuturesLoader

    def test_futures_loader_default_category(self) -> None:
        assert BybitFuturesLoader.category == "linear"


# ---------------------------------------------------------------------------
# get_pairs() tests
# ---------------------------------------------------------------------------


def _make_bybit_instruments_response(instruments: list[dict]) -> dict:
    """Build Bybit instruments response."""
    return {"retCode": 0, "retMsg": "OK", "result": {"list": instruments}}


class TestBybitClientGetPairs:
    def test_get_pairs_spot(self) -> None:
        instruments = [
            {"symbol": "BTCUSDT", "quoteCoin": "USDT", "status": "Trading"},
            {"symbol": "ETHUSDT", "quoteCoin": "USDT", "status": "Trading"},
            {"symbol": "ETHBTC", "quoteCoin": "BTC", "status": "Trading"},
            {"symbol": "OLDUSDT", "quoteCoin": "USDT", "status": "Closed"},
        ]
        body = _make_bybit_instruments_response(instruments)
        mock_resp = _mock_response(body)

        async def _run():
            async with BybitClient() as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                # All pairs
                pairs = await client.get_pairs(category="spot")
                assert pairs == ["BTCUSDT", "ETHBTC", "ETHUSDT"]
                assert "OLDUSDT" not in pairs  # Closed

        asyncio.run(_run())

    def test_get_pairs_spot_quote_filter(self) -> None:
        instruments = [
            {"symbol": "BTCUSDT", "quoteCoin": "USDT", "status": "Trading"},
            {"symbol": "ETHUSDT", "quoteCoin": "USDT", "status": "Trading"},
            {"symbol": "ETHBTC", "quoteCoin": "BTC", "status": "Trading"},
        ]
        body = _make_bybit_instruments_response(instruments)
        mock_resp = _mock_response(body)

        async def _run():
            async with BybitClient() as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                pairs = await client.get_pairs(category="spot", quote="USDT")
                assert pairs == ["BTCUSDT", "ETHUSDT"]
                assert "ETHBTC" not in pairs

        asyncio.run(_run())

    def test_get_pairs_linear(self) -> None:
        instruments = [
            {"symbol": "BTCUSDT", "settleCoin": "USDT", "status": "Trading"},
            {"symbol": "ETHUSDT", "settleCoin": "USDT", "status": "Trading"},
            {"symbol": "BTCPERP", "settleCoin": "USDC", "status": "Trading"},
        ]
        body = _make_bybit_instruments_response(instruments)
        mock_resp = _mock_response(body)

        async def _run():
            async with BybitClient() as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                # Filter by USDT settlement
                pairs = await client.get_pairs(category="linear", quote="USDT")
                assert pairs == ["BTCUSDT", "ETHUSDT"]

        asyncio.run(_run())

    def test_get_pairs_api_error(self) -> None:
        body = {"retCode": 10001, "retMsg": "Service unavailable"}
        mock_resp = _mock_response(body)

        async def _run():
            async with BybitClient(max_retries=1) as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                with pytest.raises(RuntimeError, match="Bybit API error"):
                    await client.get_pairs()

        asyncio.run(_run())


class TestBybitLoaderGetPairs:
    @patch.object(BybitClient, "get_pairs")
    def test_spot_loader_get_pairs(self, mock_get_pairs: AsyncMock) -> None:
        mock_get_pairs.return_value = ["BTCUSDT", "ETHUSDT"]

        async def _run():
            loader = BybitSpotLoader()
            pairs = await loader.get_pairs(quote="USDT")
            assert pairs == ["BTCUSDT", "ETHUSDT"]

        asyncio.run(_run())
        mock_get_pairs.assert_called_once_with(category="spot", quote="USDT")

    @patch.object(BybitClient, "get_pairs")
    def test_futures_loader_get_pairs_linear(self, mock_get_pairs: AsyncMock) -> None:
        mock_get_pairs.return_value = ["BTCUSDT", "ETHUSDT"]

        async def _run():
            loader = BybitFuturesLoader(category="linear")
            pairs = await loader.get_pairs(quote="USDT")
            assert pairs == ["BTCUSDT", "ETHUSDT"]

        asyncio.run(_run())
        mock_get_pairs.assert_called_once_with(category="linear", quote="USDT")

    @patch.object(BybitClient, "get_pairs")
    def test_futures_loader_get_pairs_inverse(self, mock_get_pairs: AsyncMock) -> None:
        mock_get_pairs.return_value = ["BTCUSD", "ETHUSD"]

        async def _run():
            loader = BybitFuturesLoader(category="inverse")
            pairs = await loader.get_pairs()
            assert pairs == ["BTCUSD", "ETHUSD"]

        asyncio.run(_run())
        mock_get_pairs.assert_called_once_with(category="inverse", quote=None)
