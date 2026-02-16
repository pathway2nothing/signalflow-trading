"""Tests for Deribit data source - DeribitClient, DeribitFuturesLoader."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from signalflow.core.enums import SfComponentType
from signalflow.core.registry import default_registry
from signalflow.data.source._helpers import (
    dt_to_ms_utc,
    normalize_deribit_pair,
    to_deribit_instrument,
)
from signalflow.data.source.deribit import (
    _DERIBIT_INTERVAL_MAP,
    DeribitClient,
    DeribitFuturesLoader,
)

START = datetime(2024, 1, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_deribit_klines_response(timestamps_ms: list[int], base_price: float = 100.0) -> dict:
    """Build a Deribit-style tradingview chart data response."""
    ticks = sorted(timestamps_ms)
    opens = [base_price + i for i in range(len(ticks))]
    highs = [p + 1 for p in opens]
    lows = [p - 1 for p in opens]
    closes = [p + 0.5 for p in opens]
    volumes = [100.0 + i for i in range(len(ticks))]

    return {
        "result": {
            "status": "ok",
            "ticks": ticks,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    }


def _make_deribit_instruments_response(instruments: list[dict]) -> dict:
    """Build Deribit instruments response."""
    return {"result": instruments}


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
# Pair normalization tests
# ---------------------------------------------------------------------------


class TestDeribitPairNormalization:
    def test_btc_perpetual(self) -> None:
        assert normalize_deribit_pair("BTC-PERPETUAL") == "BTCUSD"

    def test_eth_perpetual(self) -> None:
        assert normalize_deribit_pair("ETH-PERPETUAL") == "ETHUSD"

    def test_btc_usdc_perpetual(self) -> None:
        assert normalize_deribit_pair("BTC-USDC-PERPETUAL") == "BTCUSDC"

    def test_dated_futures(self) -> None:
        assert normalize_deribit_pair("BTC-27DEC24") == "BTCUSD"

    def test_to_deribit_btcusd(self) -> None:
        assert to_deribit_instrument("BTCUSD") == "BTC-PERPETUAL"

    def test_to_deribit_ethusdc(self) -> None:
        assert to_deribit_instrument("ETHUSDC") == "ETH-USDC-PERPETUAL"


# ---------------------------------------------------------------------------
# DeribitClient tests
# ---------------------------------------------------------------------------


class TestDeribitClient:
    def test_raises_without_context_manager(self) -> None:
        client = DeribitClient()
        with pytest.raises(RuntimeError, match="async context manager"):
            asyncio.run(client.get_klines("BTC-PERPETUAL"))

    def test_get_pairs_raises_without_context_manager(self) -> None:
        client = DeribitClient()
        with pytest.raises(RuntimeError, match="async context manager"):
            asyncio.run(client.get_pairs())

    def test_unsupported_timeframe(self) -> None:
        async def _run():
            async with DeribitClient() as client:
                with pytest.raises(ValueError, match="Unsupported timeframe"):
                    await client.get_klines("BTC-PERPETUAL", timeframe="2m")

        asyncio.run(_run())

    def test_get_klines_parses_response(self) -> None:
        ts1 = dt_to_ms_utc(START)
        ts2 = dt_to_ms_utc(START + timedelta(minutes=1))
        ts3 = dt_to_ms_utc(START + timedelta(minutes=2))

        body = _make_deribit_klines_response([ts1, ts2, ts3])
        mock_resp = _mock_response(body)

        async def _run():
            async with DeribitClient() as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                klines = await client.get_klines("BTC-PERPETUAL", timeframe="1m")

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

    def test_get_klines_shifts_to_close_time(self) -> None:
        t1 = START
        body = _make_deribit_klines_response([dt_to_ms_utc(t1)])
        mock_resp = _mock_response(body)

        async def _run():
            async with DeribitClient() as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                klines = await client.get_klines("BTC-PERPETUAL", timeframe="1m")

                # timestamp should be close time (open + 1 tf)
                assert klines[0]["timestamp"] == t1 + timedelta(minutes=1)

        asyncio.run(_run())

    def test_get_klines_api_error(self) -> None:
        body = {"error": {"code": 10001, "message": "Instrument not found"}}
        mock_resp = _mock_response(body)

        async def _run():
            async with DeribitClient(max_retries=1) as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                with pytest.raises(RuntimeError, match="Deribit API error"):
                    await client.get_klines("INVALID", timeframe="1m")

        asyncio.run(_run())

    def test_get_klines_range_empty(self) -> None:
        async def _run():
            async with DeribitClient() as client:
                result = await client.get_klines_range(
                    "BTC-PERPETUAL",
                    "1m",
                    start_time=START,
                    end_time=START - timedelta(hours=1),
                )
                assert result == []

        asyncio.run(_run())


class TestDeribitClientGetPairs:
    def test_get_pairs_parses_response(self) -> None:
        instruments = [
            {"instrument_name": "BTC-PERPETUAL", "is_active": True},
            {"instrument_name": "ETH-PERPETUAL", "is_active": True},
            {"instrument_name": "BTC-27DEC24", "is_active": True},
            {"instrument_name": "OLD-PERP", "is_active": False},
        ]
        body = _make_deribit_instruments_response(instruments)
        mock_resp = _mock_response(body)

        async def _run():
            async with DeribitClient() as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                pairs = await client.get_pairs(currency="BTC")
                assert "BTC-PERPETUAL" in pairs
                assert "ETH-PERPETUAL" in pairs
                assert "OLD-PERP" not in pairs  # inactive

        asyncio.run(_run())


class TestDeribitTimeframeMapping:
    def test_all_signalflow_timeframes_mapped(self) -> None:
        expected = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"}
        assert set(_DERIBIT_INTERVAL_MAP.keys()) == expected


# ---------------------------------------------------------------------------
# Component registration tests
# ---------------------------------------------------------------------------


class TestDeribitRegistration:
    def test_deribit_client_registered(self) -> None:
        cls = default_registry.get(SfComponentType.RAW_DATA_SOURCE, "deribit")
        assert cls is DeribitClient

    def test_deribit_futures_loader_registered(self) -> None:
        cls = default_registry.get(SfComponentType.RAW_DATA_LOADER, "deribit/futures")
        assert cls is DeribitFuturesLoader


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


class TestDeribitLoaderGetPairs:
    @patch.object(DeribitClient, "get_pairs")
    def test_futures_loader_get_pairs(self, mock_get_pairs: AsyncMock) -> None:
        mock_get_pairs.return_value = ["BTC-PERPETUAL", "ETH-PERPETUAL"]

        async def _run():
            loader = DeribitFuturesLoader()
            pairs = await loader.get_pairs()
            assert pairs == ["BTC-PERPETUAL", "ETH-PERPETUAL"]

        asyncio.run(_run())
        mock_get_pairs.assert_called_once_with(currency="BTC", kind="future")
