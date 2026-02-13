"""Tests for WhiteBIT data source - WhitebitClient, WhitebitSpotLoader."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from signalflow.core.enums import SfComponentType
from signalflow.core.registry import default_registry
from signalflow.data.source.whitebit import (
    WhitebitClient,
    WhitebitSpotLoader,
    _WHITEBIT_INTERVAL_MAP,
)
from signalflow.data.source._helpers import (
    normalize_whitebit_pair,
    to_whitebit_symbol,
    dt_to_sec_utc,
)

START = datetime(2024, 1, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_whitebit_klines_response(timestamps_sec: list[int], base_price: float = 100.0) -> dict:
    """Build a WhiteBIT-style kline response.

    WhiteBIT format: [timestamp, open, close, high, low, volume_stock, volume_money]
    Note: close comes BEFORE high/low!
    """
    result = []
    for i, ts in enumerate(sorted(timestamps_sec)):
        p = base_price + i
        result.append([
            ts,             # timestamp (open time in seconds)
            str(p),         # open
            str(p + 0.5),   # close (index 2!)
            str(p + 1),     # high
            str(p - 1),     # low
            str(100.0 + i), # volume_stock
            str(10000.0 + i * 100),  # volume_money
        ])
    return {"success": True, "result": result}


def _make_whitebit_markets_response(markets: list[str]) -> list[dict]:
    """Build WhiteBIT markets response."""
    return [{"name": m} for m in markets]


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


class TestWhitebitPairNormalization:
    def test_btc_usdt_to_btcusdt(self) -> None:
        assert normalize_whitebit_pair("BTC_USDT") == "BTCUSDT"

    def test_eth_usdt_to_ethusdt(self) -> None:
        assert normalize_whitebit_pair("ETH_USDT") == "ETHUSDT"

    def test_lowercase_btc_usdt(self) -> None:
        assert normalize_whitebit_pair("btc_usdt") == "BTCUSDT"

    def test_btcusdt_to_btc_usdt(self) -> None:
        assert to_whitebit_symbol("BTCUSDT") == "BTC_USDT"

    def test_ethusdc_to_eth_usdc(self) -> None:
        assert to_whitebit_symbol("ETHUSDC") == "ETH_USDC"

    def test_btcusd_to_btc_usd(self) -> None:
        assert to_whitebit_symbol("BTCUSD") == "BTC_USD"


# ---------------------------------------------------------------------------
# WhitebitClient tests
# ---------------------------------------------------------------------------


class TestWhitebitClient:
    def test_raises_without_context_manager(self) -> None:
        client = WhitebitClient()
        with pytest.raises(RuntimeError, match="async context manager"):
            asyncio.run(client.get_klines("BTC_USDT"))

    def test_get_pairs_raises_without_context_manager(self) -> None:
        client = WhitebitClient()
        with pytest.raises(RuntimeError, match="async context manager"):
            asyncio.run(client.get_pairs())

    def test_unsupported_timeframe(self) -> None:
        async def _run():
            async with WhitebitClient() as client:
                with pytest.raises(ValueError, match="Unsupported timeframe"):
                    await client.get_klines("BTC_USDT", timeframe="7h")

        asyncio.run(_run())

    def test_get_klines_parses_response(self) -> None:
        ts1 = dt_to_sec_utc(START)
        ts2 = dt_to_sec_utc(START + timedelta(minutes=1))
        ts3 = dt_to_sec_utc(START + timedelta(minutes=2))

        body = _make_whitebit_klines_response([ts1, ts2, ts3])
        mock_resp = _mock_response(body)

        async def _run():
            async with WhitebitClient() as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                klines = await client.get_klines("BTC_USDT", timeframe="1m")

                assert len(klines) == 3
                # Should be ascending order
                assert klines[0]["timestamp"] < klines[1]["timestamp"] < klines[2]["timestamp"]
                # Check structure
                for k in klines:
                    assert set(k.keys()) == {"timestamp", "open", "high", "low", "close", "volume", "trades"}
                    assert isinstance(k["timestamp"], datetime)
                    assert isinstance(k["open"], float)
                    assert k["trades"] == 0  # WhiteBIT doesn't provide trade count

        asyncio.run(_run())

    def test_get_klines_shifts_to_close_time(self) -> None:
        t1_sec = dt_to_sec_utc(START)
        body = _make_whitebit_klines_response([t1_sec])
        mock_resp = _mock_response(body)

        async def _run():
            async with WhitebitClient() as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                klines = await client.get_klines("BTC_USDT", timeframe="1m")

                # timestamp should be close time (open + 1 minute)
                assert klines[0]["timestamp"] == START + timedelta(minutes=1)

        asyncio.run(_run())

    def test_get_klines_api_error(self) -> None:
        body = {"success": False, "message": "Invalid market"}
        mock_resp = _mock_response(body)

        async def _run():
            async with WhitebitClient(max_retries=1) as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                with pytest.raises(RuntimeError, match="WhiteBIT API error"):
                    await client.get_klines("INVALID_USDT", timeframe="1m")

        asyncio.run(_run())

    def test_get_klines_range_empty(self) -> None:
        async def _run():
            async with WhitebitClient() as client:
                result = await client.get_klines_range(
                    "BTC_USDT",
                    "1m",
                    start_time=START,
                    end_time=START - timedelta(hours=1),
                )
                assert result == []

        asyncio.run(_run())


class TestWhitebitClientGetPairs:
    def test_get_pairs_parses_response(self) -> None:
        body = _make_whitebit_markets_response(["BTC_USDT", "ETH_USDT", "SOL_USDT", "BTC_UAH"])
        mock_resp = _mock_response(body)

        async def _run():
            async with WhitebitClient() as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                pairs = await client.get_pairs()
                assert "BTC_USDT" in pairs
                assert "ETH_USDT" in pairs
                assert "BTC_UAH" in pairs

        asyncio.run(_run())

    def test_get_pairs_filters_by_quote(self) -> None:
        body = _make_whitebit_markets_response(["BTC_USDT", "ETH_USDT", "BTC_UAH", "ETH_EUR"])
        mock_resp = _mock_response(body)

        async def _run():
            async with WhitebitClient() as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                pairs = await client.get_pairs(quote="USDT")
                assert "BTC_USDT" in pairs
                assert "ETH_USDT" in pairs
                assert "BTC_UAH" not in pairs
                assert "ETH_EUR" not in pairs

        asyncio.run(_run())


class TestWhitebitTimeframeMapping:
    def test_supported_timeframes(self) -> None:
        expected = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"}
        assert set(_WHITEBIT_INTERVAL_MAP.keys()) == expected


# ---------------------------------------------------------------------------
# Component registration tests
# ---------------------------------------------------------------------------


class TestWhitebitRegistration:
    def test_whitebit_client_registered(self) -> None:
        cls = default_registry.get(SfComponentType.RAW_DATA_SOURCE, "whitebit")
        assert cls is WhitebitClient

    def test_whitebit_spot_loader_registered(self) -> None:
        cls = default_registry.get(SfComponentType.RAW_DATA_LOADER, "whitebit/spot")
        assert cls is WhitebitSpotLoader


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


class TestWhitebitLoaderGetPairs:
    @patch.object(WhitebitClient, "get_pairs")
    def test_spot_loader_get_pairs(self, mock_get_pairs: AsyncMock) -> None:
        mock_get_pairs.return_value = ["BTC_USDT", "ETH_USDT"]

        async def _run():
            loader = WhitebitSpotLoader()
            pairs = await loader.get_pairs(quote="USDT")
            assert pairs == ["BTC_USDT", "ETH_USDT"]

        asyncio.run(_run())
        mock_get_pairs.assert_called_once_with(quote="USDT")
