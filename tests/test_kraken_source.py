"""Tests for Kraken data source - KrakenClient, KrakenSpotLoader, KrakenFuturesLoader."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from signalflow.core.enums import SfComponentType
from signalflow.core.registry import default_registry
from signalflow.data.source.kraken import (
    KrakenClient,
    KrakenSpotLoader,
    KrakenFuturesLoader,
    _KRAKEN_SPOT_INTERVAL_MAP,
    _KRAKEN_FUTURES_INTERVAL_MAP,
)
from signalflow.data.source._helpers import (
    normalize_kraken_spot_pair,
    to_kraken_spot_symbol,
    normalize_kraken_futures_pair,
    to_kraken_futures_symbol,
    dt_to_sec_utc,
)

START = datetime(2024, 1, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kraken_spot_response(timestamps_sec: list[int], pair: str = "XXBTZUSD", base_price: float = 100.0) -> dict:
    """Build a Kraken spot OHLC response."""
    rows = []
    for i, ts in enumerate(sorted(timestamps_sec)):
        p = base_price + i
        rows.append([
            ts,         # time (open time in seconds)
            str(p),     # open
            str(p + 1), # high
            str(p - 1), # low
            str(p + 0.5), # close
            str(p + 0.25), # vwap
            str(100.0 + i), # volume
            10 + i,     # count
        ])
    return {"error": [], "result": {pair: rows, "last": timestamps_sec[-1] if timestamps_sec else 0}}


def _make_kraken_futures_response(timestamps_sec: list[int], base_price: float = 100.0) -> dict:
    """Build a Kraken futures candles response."""
    candles = []
    for i, ts in enumerate(sorted(timestamps_sec)):
        p = base_price + i
        candles.append({
            "time": ts,
            "open": str(p),
            "high": str(p + 1),
            "low": str(p - 1),
            "close": str(p + 0.5),
            "volume": str(100.0 + i),
        })
    return {"candles": candles}


def _make_kraken_spot_pairs_response(pairs: dict[str, dict]) -> dict:
    """Build Kraken spot asset pairs response."""
    return {"error": [], "result": pairs}


def _make_kraken_futures_tickers_response(tickers: list[dict]) -> dict:
    """Build Kraken futures tickers response."""
    return {"result": "success", "tickers": tickers}


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


class TestKrakenSpotPairNormalization:
    def test_xxbtzusd_to_btcusd(self) -> None:
        assert normalize_kraken_spot_pair("XXBTZUSD") == "BTCUSD"

    def test_xethzusd_to_ethusd(self) -> None:
        assert normalize_kraken_spot_pair("XETHZUSD") == "ETHUSD"

    def test_xxbtzeur_to_btceur(self) -> None:
        assert normalize_kraken_spot_pair("XXBTZEUR") == "BTCEUR"

    def test_btcusd_to_xxbtzusd(self) -> None:
        assert to_kraken_spot_symbol("BTCUSD") == "XXBTZUSD"

    def test_ethusd_to_xethzusd(self) -> None:
        assert to_kraken_spot_symbol("ETHUSD") == "XETHZUSD"


class TestKrakenFuturesPairNormalization:
    def test_pi_xbtusd_to_btcusd(self) -> None:
        assert normalize_kraken_futures_pair("PI_XBTUSD") == "BTCUSD"

    def test_pi_ethusd_to_ethusd(self) -> None:
        assert normalize_kraken_futures_pair("PI_ETHUSD") == "ETHUSD"

    def test_lowercase_pi_xbtusd(self) -> None:
        assert normalize_kraken_futures_pair("pi_xbtusd") == "BTCUSD"

    def test_pf_prefix(self) -> None:
        assert normalize_kraken_futures_pair("PF_XBTUSD") == "BTCUSD"

    def test_btcusd_to_pi_xbtusd(self) -> None:
        assert to_kraken_futures_symbol("BTCUSD") == "pi_xbtusd"

    def test_ethusd_to_pi_ethusd(self) -> None:
        assert to_kraken_futures_symbol("ETHUSD") == "pi_ethusd"


# ---------------------------------------------------------------------------
# KrakenClient tests
# ---------------------------------------------------------------------------


class TestKrakenClient:
    def test_raises_without_context_manager(self) -> None:
        client = KrakenClient()
        with pytest.raises(RuntimeError, match="async context manager"):
            asyncio.run(client.get_spot_klines("XXBTZUSD"))

    def test_get_spot_pairs_raises_without_context_manager(self) -> None:
        client = KrakenClient()
        with pytest.raises(RuntimeError, match="async context manager"):
            asyncio.run(client.get_spot_pairs())

    def test_unsupported_spot_timeframe(self) -> None:
        async def _run():
            async with KrakenClient() as client:
                with pytest.raises(ValueError, match="Unsupported timeframe"):
                    await client.get_spot_klines("XXBTZUSD", timeframe="2h")

        asyncio.run(_run())

    def test_get_spot_klines_parses_response(self) -> None:
        ts1 = dt_to_sec_utc(START)
        ts2 = dt_to_sec_utc(START + timedelta(minutes=1))
        ts3 = dt_to_sec_utc(START + timedelta(minutes=2))

        body = _make_kraken_spot_response([ts1, ts2, ts3])
        mock_resp = _mock_response(body)

        async def _run():
            async with KrakenClient() as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                klines = await client.get_spot_klines("XXBTZUSD", timeframe="1m")

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

    def test_get_spot_klines_shifts_to_close_time(self) -> None:
        t1_sec = dt_to_sec_utc(START)
        body = _make_kraken_spot_response([t1_sec])
        mock_resp = _mock_response(body)

        async def _run():
            async with KrakenClient() as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                klines = await client.get_spot_klines("XXBTZUSD", timeframe="1m")

                # timestamp should be close time (open + 1 minute)
                assert klines[0]["timestamp"] == START + timedelta(minutes=1)

        asyncio.run(_run())

    def test_get_spot_klines_api_error(self) -> None:
        body = {"error": ["EGeneral:Invalid arguments"]}
        mock_resp = _mock_response(body)

        async def _run():
            async with KrakenClient(max_retries=1) as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                with pytest.raises(RuntimeError, match="Kraken API error"):
                    await client.get_spot_klines("INVALID", timeframe="1m")

        asyncio.run(_run())

    def test_get_spot_klines_range_empty(self) -> None:
        async def _run():
            async with KrakenClient() as client:
                result = await client.get_spot_klines_range(
                    "XXBTZUSD",
                    "1m",
                    start_time=START,
                    end_time=START - timedelta(hours=1),
                )
                assert result == []

        asyncio.run(_run())


class TestKrakenClientFutures:
    def test_unsupported_futures_timeframe(self) -> None:
        async def _run():
            async with KrakenClient() as client:
                with pytest.raises(ValueError, match="Unsupported timeframe"):
                    await client.get_futures_klines("PI_XBTUSD", timeframe="2h")

        asyncio.run(_run())

    def test_get_futures_klines_parses_response(self) -> None:
        ts1 = dt_to_sec_utc(START + timedelta(minutes=1))
        ts2 = dt_to_sec_utc(START + timedelta(minutes=2))
        ts3 = dt_to_sec_utc(START + timedelta(minutes=3))

        body = _make_kraken_futures_response([ts1, ts2, ts3])
        mock_resp = _mock_response(body)

        async def _run():
            async with KrakenClient() as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                klines = await client.get_futures_klines("PI_XBTUSD", timeframe="1m")

                assert len(klines) == 3
                # Should be ascending order
                assert klines[0]["timestamp"] < klines[1]["timestamp"] < klines[2]["timestamp"]
                # Check structure
                for k in klines:
                    assert set(k.keys()) == {"timestamp", "open", "high", "low", "close", "volume", "trades"}
                    assert isinstance(k["timestamp"], datetime)

        asyncio.run(_run())

    def test_get_futures_klines_range_empty(self) -> None:
        async def _run():
            async with KrakenClient() as client:
                result = await client.get_futures_klines_range(
                    "PI_XBTUSD",
                    "1m",
                    start_time=START,
                    end_time=START - timedelta(hours=1),
                )
                assert result == []

        asyncio.run(_run())


class TestKrakenClientGetPairs:
    def test_get_spot_pairs(self) -> None:
        pairs = {
            "XXBTZUSD": {"status": "online", "quote": "ZUSD"},
            "XETHZUSD": {"status": "online", "quote": "ZUSD"},
            "OLDPAIR": {"status": "offline", "quote": "ZUSD"},
        }
        body = _make_kraken_spot_pairs_response(pairs)
        mock_resp = _mock_response(body)

        async def _run():
            async with KrakenClient() as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                result = await client.get_spot_pairs()
                assert "XXBTZUSD" in result
                assert "XETHZUSD" in result
                assert "OLDPAIR" not in result  # offline

        asyncio.run(_run())

    def test_get_futures_pairs(self) -> None:
        tickers = [
            {"symbol": "PI_XBTUSD"},
            {"symbol": "PI_ETHUSD"},
            {"symbol": "FI_XBTUSD_230929"},
        ]
        body = _make_kraken_futures_tickers_response(tickers)
        mock_resp = _mock_response(body)

        async def _run():
            async with KrakenClient() as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                result = await client.get_futures_pairs()
                assert "PI_XBTUSD" in result
                assert "PI_ETHUSD" in result
                assert "FI_XBTUSD_230929" in result

        asyncio.run(_run())


class TestKrakenTimeframeMapping:
    def test_spot_timeframes_limited(self) -> None:
        # Kraken spot has limited interval support
        expected = {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}
        assert set(_KRAKEN_SPOT_INTERVAL_MAP.keys()) == expected

    def test_futures_timeframes(self) -> None:
        expected = {"1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d", "1w"}
        assert set(_KRAKEN_FUTURES_INTERVAL_MAP.keys()) == expected


# ---------------------------------------------------------------------------
# Component registration tests
# ---------------------------------------------------------------------------


class TestKrakenRegistration:
    def test_kraken_client_registered(self) -> None:
        cls = default_registry.get(SfComponentType.RAW_DATA_SOURCE, "kraken")
        assert cls is KrakenClient

    def test_kraken_spot_loader_registered(self) -> None:
        cls = default_registry.get(SfComponentType.RAW_DATA_LOADER, "kraken/spot")
        assert cls is KrakenSpotLoader

    def test_kraken_futures_loader_registered(self) -> None:
        cls = default_registry.get(SfComponentType.RAW_DATA_LOADER, "kraken/futures")
        assert cls is KrakenFuturesLoader


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


class TestKrakenLoaderGetPairs:
    @patch.object(KrakenClient, "get_spot_pairs")
    def test_spot_loader_get_pairs(self, mock_get_pairs: AsyncMock) -> None:
        mock_get_pairs.return_value = ["XXBTZUSD", "XETHZUSD"]

        async def _run():
            loader = KrakenSpotLoader()
            pairs = await loader.get_pairs(quote="USD")
            assert pairs == ["XXBTZUSD", "XETHZUSD"]

        asyncio.run(_run())
        mock_get_pairs.assert_called_once_with(quote="USD")

    @patch.object(KrakenClient, "get_futures_pairs")
    def test_futures_loader_get_pairs(self, mock_get_pairs: AsyncMock) -> None:
        mock_get_pairs.return_value = ["PI_XBTUSD", "PI_ETHUSD"]

        async def _run():
            loader = KrakenFuturesLoader()
            pairs = await loader.get_pairs()
            assert pairs == ["PI_XBTUSD", "PI_ETHUSD"]

        asyncio.run(_run())
        mock_get_pairs.assert_called_once_with(settlement=None)
