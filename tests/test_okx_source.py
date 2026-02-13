"""Tests for OKX data source - OkxClient, OkxSpotLoader, OkxFuturesLoader."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from signalflow.core.enums import SfComponentType
from signalflow.core.registry import default_registry
from signalflow.data.source.okx import (
    OkxClient,
    OkxFuturesLoader,
    OkxSpotLoader,
    _OKX_BAR_MAP,
    _to_okx_inst_id,
)
from signalflow.data.source._helpers import dt_to_ms_utc, ms_to_dt_utc_naive

START = datetime(2024, 1, 1)
PAIR = "BTCUSDT"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_okx_response(timestamps_ms: list[int], base_price: float = 100.0) -> dict:
    """Build an OKX-style JSON response body (descending order)."""
    rows = []
    for i, ts in enumerate(sorted(timestamps_ms, reverse=True)):
        p = base_price + i
        rows.append(
            [
                str(ts),  # timestamp ms
                str(p),  # open
                str(p + 1),  # high
                str(p - 1),  # low
                str(p + 0.5),  # close
                str(100.0 + i),  # vol
                str(50000.0 + i),  # volCcy
                str(50000.0 + i),  # volCcyQuote
                "1",  # confirm
            ]
        )
    return {"code": "0", "msg": "", "data": rows}


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
# OKX instrument ID conversion tests
# ---------------------------------------------------------------------------


class TestOkxInstIdConversion:
    def test_btcusdt_spot(self) -> None:
        assert _to_okx_inst_id("BTCUSDT") == "BTC-USDT"

    def test_ethbtc_spot(self) -> None:
        assert _to_okx_inst_id("ETHBTC") == "ETH-BTC"

    def test_ethusdc_spot(self) -> None:
        assert _to_okx_inst_id("ETHUSDC") == "ETH-USDC"

    def test_btcusdt_swap(self) -> None:
        assert _to_okx_inst_id("BTCUSDT", suffix="-SWAP") == "BTC-USDT-SWAP"

    def test_btcusdt_dated_futures(self) -> None:
        assert _to_okx_inst_id("BTCUSDT", suffix="-240329") == "BTC-USDT-240329"

    def test_unknown_quote_fallback(self) -> None:
        assert _to_okx_inst_id("XYZABC") == "XYZABC"

    def test_unknown_quote_with_suffix(self) -> None:
        assert _to_okx_inst_id("XYZABC", suffix="-SWAP") == "XYZABC-SWAP"


# ---------------------------------------------------------------------------
# OkxClient tests
# ---------------------------------------------------------------------------


class TestOkxClient:
    def test_raises_without_context_manager(self) -> None:
        client = OkxClient()
        with pytest.raises(RuntimeError, match="async context manager"):
            asyncio.run(client.get_klines("BTC-USDT"))

    def test_get_pairs_raises_without_context_manager(self) -> None:
        client = OkxClient()
        with pytest.raises(RuntimeError, match="async context manager"):
            asyncio.run(client.get_pairs())

    def test_unsupported_timeframe(self) -> None:
        async def _run():
            async with OkxClient() as client:
                with pytest.raises(ValueError, match="Unsupported timeframe"):
                    await client.get_klines("BTC-USDT", timeframe="2m")

        asyncio.run(_run())

    def test_get_klines_parses_response(self) -> None:
        ts1 = dt_to_ms_utc(START)
        ts2 = dt_to_ms_utc(START + timedelta(minutes=1))
        ts3 = dt_to_ms_utc(START + timedelta(minutes=2))

        body = _make_okx_response([ts1, ts2, ts3])
        mock_resp = _mock_response(body)

        async def _run():
            async with OkxClient() as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                klines = await client.get_klines("BTC-USDT", timeframe="1m")

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

        body = _make_okx_response([dt_to_ms_utc(t1), dt_to_ms_utc(t2), dt_to_ms_utc(t3)])
        mock_resp = _mock_response(body)

        async def _run():
            async with OkxClient() as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                klines = await client.get_klines("BTC-USDT", timeframe="1m")

                # timestamps are shifted to close time (open + 1 tf)
                assert klines[0]["timestamp"] == t1 + timedelta(minutes=1)
                assert klines[2]["timestamp"] == t3 + timedelta(minutes=1)

        asyncio.run(_run())

    def test_get_klines_api_error(self) -> None:
        body = {"code": "51001", "msg": "Instrument ID does not exist"}
        mock_resp = _mock_response(body)

        async def _run():
            async with OkxClient(max_retries=1) as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                with pytest.raises(RuntimeError, match="OKX API error"):
                    await client.get_klines("INVALID-PAIR", timeframe="1m")

        asyncio.run(_run())

    def test_get_klines_range_empty(self) -> None:
        async def _run():
            async with OkxClient() as client:
                result = await client.get_klines_range(
                    "BTC-USDT",
                    "1m",
                    start_time=START,
                    end_time=START - timedelta(hours=1),
                )
                assert result == []

        asyncio.run(_run())

    def test_get_klines_range_unsupported_timeframe(self) -> None:
        async def _run():
            async with OkxClient() as client:
                with pytest.raises(ValueError, match="Unsupported timeframe"):
                    await client.get_klines_range(
                        "BTC-USDT",
                        "2m",
                        start_time=START,
                        end_time=START + timedelta(hours=1),
                    )

        asyncio.run(_run())


class TestOkxTimeframeMapping:
    def test_all_signalflow_timeframes_mapped(self) -> None:
        expected = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"}
        assert set(_OKX_BAR_MAP.keys()) == expected

    def test_minute_intervals_lowercase(self) -> None:
        assert _OKX_BAR_MAP["1m"] == "1m"
        assert _OKX_BAR_MAP["5m"] == "5m"
        assert _OKX_BAR_MAP["30m"] == "30m"

    def test_hour_intervals_uppercase(self) -> None:
        assert _OKX_BAR_MAP["1h"] == "1H"
        assert _OKX_BAR_MAP["4h"] == "4H"
        assert _OKX_BAR_MAP["12h"] == "12H"

    def test_day_interval_uppercase(self) -> None:
        assert _OKX_BAR_MAP["1d"] == "1D"


# ---------------------------------------------------------------------------
# Component registration tests
# ---------------------------------------------------------------------------


class TestOkxRegistration:
    def test_okx_client_registered(self) -> None:
        cls = default_registry.get(SfComponentType.RAW_DATA_SOURCE, "okx")
        assert cls is OkxClient

    def test_okx_spot_loader_registered(self) -> None:
        cls = default_registry.get(SfComponentType.RAW_DATA_LOADER, "okx/spot")
        assert cls is OkxSpotLoader

    def test_okx_futures_loader_registered(self) -> None:
        cls = default_registry.get(SfComponentType.RAW_DATA_LOADER, "okx/futures")
        assert cls is OkxFuturesLoader

    def test_futures_loader_default_suffix(self) -> None:
        assert OkxFuturesLoader.inst_suffix == "-SWAP"


# ---------------------------------------------------------------------------
# get_pairs() tests
# ---------------------------------------------------------------------------


def _make_okx_instruments_response(instruments: list[dict]) -> dict:
    """Build OKX instruments response."""
    return {"code": "0", "msg": "", "data": instruments}


class TestOkxClientGetPairs:
    def test_get_pairs_spot(self) -> None:
        instruments = [
            {"instId": "BTC-USDT", "quoteCcy": "USDT", "state": "live"},
            {"instId": "ETH-USDT", "quoteCcy": "USDT", "state": "live"},
            {"instId": "ETH-BTC", "quoteCcy": "BTC", "state": "live"},
            {"instId": "OLD-USDT", "quoteCcy": "USDT", "state": "suspend"},
        ]
        body = _make_okx_instruments_response(instruments)
        mock_resp = _mock_response(body)

        async def _run():
            async with OkxClient() as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                # All pairs
                pairs = await client.get_pairs(inst_type="SPOT")
                assert pairs == ["BTC-USDT", "ETH-BTC", "ETH-USDT"]
                assert "OLD-USDT" not in pairs  # suspended

        asyncio.run(_run())

    def test_get_pairs_spot_quote_filter(self) -> None:
        instruments = [
            {"instId": "BTC-USDT", "quoteCcy": "USDT", "state": "live"},
            {"instId": "ETH-USDT", "quoteCcy": "USDT", "state": "live"},
            {"instId": "ETH-BTC", "quoteCcy": "BTC", "state": "live"},
        ]
        body = _make_okx_instruments_response(instruments)
        mock_resp = _mock_response(body)

        async def _run():
            async with OkxClient() as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                pairs = await client.get_pairs(inst_type="SPOT", quote="USDT")
                assert pairs == ["BTC-USDT", "ETH-USDT"]
                assert "ETH-BTC" not in pairs

        asyncio.run(_run())

    def test_get_pairs_swap(self) -> None:
        instruments = [
            {"instId": "BTC-USDT-SWAP", "settleCcy": "USDT", "state": "live"},
            {"instId": "ETH-USDT-SWAP", "settleCcy": "USDT", "state": "live"},
            {"instId": "BTC-USD-SWAP", "settleCcy": "BTC", "state": "live"},
        ]
        body = _make_okx_instruments_response(instruments)
        mock_resp = _mock_response(body)

        async def _run():
            async with OkxClient() as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                # Filter by USDT settlement
                pairs = await client.get_pairs(inst_type="SWAP", quote="USDT")
                assert pairs == ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]

        asyncio.run(_run())

    def test_get_pairs_api_error(self) -> None:
        body = {"code": "50001", "msg": "Service unavailable"}
        mock_resp = _mock_response(body)

        async def _run():
            async with OkxClient(max_retries=1) as client:
                mock_session = AsyncMock()
                mock_session.get = MagicMock(return_value=mock_resp)
                client._session = mock_session

                with pytest.raises(RuntimeError, match="OKX API error"):
                    await client.get_pairs()

        asyncio.run(_run())


class TestOkxLoaderGetPairs:
    @patch.object(OkxClient, "get_pairs")
    def test_spot_loader_get_pairs(self, mock_get_pairs: AsyncMock) -> None:
        mock_get_pairs.return_value = ["BTC-USDT", "ETH-USDT"]

        async def _run():
            loader = OkxSpotLoader()
            pairs = await loader.get_pairs(quote="USDT")
            assert pairs == ["BTC-USDT", "ETH-USDT"]

        asyncio.run(_run())
        mock_get_pairs.assert_called_once_with(inst_type="SPOT", quote="USDT")

    @patch.object(OkxClient, "get_pairs")
    def test_futures_loader_get_pairs_swap(self, mock_get_pairs: AsyncMock) -> None:
        mock_get_pairs.return_value = ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]

        async def _run():
            loader = OkxFuturesLoader(inst_suffix="-SWAP")
            pairs = await loader.get_pairs(quote="USDT")
            assert pairs == ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]

        asyncio.run(_run())
        mock_get_pairs.assert_called_once_with(inst_type="SWAP", quote="USDT")

    @patch.object(OkxClient, "get_pairs")
    def test_futures_loader_get_pairs_futures(self, mock_get_pairs: AsyncMock) -> None:
        mock_get_pairs.return_value = ["BTC-USDT-240329"]

        async def _run():
            loader = OkxFuturesLoader(inst_suffix="-240329")
            pairs = await loader.get_pairs()
            assert pairs == ["BTC-USDT-240329"]

        asyncio.run(_run())
        mock_get_pairs.assert_called_once_with(inst_type="FUTURES", quote=None)
