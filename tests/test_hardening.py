"""WP7: brokers, live loop, LLM strategy and risk never fail silently and paper matches live sizing."""

import io
import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timedelta

import polars as pl
import pytest

import signalflow as sf
from signalflow.engine.engine import Engine
from signalflow.engine.types import Intent, Order, PortfolioSnapshot
from signalflow.enums import IntentKind, Side
from signalflow.flow.live import PollingFeed, load_state, run_live_loop
from signalflow.strategy.llm import LLMStrategy
from signalflow.strategy.observation import Observation

_EXCHANGE_INFO = {
    "symbols": [
        {
            "filters": [
                {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                {"filterType": "NOTIONAL", "minNotional": "10"},
            ]
        }
    ]
}
_FILLED = {
    "executedQty": "0.117",
    "cummulativeQuoteQty": "11700",
    "fills": [{"commission": "0.0001", "commissionAsset": "BTC"}],
}


class _Resp:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _bar(price: float = 100_000.0):
    frame = pl.DataFrame({"pair": ["BTCUSDT"], "ts": [datetime(2024, 1, 1)], "close": [price]})
    return sf.data.__globals__["Bar"](ts=datetime(2024, 1, 1), frame=frame, prices={"BTCUSDT": price})


def _install(monkeypatch, on_post, on_query=None):
    posts: list[str] = []

    def fake_urlopen(req, timeout=None):
        url = req.full_url
        if "exchangeInfo" in url:
            return _Resp(_EXCHANGE_INFO)
        if req.get_method() == "POST":
            posts.append(url)
            return on_post(len(posts))
        if on_query is not None:
            return on_query(url)
        raise AssertionError(f"unexpected request {url}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return posts


def test_broker_repr_hides_secrets():
    broker = sf.BinanceBroker(api_key="AKIA_KEY", api_secret="SUPER_SECRET")
    assert "SUPER_SECRET" not in repr(broker) and "AKIA_KEY" not in repr(broker)


def test_retry_re_signs_with_a_fresh_timestamp(monkeypatch):
    import signalflow.engine.broker as broker_module

    class _Clock:
        t = 1_700_000_000.0

        def time(self):
            _Clock.t += 1.0
            return _Clock.t

        def sleep(self, seconds):
            pass

    monkeypatch.setattr(broker_module, "time", _Clock())

    def on_post(n):
        if n == 1:
            raise urllib.error.URLError("connection reset")
        return _Resp(_FILLED)

    posts = _install(monkeypatch, on_post)
    broker = sf.BinanceBroker(api_key="k", api_secret="s", max_retries=2, retry_delay=0.0)
    fills = broker.execute([Order("BTCUSDT", Side.BUY, 0.117, ts="2024-01-01")], _bar())
    assert len(fills) == 1 and len(posts) == 2
    stamps = [re.search(r"timestamp=(\d+)", u).group(1) for u in posts]
    assert stamps[0] != stamps[1], "a retried send must carry a fresh timestamp/signature"


def test_failed_send_is_resolved_by_querying_the_venue(monkeypatch):
    def on_post(n):
        raise urllib.error.HTTPError("url", 504, "gateway timeout", hdrs=None, fp=io.BytesIO(b"timeout"))

    def on_query(url):
        assert "origClientOrderId=" in url
        return _Resp(_FILLED)

    _install(monkeypatch, on_post, on_query)
    broker = sf.BinanceBroker(api_key="k", api_secret="s", max_retries=0, retry_delay=0.0)
    fills = broker.execute([Order("BTCUSDT", Side.BUY, 0.117, ts="2024-01-01")], _bar())
    assert len(fills) == 1 and fills[0].qty == pytest.approx(0.117)


def test_sim_broker_quantizes_like_the_venue():
    filters = {"BTCUSDT": {"stepSize": "0.001", "tickSize": "0.01", "minQty": "0.001", "minNotional": "10"}}
    broker = sf.SimBroker(filters=filters, slippage=0.0)
    bar = _bar(price=100_000.0)
    fills = broker.execute([Order("BTCUSDT", Side.BUY, 0.0123456, ts=bar.ts)], bar)
    assert fills[0].qty == pytest.approx(0.012)
    assert broker.execute([Order("BTCUSDT", Side.BUY, 0.00005, ts=bar.ts)], bar) == []  # below minNotional


@pytest.fixture(scope="module")
def ds():
    return sf.dataset("synthetic", pairs=["BTCUSDT", "ETHUSDT"], start="2023-01-01", end="2023-02-15", interval="1h")


def _cross_flow():
    return sf.Flow(name="x", detectors=[sf.SmaCrossDetector(fast=5, slow=20)], strategy=sf.RulesStrategy())


def test_next_open_fills_at_the_next_bars_open_and_keeps_parity(ds):
    flow = _cross_flow()
    bt = flow.backtest(ds, capital=10_000, broker=sf.SimBroker(fill="next_open"))
    sim = flow.simulate(ds, capital=10_000, broker=sf.SimBroker(fill="next_open"))
    assert bt.fills and len(bt.fills) == len(sim.fills)
    assert sim.final_equity == pytest.approx(bt.final_equity)
    opens = {(r["pair"], r["ts"]): r["open"] for r in ds.frame.select(["pair", "ts", "open"]).to_dicts()}
    slip = sf.SimBroker().slippage
    for f in bt.fills:
        ref = opens[(f.pair, f.ts)]
        assert f.price == pytest.approx(ref * (1 + slip) if f.side == Side.BUY else ref * (1 - slip))
    close_run = flow.backtest(ds, capital=10_000)
    assert {f.ts for f in bt.fills} != {f.ts for f in close_run.fills}  # deferred by one bar


def test_state_file_is_atomic_and_fills_are_journaled(ds, tmp_path):
    path = str(tmp_path / "book.json")
    run = run_live_loop(_cross_flow(), sf.ReplayFeed(ds), 10_000.0, sf.SimBroker(), state_path=path, max_bars=400)
    assert run.fills and not os.path.exists(path + ".tmp")
    journal = tmp_path / "book.fills.jsonl"
    assert journal.exists() and len(journal.read_text().splitlines()) == len(run.fills)
    restored = Engine(10_000.0)
    assert load_state(restored, path)
    assert len(restored.event_log) == len(run.fills)
    assert run.meta == {"skipped_bars": 0, "feed_errors": 0, "strategy_fallbacks": 0}


class _FlakySource:
    name = "flaky"

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0

    def fetch(self, pairs, start, end=None, interval="1m"):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("binance request failed")
        ts = datetime(2024, 1, 1, 0, 0)
        one = [1.0]
        return pl.DataFrame(
            {"pair": ["BTCUSDT"], "ts": [ts], "open": one, "high": one, "low": one, "close": one, "volume": one}
        )


class _Clock:
    def __init__(self, wall: float):
        self._wall = wall
        self.sleeps: list[float] = []

    def wall(self):
        return self._wall

    def now(self):
        return self._wall

    def sleep(self, seconds):
        self.sleeps.append(seconds)


def test_polling_feed_retries_transient_source_errors():
    now = datetime(2024, 1, 1, 0, 5).timestamp()
    src = _FlakySource(fail_times=2)
    feed = PollingFeed(
        src, ["BTCUSDT"], interval="1m", clock=_Clock(now), max_bars=1, fetch_retries=3, retry_backoff_s=0.5
    )
    bars = list(feed.stream())
    assert len(bars) == 1 and src.calls == 3 and feed.errors == 0
    assert 0.5 in feed.clock.sleeps and 1.0 in feed.clock.sleeps  # linear backoff between attempts

    giving_up = PollingFeed(
        _FlakySource(fail_times=99), ["BTCUSDT"], clock=_Clock(now), fetch_retries=2, retry_backoff_s=0
    )
    assert giving_up._fetch_with_retry(int(now) - 120).height == 0
    assert giving_up.errors == 1


def _observation():
    signals = pl.DataFrame({"pair": ["BTCUSDT"], "ts": [datetime(2024, 1, 1)], "signal": ["rise"]})
    snap = PortfolioSnapshot(datetime(2024, 1, 1), "USDT", {"USDT": 10_000.0}, {}, 10_000.0, {"BTCUSDT": 100.0})
    return Observation(datetime(2024, 1, 1), signals, snap)


class _BoomClient:
    def decide(self, context, schema):
        raise RuntimeError("server down")


def test_llm_fallback_is_counted_and_optional():
    counted = LLMStrategy(client=_BoomClient(), fallback=sf.RulesStrategy())
    intents = counted.decide(_observation())
    assert counted.fallbacks == 1 and intents  # the rules strategy opened on the RISE signal
    strict = LLMStrategy(client=_BoomClient(), fallback=None)
    with pytest.raises(sf.KillSwitchTripped):
        strict.decide(_observation())
    cfg = strict.to_config()
    assert cfg["params"]["fallback"] is None and LLMStrategy.from_config(cfg).fallback is None


def test_llm_cache_is_bounded():
    class _Ok:
        def decide(self, context, schema):
            return {"decisions": []}

    strat = LLMStrategy(client=_Ok(), cache_size=3)
    for i in range(10):
        obs = _observation()
        obs = Observation(datetime(2024, 1, 1) + timedelta(hours=i), obs.signals, obs.portfolio)
        strat.decide(obs)
    assert len(strat._cache) == 3


def test_risk_clip_does_not_mutate_intents():
    snap = PortfolioSnapshot(datetime(2024, 1, 1), "USDT", {"USDT": 10_000.0}, {}, 10_000.0, {})
    intent = Intent("BTCUSDT", IntentKind.OPEN, Side.BUY, notional=5_000.0)
    out = sf.Risk(max_notional_per_pair=0.1).clip([intent], snap, 10_000.0)
    assert out[0].notional == pytest.approx(1_000.0)
    assert intent.notional == 5_000.0
