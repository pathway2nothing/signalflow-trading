"""Tests for GridMetrics — grid-specific strategy metrics."""

from datetime import datetime

from signalflow.analytic.strategy.grid_metrics import GridMetrics
from signalflow.core.containers.position import Position
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.core.enums import PositionType


def _state() -> StrategyState:
    state = StrategyState(strategy_id="test")
    state.portfolio.cash = 10000.0
    return state


def _add_position(
    state: StrategyState,
    *,
    pair: str = "BTCUSDT",
    entry_price: float = 100.0,
    qty: float = 1.0,
    position_type: PositionType = PositionType.LONG,
    is_closed: bool = False,
    realized_pnl: float = 0.0,
) -> Position:
    pos_id = f"pos_{pair}_{entry_price}_{len(state.portfolio.positions)}"
    pos = Position(
        id=pos_id,
        pair=pair,
        position_type=position_type,
        entry_price=entry_price,
        last_price=entry_price,
        qty=qty,
        is_closed=is_closed,
        realized_pnl=realized_pnl,
        entry_time=datetime(2024, 1, 1),
        last_time=datetime(2024, 1, 1),
    )
    state.portfolio.positions[pos.id] = pos
    return pos


class TestGridMetricsEmpty:
    def test_no_positions_returns_zeroes(self):
        m = GridMetrics()
        result = m.compute(_state(), {})
        assert result["grid_open_levels"] == 0.0
        assert result["grid_closed_levels"] == 0.0
        assert result["grid_total_levels"] == 0.0
        assert result["grid_total_pnl"] == 0.0
        assert result["grid_efficiency"] == 0.0
        assert result["grid_price_spread"] == 0.0
        assert result["grid_avg_entry_price"] == 0.0


class TestGridMetricsCounts:
    def test_open_positions_counted(self):
        m = GridMetrics()
        s = _state()
        _add_position(s, entry_price=100.0)
        _add_position(s, entry_price=95.0)
        result = m.compute(s, {"BTCUSDT": 100.0})
        assert result["grid_open_levels"] == 2.0
        assert result["grid_closed_levels"] == 0.0
        assert result["grid_total_levels"] == 2.0

    def test_closed_positions_counted(self):
        m = GridMetrics()
        s = _state()
        _add_position(s, entry_price=100.0, is_closed=True, realized_pnl=5.0)
        _add_position(s, entry_price=95.0, is_closed=True, realized_pnl=-2.0)
        result = m.compute(s, {})
        assert result["grid_open_levels"] == 0.0
        assert result["grid_closed_levels"] == 2.0
        assert result["grid_total_levels"] == 2.0

    def test_mixed_open_closed(self):
        m = GridMetrics()
        s = _state()
        _add_position(s, entry_price=100.0)
        _add_position(s, entry_price=95.0, is_closed=True, realized_pnl=3.0)
        result = m.compute(s, {"BTCUSDT": 105.0})
        assert result["grid_open_levels"] == 1.0
        assert result["grid_closed_levels"] == 1.0
        assert result["grid_total_levels"] == 2.0


class TestGridMetricsPnL:
    def test_open_position_unrealized_pnl(self):
        m = GridMetrics()
        s = _state()
        _add_position(s, entry_price=100.0, qty=2.0)
        # LONG: (110 - 100) * 2 = 20
        result = m.compute(s, {"BTCUSDT": 110.0})
        assert result["grid_total_pnl"] == 20.0
        assert result["grid_avg_pnl_per_level"] == 20.0

    def test_closed_position_realized_pnl(self):
        m = GridMetrics()
        s = _state()
        _add_position(s, entry_price=100.0, is_closed=True, realized_pnl=15.0)
        _add_position(s, entry_price=95.0, is_closed=True, realized_pnl=-5.0)
        result = m.compute(s, {})
        assert result["grid_total_pnl"] == 10.0
        assert result["grid_avg_pnl_per_level"] == 5.0
        assert result["grid_best_level_pnl"] == 15.0
        assert result["grid_worst_level_pnl"] == -5.0

    def test_short_position_pnl(self):
        m = GridMetrics()
        s = _state()
        _add_position(s, entry_price=100.0, qty=1.0, position_type=PositionType.SHORT)
        # SHORT: (100 - 90) * 1 = 10
        result = m.compute(s, {"BTCUSDT": 90.0})
        assert result["grid_total_pnl"] == 10.0

    def test_mixed_pnl(self):
        m = GridMetrics()
        s = _state()
        _add_position(s, entry_price=100.0, qty=1.0)  # open LONG
        _add_position(s, entry_price=90.0, is_closed=True, realized_pnl=8.0)  # closed
        # Open: (105 - 100) * 1 = 5; Closed: 8
        result = m.compute(s, {"BTCUSDT": 105.0})
        assert result["grid_total_pnl"] == 13.0


class TestGridMetricsEfficiency:
    def test_capital_deployed(self):
        m = GridMetrics()
        s = _state()
        _add_position(s, entry_price=100.0, qty=2.0)
        _add_position(s, entry_price=95.0, qty=1.0)
        result = m.compute(s, {"BTCUSDT": 100.0})
        assert result["grid_capital_deployed"] == 100.0 * 2.0 + 95.0 * 1.0

    def test_efficiency_positive(self):
        m = GridMetrics()
        s = _state()
        _add_position(s, entry_price=100.0, qty=1.0)
        # PnL = 10, capital = 100 → efficiency = 0.10
        result = m.compute(s, {"BTCUSDT": 110.0})
        assert abs(result["grid_efficiency"] - 0.10) < 1e-9

    def test_efficiency_zero_capital(self):
        m = GridMetrics()
        s = _state()
        # Only closed positions → no capital deployed
        _add_position(s, entry_price=100.0, is_closed=True, realized_pnl=5.0)
        result = m.compute(s, {})
        assert result["grid_capital_deployed"] == 0.0
        assert result["grid_efficiency"] == 0.0


class TestGridMetricsSpread:
    def test_single_position_no_spread(self):
        m = GridMetrics()
        s = _state()
        _add_position(s, entry_price=100.0)
        result = m.compute(s, {"BTCUSDT": 100.0})
        assert result["grid_price_spread"] == 0.0

    def test_two_positions_spread(self):
        m = GridMetrics()
        s = _state()
        _add_position(s, entry_price=100.0)
        _add_position(s, entry_price=90.0)
        # spread = (100 - 90) / 90 ≈ 0.1111
        result = m.compute(s, {"BTCUSDT": 100.0})
        assert abs(result["grid_price_spread"] - 10.0 / 90.0) < 1e-9

    def test_weighted_avg_entry_price(self):
        m = GridMetrics()
        s = _state()
        _add_position(s, entry_price=100.0, qty=2.0)
        _add_position(s, entry_price=90.0, qty=1.0)
        # avg = (100*2 + 90*1) / 3 = 290/3 ≈ 96.667
        result = m.compute(s, {"BTCUSDT": 100.0})
        assert abs(result["grid_avg_entry_price"] - 290.0 / 3.0) < 1e-9

    def test_closed_positions_excluded_from_spread(self):
        m = GridMetrics()
        s = _state()
        _add_position(s, entry_price=100.0)
        _add_position(s, entry_price=50.0, is_closed=True, realized_pnl=10.0)
        # Only one open position → no spread
        result = m.compute(s, {"BTCUSDT": 100.0})
        assert result["grid_price_spread"] == 0.0
