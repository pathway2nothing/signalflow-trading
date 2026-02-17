"""Tests for VirtualRealtimeBroker."""

from __future__ import annotations

from datetime import datetime

import pytest

from signalflow.core.containers.order import Order
from signalflow.core.containers.strategy_state import StrategyState
from signalflow.data.strategy_store import InMemoryStrategyStore
from signalflow.strategy.broker.executor.virtual_spot import VirtualSpotExecutor
from signalflow.strategy.broker.virtual_broker import VirtualRealtimeBroker


@pytest.fixture
def strategy_store() -> InMemoryStrategyStore:
    store = InMemoryStrategyStore()
    store.init()
    return store


@pytest.fixture
def broker(strategy_store: InMemoryStrategyStore) -> VirtualRealtimeBroker:
    return VirtualRealtimeBroker(
        executor=VirtualSpotExecutor(fee_rate=0.001, slippage_pct=0.0),
        store=strategy_store,
    )


@pytest.fixture
def state() -> StrategyState:
    state = StrategyState(strategy_id="test")
    state.portfolio.cash = 10000.0
    return state


class TestOrderLogging:
    def test_buy_order_logged(self, broker: VirtualRealtimeBroker, state: StrategyState) -> None:
        order = Order(
            pair="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            qty=0.1,
            signal_strength=0.85,
        )
        prices = {"BTCUSDT": 45000.0}
        ts = datetime(2024, 1, 1, 12, 0, 0)

        broker.submit_orders([order], prices, ts)

        assert len(broker.order_log) == 1
        logged = broker.order_log[0]
        assert logged["order_id"] == order.id
        assert logged["pair"] == "BTCUSDT"
        assert logged["side"] == "BUY"
        assert logged["qty"] == 0.1
        assert logged["price"] == 45000.0

    def test_sell_order_logged(self, broker: VirtualRealtimeBroker, state: StrategyState) -> None:
        order = Order(
            pair="ETHUSDT",
            side="SELL",
            order_type="MARKET",
            qty=2.5,
            signal_strength=0.75,
        )
        prices = {"ETHUSDT": 3000.0}
        ts = datetime(2024, 1, 1, 12, 0, 0)

        broker.submit_orders([order], prices, ts)

        assert len(broker.order_log) == 1
        logged = broker.order_log[0]
        assert logged["side"] == "SELL"
        assert logged["pair"] == "ETHUSDT"
        assert logged["qty"] == 2.5

    def test_empty_orders_no_log(self, broker: VirtualRealtimeBroker) -> None:
        fills = broker.submit_orders([], {}, datetime.now())
        assert len(fills) == 0
        assert len(broker.order_log) == 0

    def test_multiple_orders_logged(self, broker: VirtualRealtimeBroker) -> None:
        orders = [
            Order(pair="BTCUSDT", side="BUY", qty=0.1),
            Order(pair="ETHUSDT", side="BUY", qty=1.0),
        ]
        prices = {"BTCUSDT": 45000.0, "ETHUSDT": 3000.0}
        ts = datetime.now()

        broker.submit_orders(orders, prices, ts)

        assert len(broker.order_log) == 2
        assert broker.order_log[0]["pair"] == "BTCUSDT"
        assert broker.order_log[1]["pair"] == "ETHUSDT"


class TestFillLogging:
    def test_fill_logged_on_execution(self, broker: VirtualRealtimeBroker) -> None:
        order = Order(pair="BTCUSDT", side="BUY", qty=0.5)
        prices = {"BTCUSDT": 45000.0}
        ts = datetime.now()

        fills = broker.submit_orders([order], prices, ts)

        assert len(fills) == 1
        assert len(broker.fill_log) == 1
        logged = broker.fill_log[0]
        assert logged["pair"] == "BTCUSDT"
        assert logged["side"] == "BUY"
        assert logged["qty"] == 0.5
        assert logged["price"] == 45000.0

    def test_fill_details_correct(self, broker: VirtualRealtimeBroker) -> None:
        order = Order(pair="ETHUSDT", side="SELL", qty=1.0)
        prices = {"ETHUSDT": 3000.0}
        ts = datetime(2024, 1, 1, 12, 30, 0)

        fills = broker.submit_orders([order], prices, ts)

        logged = broker.fill_log[0]
        assert logged["fill_id"] == fills[0].id
        assert logged["order_id"] == order.id
        assert logged["fee"] > 0
        assert logged["ts"] == ts


class TestProcessFillsLogging:
    def test_entry_trade_logged(self, broker: VirtualRealtimeBroker, state: StrategyState) -> None:
        order = Order(pair="BTCUSDT", side="BUY", qty=0.1)
        prices = {"BTCUSDT": 45000.0}
        ts = datetime.now()

        fills = broker.submit_orders([order], prices, ts)
        trades = broker.process_fills(fills, [order], state)

        assert len(trades) == 1
        assert trades[0].side == "BUY"
        assert trades[0].pair == "BTCUSDT"

    def test_exit_trade_logged(self, broker: VirtualRealtimeBroker, state: StrategyState) -> None:
        # Create entry
        entry_order = Order(pair="BTCUSDT", side="BUY", qty=0.1)
        prices = {"BTCUSDT": 45000.0}
        ts1 = datetime(2024, 1, 1, 12, 0, 0)

        entry_fills = broker.submit_orders([entry_order], prices, ts1)
        broker.process_fills(entry_fills, [entry_order], state)

        # Create exit
        position = next(iter(state.portfolio.open_positions()))
        exit_order = Order(pair="BTCUSDT", side="SELL", qty=0.1, position_id=position.id)
        ts2 = datetime(2024, 1, 1, 13, 0, 0)
        prices_exit = {"BTCUSDT": 46000.0}

        exit_fills = broker.submit_orders([exit_order], prices_exit, ts2)
        exit_trades = broker.process_fills(exit_fills, [exit_order], state)

        assert len(exit_trades) == 1
        assert exit_trades[0].side == "SELL"
        assert exit_trades[0].meta["type"] == "exit"


class TestLedgerDataFrames:
    def test_order_log_df_empty(self, broker: VirtualRealtimeBroker) -> None:
        df = broker.order_log_df()
        assert df.height == 0

    def test_fill_log_df_empty(self, broker: VirtualRealtimeBroker) -> None:
        df = broker.fill_log_df()
        assert df.height == 0

    def test_equity_curve_df_empty(self, broker: VirtualRealtimeBroker) -> None:
        df = broker.equity_curve_df()
        assert df.height == 0

    def test_equity_curve_df_populated(self, broker: VirtualRealtimeBroker, state: StrategyState) -> None:
        prices = {"BTCUSDT": 45000.0}
        ts = datetime(2024, 1, 1, 12, 0, 0)
        broker.mark_positions(state, prices, ts)

        df = broker.equity_curve_df()
        assert df.height == 1
        assert "ts" in df.columns
        assert "equity" in df.columns
        assert "cash" in df.columns

    def test_order_log_df_populated(self, broker: VirtualRealtimeBroker) -> None:
        orders = [
            Order(pair="BTCUSDT", side="BUY", qty=0.1),
            Order(pair="ETHUSDT", side="SELL", qty=1.0),
        ]
        prices = {"BTCUSDT": 45000.0, "ETHUSDT": 3000.0}
        broker.submit_orders(orders, prices, datetime.now())

        df = broker.order_log_df()
        assert df.height == 2
        assert "pair" in df.columns
        assert "side" in df.columns
        assert "qty" in df.columns

    def test_fill_log_df_populated(self, broker: VirtualRealtimeBroker) -> None:
        order = Order(pair="BTCUSDT", side="BUY", qty=0.5)
        prices = {"BTCUSDT": 45000.0}
        broker.submit_orders([order], prices, datetime.now())

        df = broker.fill_log_df()
        assert df.height == 1
        assert "fill_id" in df.columns
        assert "order_id" in df.columns
        assert "price" in df.columns
        assert "fee" in df.columns


class TestRiskManagerIntegration:
    """Tests for risk manager integration."""

    def test_risk_manager_rejects_orders(self, strategy_store: InMemoryStrategyStore, state: StrategyState) -> None:
        """Test that risk manager can reject orders."""
        from unittest.mock import MagicMock

        # Create mock risk manager that rejects all orders
        mock_risk_manager = MagicMock()
        mock_result = MagicMock()
        mock_result.allowed = False
        mock_result.passed_orders = []
        mock_result.violations = [("TestLimit", "Order rejected for test")]
        mock_risk_manager.check.return_value = mock_result

        broker = VirtualRealtimeBroker(
            executor=VirtualSpotExecutor(fee_rate=0.001, slippage_pct=0.0),
            store=strategy_store,
            risk_manager=mock_risk_manager,
        )
        broker._last_state = state  # Set last state for risk check

        order = Order(pair="BTCUSDT", side="BUY", qty=0.1)
        prices = {"BTCUSDT": 45000.0}
        ts = datetime(2024, 1, 1, 12, 0, 0)

        fills = broker.submit_orders([order], prices, ts)

        assert len(fills) == 0
        assert len(broker.order_log) == 0

    def test_risk_manager_allows_partial_orders(self, strategy_store: InMemoryStrategyStore, state: StrategyState) -> None:
        """Test that risk manager can pass some orders."""
        from unittest.mock import MagicMock

        # Create broker with mock risk manager
        mock_risk_manager = MagicMock()
        mock_result = MagicMock()
        mock_result.allowed = True

        broker = VirtualRealtimeBroker(
            executor=VirtualSpotExecutor(fee_rate=0.001, slippage_pct=0.0),
            store=strategy_store,
            risk_manager=mock_risk_manager,
        )
        broker._last_state = state

        order1 = Order(pair="BTCUSDT", side="BUY", qty=0.1)
        order2 = Order(pair="ETHUSDT", side="BUY", qty=1.0)

        # Mock returns only the first order as passed
        mock_result.passed_orders = [order1]
        mock_risk_manager.check.return_value = mock_result

        prices = {"BTCUSDT": 45000.0, "ETHUSDT": 3000.0}
        ts = datetime(2024, 1, 1, 12, 0, 0)

        fills = broker.submit_orders([order1, order2], prices, ts)

        assert len(fills) == 1  # Only one order passed
        assert len(broker.order_log) == 1


class TestInheritedBehavior:
    def test_position_created(self, broker: VirtualRealtimeBroker, state: StrategyState) -> None:
        order = Order(pair="BTCUSDT", side="BUY", qty=0.1)
        prices = {"BTCUSDT": 45000.0}
        ts = datetime.now()

        fills = broker.submit_orders([order], prices, ts)
        broker.process_fills(fills, [order], state)

        assert len(state.portfolio.open_positions()) == 1
        position = next(iter(state.portfolio.open_positions()))
        assert position.pair == "BTCUSDT"
        assert position.qty == 0.1

    def test_cash_updated(self, broker: VirtualRealtimeBroker, state: StrategyState) -> None:
        initial_cash = state.portfolio.cash
        order = Order(pair="BTCUSDT", side="BUY", qty=0.1)
        prices = {"BTCUSDT": 45000.0}
        ts = datetime.now()

        fills = broker.submit_orders([order], prices, ts)
        broker.process_fills(fills, [order], state)

        # Cash should decrease by notional + fee
        assert state.portfolio.cash < initial_cash
        spent = initial_cash - state.portfolio.cash
        expected = 45000.0 * 0.1 * (1 + 0.001)  # price * qty * (1 + fee_rate)
        assert abs(spent - expected) < 0.01

    def test_mark_positions(self, broker: VirtualRealtimeBroker, state: StrategyState) -> None:
        # Create position
        order = Order(pair="BTCUSDT", side="BUY", qty=0.1)
        prices = {"BTCUSDT": 45000.0}
        ts1 = datetime(2024, 1, 1, 12, 0, 0)

        fills = broker.submit_orders([order], prices, ts1)
        broker.process_fills(fills, [order], state)

        # Mark at new price
        new_prices = {"BTCUSDT": 46000.0}
        ts2 = datetime(2024, 1, 1, 13, 0, 0)
        broker.mark_positions(state, new_prices, ts2)

        position = next(iter(state.portfolio.open_positions()))
        assert position.last_price == 46000.0
        assert position.last_time == ts2
