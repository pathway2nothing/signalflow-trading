"""Extended strategy metrics for advanced performance analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from signalflow.analytic.base import StrategyMetric
from signalflow.core import StrategyState, sf_component


@dataclass
@sf_component(name="sortino_ratio", override=True)
class SortinoRatioMetric(StrategyMetric):
    """Computes Sortino ratio using only downside volatility."""

    initial_capital: float = 10000.0
    window_size: int = 100
    risk_free_rate: float = 0.0
    target_return: float = 0.0
    _returns_history: list[float] = None

    def __post_init__(self):
        self._returns_history = []

    def compute(self, state: StrategyState, prices: dict[str, float], **kwargs) -> dict[str, float]:
        equity = state.portfolio.equity(prices=prices)
        current_return = (equity - self.initial_capital) / self.initial_capital

        self._returns_history.append(current_return)

        if len(self._returns_history) > self.window_size:
            self._returns_history.pop(0)

        if len(self._returns_history) < 2:
            return {"sortino_ratio": 0.0}

        returns_array = np.array(self._returns_history)
        returns_diff = np.diff(returns_array)

        downside_returns = returns_diff[returns_diff < self.target_return]
        if len(downside_returns) < 2:
            return {"sortino_ratio": 0.0}

        downside_std = np.std(downside_returns)
        mean_return = np.mean(returns_diff)

        if downside_std == 0:
            return {"sortino_ratio": 0.0}

        sortino = (mean_return - self.risk_free_rate) / downside_std
        return {"sortino_ratio": sortino}


@dataclass
@sf_component(name="calmar_ratio", override=True)
class CalmarRatioMetric(StrategyMetric):
    """Computes Calmar ratio (return / max drawdown)."""

    initial_capital: float = 10000.0
    _peak_equity: float = 0.0
    _max_drawdown: float = 0.0
    _initial_equity: float = 0.0

    def compute(self, state: StrategyState, prices: dict[str, float], **kwargs) -> dict[str, float]:
        equity = state.portfolio.equity(prices=prices)

        if self._initial_equity == 0.0:
            self._initial_equity = equity

        if equity > self._peak_equity:
            self._peak_equity = equity

        current_dd = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0.0
        if current_dd > self._max_drawdown:
            self._max_drawdown = current_dd

        total_return = (equity - self._initial_equity) / self._initial_equity if self._initial_equity > 0 else 0.0
        calmar = total_return / self._max_drawdown if self._max_drawdown > 0 else 0.0

        return {
            "calmar_ratio": calmar,
            "annualized_return": total_return,
            "max_drawdown_calmar": self._max_drawdown,
        }


@dataclass
@sf_component(name="profit_factor", override=True)
class ProfitFactorMetric(StrategyMetric):
    """Computes profit factor (gross profit / gross loss)."""

    def compute(self, state: StrategyState, prices: dict[str, float], **kwargs) -> dict[str, float]:
        closed_positions = [p for p in state.portfolio.positions.values() if p.is_closed]

        if not closed_positions:
            return {"profit_factor": 0.0, "gross_profit": 0.0, "gross_loss": 0.0}

        gross_profit = sum(p.realized_pnl for p in closed_positions if p.realized_pnl > 0)
        gross_loss = abs(sum(p.realized_pnl for p in closed_positions if p.realized_pnl < 0))

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        return {
            "profit_factor": profit_factor,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
        }


@dataclass
@sf_component(name="average_trade", override=True)
class AverageTradeMetric(StrategyMetric):
    """Computes average profit, loss, and trade duration."""

    def compute(self, state: StrategyState, prices: dict[str, float], **kwargs) -> dict[str, float]:
        closed_positions = [p for p in state.portfolio.positions.values() if p.is_closed]

        if not closed_positions:
            return {
                "avg_profit": 0.0,
                "avg_loss": 0.0,
                "avg_trade": 0.0,
                "avg_duration_minutes": 0.0,
                "avg_win_duration": 0.0,
                "avg_loss_duration": 0.0,
            }

        winners = [p for p in closed_positions if p.realized_pnl > 0]
        losers = [p for p in closed_positions if p.realized_pnl <= 0]

        avg_profit = float(np.mean([p.realized_pnl for p in winners])) if winners else 0.0
        avg_loss = float(np.mean([p.realized_pnl for p in losers])) if losers else 0.0
        avg_trade = float(np.mean([p.realized_pnl for p in closed_positions]))

        def get_duration_minutes(pos) -> float:
            if pos.entry_time and pos.last_time:
                return (pos.last_time - pos.entry_time).total_seconds() / 60
            return 0.0

        all_durations = [get_duration_minutes(p) for p in closed_positions]
        win_durations = [get_duration_minutes(p) for p in winners]
        loss_durations = [get_duration_minutes(p) for p in losers]

        return {
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "avg_trade": avg_trade,
            "avg_duration_minutes": float(np.mean(all_durations)) if all_durations else 0.0,
            "avg_win_duration": float(np.mean(win_durations)) if win_durations else 0.0,
            "avg_loss_duration": float(np.mean(loss_durations)) if loss_durations else 0.0,
        }


@dataclass
@sf_component(name="expectancy", override=True)
class ExpectancyMetric(StrategyMetric):
    """Computes trade expectancy (win_rate * avg_win - loss_rate * avg_loss)."""

    def compute(self, state: StrategyState, prices: dict[str, float], **kwargs) -> dict[str, float]:
        closed_positions = [p for p in state.portfolio.positions.values() if p.is_closed]

        if not closed_positions:
            return {"expectancy": 0.0, "expectancy_ratio": 0.0}

        winners = [p for p in closed_positions if p.realized_pnl > 0]
        losers = [p for p in closed_positions if p.realized_pnl <= 0]

        win_rate = len(winners) / len(closed_positions)
        loss_rate = 1 - win_rate

        avg_win = float(np.mean([p.realized_pnl for p in winners])) if winners else 0.0
        avg_loss = abs(float(np.mean([p.realized_pnl for p in losers]))) if losers else 0.0

        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        expectancy_ratio = expectancy / avg_loss if avg_loss > 0 else 0.0

        return {
            "expectancy": expectancy,
            "expectancy_ratio": expectancy_ratio,
        }


@dataclass
@sf_component(name="risk_reward", override=True)
class RiskRewardMetric(StrategyMetric):
    """Computes risk/reward ratio (avg_win / avg_loss)."""

    def compute(self, state: StrategyState, prices: dict[str, float], **kwargs) -> dict[str, float]:
        closed_positions = [p for p in state.portfolio.positions.values() if p.is_closed]

        if not closed_positions:
            return {"risk_reward_ratio": 0.0, "payoff_ratio": 0.0}

        winners = [p for p in closed_positions if p.realized_pnl > 0]
        losers = [p for p in closed_positions if p.realized_pnl < 0]

        avg_win = float(np.mean([p.realized_pnl for p in winners])) if winners else 0.0
        avg_loss = abs(float(np.mean([p.realized_pnl for p in losers]))) if losers else 0.0

        risk_reward = avg_win / avg_loss if avg_loss > 0 else 0.0

        return {
            "risk_reward_ratio": risk_reward,
            "payoff_ratio": risk_reward,
        }


@dataclass
@sf_component(name="max_consecutive", override=True)
class MaxConsecutiveMetric(StrategyMetric):
    """Tracks maximum consecutive wins and losses."""

    _last_closed_count: int = 0
    _current_win_streak: int = 0
    _current_loss_streak: int = 0
    _max_win_streak: int = 0
    _max_loss_streak: int = 0
    _last_result_win: bool | None = None

    def compute(self, state: StrategyState, prices: dict[str, float], **kwargs) -> dict[str, float]:
        closed_positions = [p for p in state.portfolio.positions.values() if p.is_closed]
        current_count = len(closed_positions)

        if current_count > self._last_closed_count:
            new_positions = sorted(
                closed_positions,
                key=lambda p: p.last_time or datetime.min,
            )
            new_positions = new_positions[self._last_closed_count :]

            for pos in new_positions:
                is_win = pos.realized_pnl > 0

                if self._last_result_win is None:
                    self._last_result_win = is_win
                    if is_win:
                        self._current_win_streak = 1
                    else:
                        self._current_loss_streak = 1
                elif is_win == self._last_result_win:
                    if is_win:
                        self._current_win_streak += 1
                    else:
                        self._current_loss_streak += 1
                else:
                    if is_win:
                        self._max_loss_streak = max(self._max_loss_streak, self._current_loss_streak)
                        self._current_loss_streak = 0
                        self._current_win_streak = 1
                    else:
                        self._max_win_streak = max(self._max_win_streak, self._current_win_streak)
                        self._current_win_streak = 0
                        self._current_loss_streak = 1
                    self._last_result_win = is_win

            self._last_closed_count = current_count

        max_wins = max(self._max_win_streak, self._current_win_streak)
        max_losses = max(self._max_loss_streak, self._current_loss_streak)

        return {
            "max_consecutive_wins": max_wins,
            "max_consecutive_losses": max_losses,
            "current_win_streak": self._current_win_streak,
            "current_loss_streak": self._current_loss_streak,
        }
