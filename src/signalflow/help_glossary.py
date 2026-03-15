"""Built-in glossary of trading, ML, and quantitative finance terminology.

Used by :mod:`signalflow.help` to provide contextual explanations for
metrics, concepts, and algorithms referenced throughout SignalFlow.
"""

from __future__ import annotations

GLOSSARY: dict[str, dict[str, str]] = {
    # ── Performance Metrics ────────────────────────────────────────────
    "sharpe_ratio": {
        "definition": "Risk-adjusted return measuring excess return per unit of volatility.",
        "formula": "(R_p - R_f) / sigma_p, annualized",
        "interpretation": "Higher is better. Compares return to risk taken.",
        "good_range": "> 1.0 acceptable, > 2.0 good, > 3.0 excellent",
        "category": "Performance Metrics",
    },
    "sortino_ratio": {
        "definition": "Like Sharpe but only penalizes downside volatility, not upside.",
        "formula": "(R_p - R_f) / sigma_downside",
        "interpretation": "Higher is better. More appropriate when returns are asymmetric.",
        "good_range": "> 1.5 acceptable, > 3.0 good",
        "category": "Performance Metrics",
    },
    "calmar_ratio": {
        "definition": "Annualized return divided by maximum drawdown.",
        "formula": "Annualized_Return / |Max_Drawdown|",
        "interpretation": "Higher is better. Measures return per unit of peak-to-trough loss.",
        "good_range": "> 1.0 acceptable, > 3.0 good",
        "category": "Performance Metrics",
    },
    "total_return": {
        "definition": "Total percentage gain/loss over the entire backtest period.",
        "formula": "(Final_Capital - Initial_Capital) / Initial_Capital",
        "interpretation": "Positive = profit, negative = loss. Compare vs benchmark.",
        "category": "Performance Metrics",
    },
    "annualized_return": {
        "definition": "Total return normalized to a yearly rate for comparison across timeframes.",
        "formula": "(1 + Total_Return)^(365/Days) - 1",
        "interpretation": "Allows comparison of strategies with different durations.",
        "category": "Performance Metrics",
    },
    "profit_factor": {
        "definition": "Ratio of gross profits to gross losses across all trades.",
        "formula": "Sum(winning_trades) / |Sum(losing_trades)|",
        "interpretation": "Must be > 1.0 for profitability. > 2.0 is strong.",
        "good_range": "> 1.5 acceptable, > 2.0 good",
        "category": "Performance Metrics",
    },
    "win_rate": {
        "definition": "Percentage of trades that are profitable.",
        "formula": "Winning_Trades / Total_Trades",
        "interpretation": "High win rate alone doesn't guarantee profitability (depends on risk:reward).",
        "good_range": "Context-dependent. Trend: 30-45%, Mean-reversion: 55-70%",
        "category": "Performance Metrics",
    },
    # ── Risk Metrics ───────────────────────────────────────────────────
    "max_drawdown": {
        "definition": "Largest peak-to-trough decline in equity curve.",
        "formula": "max((Peak - Trough) / Peak) over all time",
        "interpretation": "Measures worst-case loss from any high point. Lower magnitude is better.",
        "good_range": "< 10% conservative, < 20% moderate, < 30% aggressive",
        "category": "Risk Metrics",
    },
    "risk_of_ruin": {
        "definition": "Probability that a strategy will lose a specified percentage of capital.",
        "interpretation": "Estimated via Monte Carlo simulation by shuffling trade order.",
        "good_range": "< 5% is acceptable, < 1% is safe",
        "category": "Risk Metrics",
    },
    "value_at_risk": {
        "definition": "Maximum expected loss at a given confidence level over a time period.",
        "formula": "VaR_a = quantile(returns, a)",
        "interpretation": "E.g. 95% VaR of -2% means: 95% of the time, daily loss won't exceed 2%.",
        "category": "Risk Metrics",
    },
    "stop_loss": {
        "definition": "Order to close a position when price moves against you by a set amount.",
        "interpretation": "Limits downside risk per trade. Can be fixed %, ATR-based, or trailing.",
        "category": "Risk Management",
    },
    "take_profit": {
        "definition": "Order to close a position when price reaches a target profit level.",
        "interpretation": "Locks in gains. Often set as R-multiple of stop loss distance.",
        "category": "Risk Management",
    },
    "trailing_stop": {
        "definition": "Stop loss that moves with the price to lock in profits as position moves favorably.",
        "interpretation": "Combines trend following with risk management. Distance can be ATR-based.",
        "category": "Risk Management",
    },
    # ── Statistical Validation ─────────────────────────────────────────
    "psr": {
        "definition": "Probabilistic Sharpe Ratio -- probability that true Sharpe exceeds a benchmark.",
        "formula": "PSR = Phi((SR - SR*) * sqrt(n-1) / sqrt(1 - skew*SR + (kurt-1)/4*SR^2))",
        "interpretation": "Accounts for skewness and kurtosis. > 0.95 means statistically significant.",
        "good_range": "> 0.95 for 95% confidence",
        "category": "Statistical Validation",
    },
    "dsr": {
        "definition": "Deflated Sharpe Ratio -- adjusts PSR for multiple testing (data snooping).",
        "interpretation": "Use when comparing many strategy variants. Penalizes overfitting.",
        "category": "Statistical Validation",
    },
    "min_track_record_length": {
        "definition": "Minimum number of trades needed for the Sharpe ratio to be statistically significant.",
        "interpretation": "If your track record is shorter, the Sharpe ratio is unreliable.",
        "category": "Statistical Validation",
    },
    "bootstrap": {
        "definition": "Resampling method to estimate confidence intervals for performance metrics.",
        "interpretation": "Draws random samples (with replacement) from trades to build distribution of outcomes.",
        "category": "Statistical Validation",
    },
    "monte_carlo": {
        "definition": "Simulation that shuffles trade execution order to estimate outcome distribution.",
        "interpretation": "Tests whether results depend on specific trade sequence. Estimates risk of ruin.",
        "category": "Statistical Validation",
    },
    "confidence_interval": {
        "definition": "Range of values within which the true parameter value lies with specified probability.",
        "formula": "[lower, upper] at confidence level a",
        "interpretation": "95% CI means: if we repeated the experiment, 95% of intervals would contain the true value.",
        "category": "Statistical Validation",
    },
    # ── Strategy Types ─────────────────────────────────────────────────
    "momentum": {
        "definition": "Strategy that bets on continuation of existing price trends.",
        "interpretation": "Buy winners, sell losers. Works well in trending markets, poor in ranging.",
        "category": "Strategy Types",
    },
    "mean_reversion": {
        "definition": "Strategy that bets on prices returning to a historical average or equilibrium.",
        "interpretation": "Buy oversold, sell overbought. Works in ranging markets, poor in trends.",
        "category": "Strategy Types",
    },
    "grid_strategy": {
        "definition": "Strategy that places buy/sell orders at fixed intervals around current price.",
        "interpretation": "Profits from price oscillation. Grid spacing can be fixed or ATR-based.",
        "category": "Strategy Types",
    },
    "trend_following": {
        "definition": "Strategy that identifies and follows the dominant market trend direction.",
        "interpretation": "Uses indicators like EMA crossovers, ADX, or breakouts to determine trend.",
        "category": "Strategy Types",
    },
    "scalping": {
        "definition": "High-frequency strategy targeting small, quick profits on short timeframes.",
        "interpretation": "Requires tight spreads and low latency. High win rate, small risk:reward.",
        "category": "Strategy Types",
    },
    # ── Technical Indicators ───────────────────────────────────────────
    "sma": {
        "definition": "Simple Moving Average -- arithmetic mean of prices over N periods.",
        "formula": "SMA(N) = Sum(close, N) / N",
        "category": "Technical Indicators",
    },
    "ema": {
        "definition": "Exponential Moving Average -- weighted average giving more weight to recent prices.",
        "formula": "EMA(t) = a * price(t) + (1-a) * EMA(t-1), where a = 2/(N+1)",
        "category": "Technical Indicators",
    },
    "rsi": {
        "definition": "Relative Strength Index -- momentum oscillator measuring speed and change of price movements.",
        "formula": "RSI = 100 - 100/(1 + RS), where RS = avg_gain / avg_loss",
        "interpretation": "> 70 overbought, < 30 oversold (default thresholds).",
        "category": "Technical Indicators",
    },
    "macd": {
        "definition": "Moving Average Convergence Divergence -- trend-following momentum indicator.",
        "formula": "MACD = EMA(12) - EMA(26), Signal = EMA(9) of MACD",
        "interpretation": "Bullish when MACD crosses above signal, bearish when below.",
        "category": "Technical Indicators",
    },
    "bollinger_bands": {
        "definition": "Volatility bands placed above/below a moving average at N standard deviations.",
        "formula": "Upper = SMA(20) + 2sigma, Lower = SMA(20) - 2sigma",
        "interpretation": "Price touching bands indicates potential reversal or breakout.",
        "category": "Technical Indicators",
    },
    "atr": {
        "definition": "Average True Range -- volatility indicator measuring average range of price bars.",
        "formula": "ATR(N) = EMA(max(high-low, |high-prev_close|, |low-prev_close|), N)",
        "interpretation": "Used for stop-loss placement, position sizing, and grid spacing.",
        "category": "Technical Indicators",
    },
    "adx": {
        "definition": "Average Directional Index -- measures trend strength regardless of direction.",
        "interpretation": "> 25 indicates strong trend, < 20 indicates no trend (ranging).",
        "category": "Technical Indicators",
    },
    "keltner_channel": {
        "definition": "Volatility-based envelope set above/below an EMA using ATR for band width.",
        "formula": "Upper = EMA(20) + 2*ATR(10), Lower = EMA(20) - 2*ATR(10)",
        "category": "Technical Indicators",
    },
    "stochastic": {
        "definition": "Stochastic Oscillator -- compares closing price to price range over N periods.",
        "formula": "%K = (Close - Low_N) / (High_N - Low_N) * 100",
        "interpretation": "> 80 overbought, < 20 oversold.",
        "category": "Technical Indicators",
    },
    # ── ML / Labeling ──────────────────────────────────────────────────
    "meta_labeling": {
        "definition": "Two-stage ML approach: primary model generates signals, secondary model filters them.",
        "interpretation": "Reduces false positives. Secondary model predicts signal quality, not direction.",
        "category": "ML Concepts",
    },
    "triple_barrier": {
        "definition": "Labeling method using three barriers: take-profit (top), stop-loss (bottom), time (vertical).",
        "interpretation": "Labels each trade by which barrier is hit first: +1 (TP), -1 (SL), 0 (time).",
        "category": "ML Concepts",
    },
    "fixed_horizon": {
        "definition": "Labeling method that measures return over a fixed number of bars after entry signal.",
        "interpretation": "Simpler than triple barrier. Labels based on forward return sign.",
        "category": "ML Concepts",
    },
    "walk_forward": {
        "definition": "Out-of-sample validation method that trains on expanding/rolling windows and tests forward.",
        "interpretation": "More realistic than single train/test split. Detects regime changes.",
        "category": "ML Concepts",
    },
    "temporal_cv": {
        "definition": "Cross-validation respecting time order -- no future data leakage.",
        "interpretation": "Each fold uses only past data for training, future data for testing.",
        "category": "ML Concepts",
    },
    "purging": {
        "definition": "Removing training samples near the test boundary to prevent information leakage.",
        "interpretation": "Critical for overlapping labels (e.g. triple barrier with long holding periods).",
        "category": "ML Concepts",
    },
    "embargo": {
        "definition": "Gap between training and test sets to prevent leakage from serial correlation.",
        "interpretation": "Used with purging. Typical embargo: 1-5% of test fold size.",
        "category": "ML Concepts",
    },
    # ── Flow / Architecture ────────────────────────────────────────────
    "flow_builder": {
        "definition": "Fluent API for constructing strategy pipelines: data -> features -> detection -> execution.",
        "interpretation": "Chain methods: sf.flow().data(...).detector(...).entry(...).exit(...).run()",
        "category": "Architecture",
    },
    "signal_feature": {
        "definition": "Meta-feature computed from signal history (e.g. rolling accuracy, regime detector).",
        "interpretation": "Used to filter or weight signals based on historical performance patterns.",
        "category": "Architecture",
    },
}
