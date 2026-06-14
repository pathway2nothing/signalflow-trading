"""
2-state HMM vol-regime labeler (calm / turbulent).

Uses forward-backward smoothing on rolling log-vol; emits per-bar
probability triple ``(calm, turbulent)`` plus a discrete hard label
(``"calm" / "turbulent"``) at argmax.

NOTE: forward-backward smoothing reads both past *and* future
observations within its training window, so the hard label is
**forward-looking** and is only valid as a research / labeling target
- not as a live-trading signal.
"""


from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import polars as pl

from signalflow.enums import SignalCategory
from signalflow.target.base import register_target
from signalflow.target.labeler import Labeler


def _hmm_two_state_fb(
    obs: np.ndarray,
    mu: tuple[float, float],
    sigma: tuple[float, float],
    trans: np.ndarray,
    pi: np.ndarray,
) -> np.ndarray:
    """Forward-backward smoothing for a 2-state Gaussian HMM."""
    n = len(obs)
    out = np.full((n, 2), np.nan)
    valid = np.isfinite(obs)
    if valid.sum() < 50:
        return out
    idx = np.where(valid)[0]
    o = obs[valid]
    m = len(o)

    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
    lik = np.zeros((m, 2))
    for k in (0, 1):
        z = (o - mu[k]) / sigma[k]
        lik[:, k] = (inv_sqrt_2pi / sigma[k]) * np.exp(-0.5 * z * z)
    lik = np.clip(lik, 1e-30, None)

    alpha = np.zeros((m, 2))
    c = np.zeros(m)
    alpha[0] = pi * lik[0]
    c[0] = alpha[0].sum()
    alpha[0] /= c[0]
    for t in range(1, m):
        alpha[t] = (alpha[t - 1] @ trans) * lik[t]
        c[t] = alpha[t].sum()
        alpha[t] /= c[t]

    beta = np.zeros((m, 2))
    beta[-1] = 1.0
    for t in range(m - 2, -1, -1):
        beta[t] = trans @ (lik[t + 1] * beta[t + 1])
        beta[t] /= max(beta[t].sum(), 1e-30)

    gamma = alpha * beta
    gamma /= np.clip(gamma.sum(axis=1, keepdims=True), 1e-30, None)
    out[idx] = gamma
    return out


@dataclass
@register_target("hmm_vol_regime_2state")
class HMMVolRegime2StateLabeler(Labeler):
    """
    2-state Gaussian HMM (calm / turbulent) on rolling log-vol.

    Algorithm:
        1. Compute log-returns and rolling realised log-vol over
           ``smoother`` bars.
        2. Estimate Gaussian state parameters from each
           ``train_window`` slice: μ from the 25/75 percentiles of log-vol,
           σ from a contracted slice std, transition matrix's stickiness
           from the lag-1 autocorrelation of log-vol.
        3. Run forward-backward smoothing → per-bar posterior
           ``[P(calm), P(turbulent)]``.
        4. Argmax → hard label ``"calm" / "turbulent"``.

    Output (hard mode):
        ``out_col`` ∈ ``{"calm", "turbulent"}``.

    Output (soft mode, :meth:`compute_soft`):
        ``p_calm``, ``p_turbulent`` summing to 1.

    Research provenance:
        iter-35 (sf-profit) - best soft label of all iterations.
        Soft MI = 0.391 with ``GMMVolRegime5State`` ``volreg5_q90`` and
        0.385 with ``volreg5_q10`` on the validated pool subset.
        Marginal entropy H = 0.999 bits (balanced, not degenerate
        unlike ``soft_H1_hurst``).
    """

    signal_category: SignalCategory = SignalCategory.VOLATILITY

    soft_classes: ClassVar[tuple[str, ...]] = ("calm", "turbulent")

    price_col: str = "close"
    smoother: int = 60
    train_window: int = 4320

    meta_columns: tuple[str, ...] = ("log_vol",)

    def __post_init__(self) -> None:
        if self.smoother < 5:
            raise ValueError("smoother must be >= 5")
        if self.train_window < 4 * self.smoother:
            raise ValueError("train_window must be >= 4 * smoother")

        cols = [self.out_col]
        if self.include_meta:
            cols += list(self.meta_columns)
        self.output_columns = cols

    def _posterior(self, group_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (gamma, log_rv) - posterior and the underlying log-vol."""
        close = group_df.get_column(self.price_col).to_numpy().astype(np.float64)
        n = len(close)
        c = np.maximum(close, 1e-12)
        log_c = np.log(c)
        r = np.diff(log_c, prepend=log_c[0])
        sq = r * r
        csum = np.concatenate([[0.0], np.cumsum(sq)])
        rv = np.full(n, np.nan)
        for i in range(self.smoother - 1, n):
            rv[i] = np.sqrt((csum[i + 1] - csum[i + 1 - self.smoother]) / self.smoother)
        log_rv = np.log(np.maximum(rv, 1e-12))

        gamma = np.full((n, 2), np.nan)
        if n < self.train_window:
            return gamma, log_rv
        step = max(self.train_window // 2, 1)
        for end in range(self.train_window, n, step):
            seg = log_rv[end - self.train_window : end]
            v = seg[~np.isnan(seg)]
            if len(v) < self.train_window // 4:
                continue
            q25, q75 = np.quantile(v, [0.25, 0.75])
            mu = (float(q25), float(q75))
            sd = max(float(v.std(ddof=1)) * 0.7, 1e-3)
            sigma = (sd, sd)
            acf = float(np.corrcoef(v[:-1], v[1:])[0, 1]) if len(v) > 1 else 0.0
            stay = float(np.clip(0.5 + 0.5 * acf, 0.6, 0.99))
            trans = np.array([[stay, 1 - stay], [1 - stay, stay]])
            pi = np.array([0.5, 0.5])
            start_apply = max(end - self.train_window, 0)
            stop_apply = min(end + step, n)
            g = _hmm_two_state_fb(log_rv[start_apply:stop_apply], mu, sigma, trans, pi)
            gamma[start_apply:stop_apply] = g
        return gamma, log_rv

    def compute_group(
        self,
        group_df: pl.DataFrame,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")
        if group_df.height == 0:
            return group_df

        gamma, log_rv = self._posterior(group_df)
        valid = ~np.isnan(gamma[:, 0])
        labels = np.full(group_df.height, None, dtype=object)
        argmax = np.where(gamma[:, 1] > gamma[:, 0], "turbulent", "calm")
        labels[valid] = argmax[valid]

        df = group_df.with_columns(pl.Series(self.out_col, labels.tolist(), dtype=pl.Utf8))
        if self.include_meta:
            df = df.with_columns(pl.Series("log_vol", log_rv, dtype=pl.Float64))

        if self.mask_to_signals and data_context is not None and "signal_keys" in data_context:
            df = self._apply_signal_mask(df, data_context, group_df)
        return df

    def compute_group_soft(
        self,
        group_df: pl.DataFrame,
        data_context: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Per-bar posterior ``[p_calm, p_turbulent]`` (sums to 1 per row)."""
        if self.price_col not in group_df.columns:
            raise ValueError(f"Missing required column '{self.price_col}'")
        if group_df.height == 0:
            return group_df
        gamma, _ = self._posterior(group_df)
        return group_df.with_columns(
            pl.Series(f"{self.soft_col_prefix}calm", gamma[:, 0], dtype=pl.Float64),
            pl.Series(f"{self.soft_col_prefix}turbulent", gamma[:, 1], dtype=pl.Float64),
        )
