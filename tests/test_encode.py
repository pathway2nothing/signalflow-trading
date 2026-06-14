"""WoE/IV target-encoding policy."""


import numpy as np
import polars as pl
import pytest

import signalflow as sf


def _feat_and_target(ds):
    pipe = sf.FeaturePipe(sf.SMA(20), sf.SMA(10), sf.SMA(50))
    feat = pipe.compute(ds.frame).drop_nulls()
    raw = feat.get_column("sma_50").to_numpy()
    z = (raw - raw.mean()) / (raw.std() + 1e-12)
    rng = np.random.default_rng(0)
    prob = 1 / (1 + np.exp(2.0 * z))
    y = (rng.random(len(z)) < prob).astype(int)
    return feat, pl.Series("y", y)


def test_woe_encodes_all_features_not_label(ds):
    feat, y = _feat_and_target(ds)
    woe = sf.WoE(binning=sf.Binning("monotonic", 8))
    woe.fit(feat, y)
    assert woe.outputs == ["sma_20__woe", "sma_10__woe", "sma_50__woe"]
    enc = woe.compute(feat)
    assert "sma_50__woe" in enc.columns
    assert woe.iv_["sma_50"] == max(woe.iv_.values())


def test_iv_selector_keeps_informative(ds):
    feat, y = _feat_and_target(ds)
    sel = sf.IVSelector(min_iv=0.1)
    sel.fit(feat, y)
    assert "sma_50" in sel.keep_


def test_woe_is_leak_safe_stateful():
    woe = sf.WoE()
    with pytest.raises(sf.UnfittedTransformError):
        woe.compute(pl.DataFrame({"pair": ["X"], "ts": [0], "sma_20": [1.0]}))
