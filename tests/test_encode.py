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


def test_binarize_threshold_splits_multiclass():
    from signalflow.transform.encode.woe import _binarize

    t = pl.Series([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    assert _binarize(t, positive_threshold=0.5).tolist() == [0, 1, 1, 0, 1, 1]


def test_binarize_positive_classes_isolates_class():
    from signalflow.transform.encode.woe import _binarize

    t = pl.Series([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    assert _binarize(t, positive_classes=(2.0,)).tolist() == [0, 0, 1, 0, 0, 1]


def test_binarize_single_class_raises():
    from signalflow.transform.encode.woe import _binarize

    with pytest.raises(sf.DegenerateTargetError):
        _binarize(pl.Series([1.0, 1.0, 1.0]))


def test_woe_positive_classes_fits_terciles(ds):
    feat, _ = _feat_and_target(ds)
    y = pl.Series("y", [float(i % 3) for i in range(feat.height)])
    woe = sf.WoE(binning=sf.Binning("quantile", 5), positive_classes=(2.0,))
    woe.fit(feat, y)
    assert woe.outputs


def test_iv_selector_positive_classes(ds):
    feat, _ = _feat_and_target(ds)
    y = pl.Series("y", [float(i % 3) for i in range(feat.height)])
    sel = sf.IVSelector(positive_classes=(2.0,))
    sel.fit(feat, y)
    assert isinstance(sel.keep_, list)


def test_woe_single_class_target_raises(ds):
    feat, _ = _feat_and_target(ds)
    y = pl.Series("y", [1.0] * feat.height)
    with pytest.raises(sf.DegenerateTargetError):
        sf.WoE().fit(feat, y)


def test_woe_binarization_config_round_trips():
    woe = sf.WoE(positive_threshold=0.5, positive_classes=(2.0,))
    cfg = woe.to_config()
    rebuilt = sf.WoE.from_config(cfg)
    assert rebuilt.positive_threshold == 0.5
    assert rebuilt.positive_classes == (2.0,)
    assert rebuilt.to_config() == cfg


def test_woe_none_sentinel_round_trips():
    woe = sf.WoE(refit=None, window=None)
    cfg = woe.to_config()
    rebuilt = sf.WoE.from_config(cfg)
    assert rebuilt.refit is None and rebuilt.window is None
    assert cfg["params"]["refit"] is None and cfg["params"]["window"] is None


def test_woe_none_matches_empty_string_non_rolling():
    data = sf.data("memory", pairs=["BTCUSDT"], start="2024-01-01", end="2024-01-10", interval="1h")

    def _fit(refit, window):
        return sf.ForecastModel(
            target=sf.FixedHorizon(bars=6),
            features=sf.FeaturePipe(sf.SMA(5)),
            encode=sf.WoE(refit=refit, window=window),
            min_train_rows=20,
        ).fit(data)

    assert _fit(None, None).oos_.equals(_fit("", "").oos_)
