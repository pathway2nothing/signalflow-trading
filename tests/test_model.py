"""ForecastModel: walk-forward fit, predict vs predict_oos, OOS artifact."""

from dataclasses import dataclass

import polars as pl
import pytest

import signalflow as sf


@dataclass
class _AllNanFeature(sf.Feature):
    """Feature whose single output column is entirely NaN."""

    @property
    def outputs(self) -> list[str]:
        return ["all_nan"]

    def exprs(self) -> list[pl.Expr]:
        return [(pl.col("close") * float("nan")).alias("all_nan")]


@dataclass
class _PartialNanFeature(sf.Feature):
    """Feature that is NaN on the upper ~30% of close values."""

    @property
    def outputs(self) -> list[str]:
        return ["partial_nan"]

    def exprs(self) -> list[pl.Expr]:
        return [
            pl.when(pl.col("close") > pl.col("close").quantile(0.7))
            .then(float("nan"))
            .otherwise(pl.col("close"))
            .alias("partial_nan")
        ]


def test_untrained_guard(ds):
    m = sf.ForecastModel(target=sf.FixedHorizon(12), features=sf.FeaturePipe(sf.SMA(20)))
    with pytest.raises(sf.UntrainedModelError):
        m.predict(ds)


def test_fit_predict(ds, fitted_forecast):
    pred = fitted_forecast.predict(ds)
    assert "p_rise" in pred.columns
    assert pred.height == ds.height
    p = pred.get_column("p_rise").drop_nulls()
    assert (p.min() >= 0.0) and (p.max() <= 1.0)


def test_predict_oos_is_subset(ds, fitted_forecast):
    oos = fitted_forecast.predict_oos(ds)
    n_oos = oos.get_column("p_rise").drop_nulls().len()

    assert 0 < n_oos < ds.height


def test_predict_oos_strict_raises_on_gap(ds, fitted_forecast):
    with pytest.raises(sf.FingerprintMismatch):
        fitted_forecast.predict_oos(ds, strict=True)


def test_fingerprint_present(fitted_forecast):
    fp = fitted_forecast.fingerprint
    assert fp["cv"]["scheme"] == "rolling"
    assert "model_code" in fp and fp["id"].startswith("sha256:")


def test_default_encode_is_woe(ds):
    m = sf.ForecastModel(target=sf.FixedHorizon(12), features=sf.FeaturePipe(sf.SMA(20)))
    assert isinstance(m.encode, sf.WoE)
    assert isinstance(m.select, sf.IVSelector)


def test_encode_opt_out():
    m = sf.ForecastModel(target=sf.FixedHorizon(12), features=sf.FeaturePipe(sf.SMA(20)), encode=None)
    assert m.encode is None


def test_operating_point_returns_quantile(ds, fitted_forecast):
    expected = float(fitted_forecast.predict(ds).get_column("p_rise").drop_nulls().quantile(0.9))
    assert fitted_forecast.operating_point(ds, 0.9) == expected


def test_operating_point_oos_passthrough(ds, fitted_forecast):
    expected = float(fitted_forecast.predict_oos(ds).get_column("p_rise").drop_nulls().quantile(0.5))
    assert fitted_forecast.operating_point(ds, 0.5, oos=True) == expected


def test_operating_point_ambiguous_lists_columns(fitted_forecast):
    import polars as pl

    frame = pl.DataFrame({"pair": ["X"], "ts": [0], "p_rise": [0.4], "p_fall": [0.6]})
    with pytest.raises(ValueError, match="score columns"):
        fitted_forecast._score_column(frame, None)


def test_string_horizon_embargo_resolves_not_fallback():
    small = sf.data("memory", pairs=["BTCUSDT"], start="2023-01-01", end="2023-02-01", interval="1h")
    from signalflow.target import FixedHorizon, TripleBarrierLabeler

    fh = sf.FixedHorizon(bars="1d")
    assert fh.horizon == 1
    assert fh.horizon_bars(small) == 24
    assert FixedHorizon(bars=24).horizon_bars(small) == 24
    assert TripleBarrierLabeler(horizon="1d", vol_window=12).horizon_bars(small) == 24


def test_make_folds_clamps_to_available_timestamps():
    from datetime import datetime

    from signalflow.model.oos import make_folds

    ts = [datetime(2023, 1, 1, h) for h in range(3)]
    folds = make_folds(ts, 10)
    assert 1 <= len(folds) < 10


def test_fingerprint_records_effective_folds():
    m = sf.ForecastModel(
        target=sf.FixedHorizon(12), features=sf.FeaturePipe(sf.SMA(20)), encode=None, select=None, n_folds=5
    ).fit(sf.data("memory", pairs=["BTCUSDT"], start="2023-01-01", end="2023-03-01", interval="1h"))
    cv = m.fingerprint["cv"]
    assert cv["n_folds"] == 5
    assert cv["n_folds_effective"] == 4


def test_all_nan_feature_raises(ds):
    m = sf.ForecastModel(
        target=sf.FixedHorizon(12), features=sf.FeaturePipe(_AllNanFeature()), encode=None, select=None
    )
    with pytest.raises(sf.PipeError):
        m.fit(ds)


def test_partial_nan_rows_are_dropped(ds):
    clean = sf.ForecastModel(
        target=sf.FixedHorizon(12), features=sf.FeaturePipe(sf.SMA(20)), encode=None, select=None
    ).fit(ds)
    dirty = sf.ForecastModel(
        target=sf.FixedHorizon(12),
        features=sf.FeaturePipe(sf.SMA(20), _PartialNanFeature()),
        encode=None,
        select=None,
    ).fit(ds)
    assert dirty.oos_.height > 0
    assert dirty.oos_.height < clean.oos_.height


def test_model_embargo_uses_resolved_horizon():
    small = sf.data("memory", pairs=["BTCUSDT"], start="2023-01-01", end="2023-02-01", interval="1h")
    str_model = sf.ForecastModel(
        target=sf.FixedHorizon(bars="1d"),
        features=sf.FeaturePipe(sf.SMA(5)),
        encode=None,
        select=None,
        min_train_rows=20,
    ).fit(small)
    int_model = sf.ForecastModel(
        target=sf.FixedHorizon(bars=24), features=sf.FeaturePipe(sf.SMA(5)), encode=None, select=None, min_train_rows=20
    ).fit(small)
    assert str_model.fingerprint["cv"]["embargo"] == 24
    assert int_model.fingerprint["cv"]["embargo"] == 24
