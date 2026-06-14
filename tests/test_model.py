"""ForecastModel: walk-forward fit, predict vs predict_oos, OOS artifact."""


import pytest

import signalflow as sf


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
