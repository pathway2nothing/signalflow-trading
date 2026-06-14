"""ForecastModel - the trainable tier-1 model."""


import warnings
from dataclasses import dataclass, field
from datetime import timedelta

warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
import polars as pl
from loguru import logger
from sklearn.base import clone

from signalflow.data.dataset import Dataset
from signalflow.errors import UntrainedModelError
from signalflow.model.oos import build_fingerprint, make_folds, median_dt
from signalflow.sampler import Sampler, UniformSampler
from signalflow.target import LABEL_COL, Target
from signalflow.transform import FeaturePipe
from signalflow.transform.encode import IVSelector, WoE

_DEFAULT = "default"
_RESERVED = {"pair", "ts", "open", "high", "low", "close", "volume", "signal"}


def _make_estimator(backend, params: dict):
    if not isinstance(backend, str):
        return clone(backend)
    b = backend.lower()
    if b == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**{"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.05,
                                 "verbosity": -1, **params})
    if b in ("logreg", "logistic"):
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=1000, **params)
    if b in ("rf", "random_forest"):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(**params)
    raise ValueError(f"unknown backend {backend!r}")


def _encode_frame(enc: WoE | None, features: FeaturePipe, feat_frame: pl.DataFrame):
    """Apply (fitted) encode and return (frame, feature_columns) for the matrix."""
    if enc is None:
        return feat_frame, list(features.outputs)
    frame = enc.compute(feat_frame)
    frame = frame.drop([c for c in features.outputs if c in frame.columns])
    return frame, list(enc.outputs)


@dataclass
class ForecastModel:
    """Trainable continuous predictor; outputs one probability column."""

    backend: object = "lightgbm"
    target: Target | None = None
    features: FeaturePipe | None = None
    encode: object = _DEFAULT
    select: object = _DEFAULT
    sampler: Sampler | None = None
    backend_params: dict = field(default_factory=dict)
    output: str = "p_rise"
    n_folds: int = 5
    min_train_rows: int = 50

    def __post_init__(self) -> None:
        if self.features is None:
            self.features = FeaturePipe()
        if self.encode == _DEFAULT:
            self.encode = WoE()
        if self.select == _DEFAULT:
            self.select = IVSelector()
        self._fitted = False


    @property
    def is_fitted(self) -> bool:
        return getattr(self, "_fitted", False)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise UntrainedModelError(f"ForecastModel(output={self.output!r}) is not fitted")


    def fit(self, data: Dataset, sampler: Sampler | None = None) -> "ForecastModel":
        if self.target is None:
            raise ValueError("ForecastModel requires a target to fit")
        sampler = sampler or self.sampler or UniformSampler()

        feat = self.features.compute(data.frame)
        ss = sampler.sample(data)
        idx = ss.index
        if ss.weights is not None:
            idx = idx.with_columns(ss.weights.alias("_w"))
        labels = self.target.labels(data, at=ss.index)

        base = (
            idx.join(feat, on=["pair", "ts"], how="inner")
            .join(labels, on=["pair", "ts"], how="left")
            .drop_nulls(subset=[LABEL_COL, *self.features.outputs])
            .sort("ts")
        )
        if base.height < self.min_train_rows:
            raise ValueError(f"not enough labeled samples to fit ({base.height})")


        ts_unique = base.get_column("ts").unique().sort().to_list()
        folds = make_folds(ts_unique, self.n_folds)
        embargo = timedelta(seconds=self.target.horizon * median_dt(ts_unique))
        oos_parts: list[pl.DataFrame] = []
        for fold in folds:
            train = base.filter(pl.col("ts") < (fold.test_start_ts - embargo))
            test = base.filter((pl.col("ts") >= fold.test_start_ts) & (pl.col("ts") <= fold.test_end_ts))
            if train.height < self.min_train_rows or test.height == 0:
                continue
            preds = self._fit_fold_predict(train, test)
            if preds is not None:
                oos_parts.append(preds)

        self.oos_ = (
            pl.concat(oos_parts)
            if oos_parts
            else pl.DataFrame(schema={"pair": pl.Utf8, "ts": base.schema["ts"], self.output: pl.Float64})
        )


        self.encode_, self.select_, self.model_ = self._fit_stack(base)

        self.encode = self.encode_
        self.select = self.select_

        self._build_fingerprint(data, ts_unique)
        self._fitted = True
        logger.debug(f"ForecastModel fitted: oos rows={self.oos_.height}, folds={len(folds)}")
        return self

    def _fit_stack(self, train: pl.DataFrame):
        y = train.get_column(LABEL_COL)
        enc = self._fresh_encode()
        if enc is not None:
            enc.fit(train, y)
        frame, cols = _encode_frame(enc, self.features, train)
        sel = self._fresh_select()
        if sel is not None:
            sel.fit(frame, y)
            cols = [c for c in cols if c in sel.keep_]
        cols = cols or list((enc.outputs if enc else self.features.outputs))
        X = frame.select(cols).fill_null(0.0).to_numpy()
        est = _make_estimator(self.backend, self.backend_params)
        w = train.get_column("_w").to_numpy() if "_w" in train.columns else None
        est._sf_cols = cols
        if len(np.unique(y.to_numpy())) < 2:
            est._sf_degenerate = float(y.mean())
        else:
            est.fit(X, y.to_numpy(), sample_weight=w)
        return enc, sel, est

    def _fit_fold_predict(self, train: pl.DataFrame, test: pl.DataFrame) -> pl.DataFrame | None:
        enc, sel, est = self._fit_stack(train)
        p = self._predict_stack(enc, sel, est, test)
        return test.select(["pair", "ts"]).with_columns(pl.Series(self.output, p))

    def _predict_stack(self, enc, sel, est, frame: pl.DataFrame) -> np.ndarray:
        cols = getattr(est, "_sf_cols", None)
        efrt, default_cols = _encode_frame(enc, self.features, frame)
        cols = cols or default_cols
        X = efrt.select(cols).fill_null(0.0).to_numpy()
        if hasattr(est, "_sf_degenerate"):
            return np.full(X.shape[0], est._sf_degenerate, dtype=float)
        return est.predict_proba(X)[:, 1]

    def _fresh_encode(self):
        if self.encode is None:
            return None
        return WoE(binning=self.encode.binning, refit=self.encode.refit, window=self.encode.window,
                   smoothing=self.encode.smoothing, columns=list(self.features.outputs))

    def _fresh_select(self):
        if self.select is None:
            return None
        return IVSelector(min_iv=self.select.min_iv, max_bins=self.select.max_bins,
                          smoothing=self.select.smoothing)


    def predict(self, data: Dataset) -> pl.DataFrame:
        """Production prediction (in-sample on history - never feed to training)."""
        self._check_fitted()
        feat = self.features.compute(data.frame)
        p = self._predict_stack(self.encode_, self.select_, self.model_, feat)
        return feat.select(["pair", "ts"]).with_columns(pl.Series(self.output, p))

    def predict_oos(self, data: Dataset) -> pl.DataFrame:
        """Leak-free out-of-fold predictions over the training span."""
        self._check_fitted()
        want = data.index()
        out = want.join(self.oos_, on=["pair", "ts"], how="left")
        missing = out.get_column(self.output).null_count()
        if missing:
            logger.warning(f"predict_oos: {missing} rows outside cached OOS span (null)")
        return out


    def _build_fingerprint(self, data: Dataset, ts_unique: list) -> None:
        enc_cfg = self.encode_.to_config()["params"] if self.encode_ else None
        sel_cfg = {"min_iv": self.select_.min_iv} if self.select_ else None
        self.fingerprint = build_fingerprint(
            backend=self.backend if isinstance(self.backend, str) else type(self.backend).__name__,
            backend_params=self.backend_params,
            target_cfg=self.target.to_config(),
            features_cfg=self.features.to_config(),
            encode_cfg=enc_cfg,
            select_cfg=sel_cfg,
            dataset_params=data.source_params,
            cv={
                "scheme": "rolling",
                "refit": getattr(self.encode_, "refit", None),
                "window": getattr(self.encode_, "window", None),
                "n_folds": self.n_folds,
                "purge": self.target.horizon,
                "embargo": self.target.horizon,
                "span": [str(ts_unique[0]), str(ts_unique[-1])] if ts_unique else None,
            },
            output=self.output,
        )
        self.feature_signature = {
            "features": self.features.to_config(),
            "encode": enc_cfg,
            "select_keep": list(getattr(self.select_, "keep_", [])) if self.select_ else None,
            "output": self.output,
            "warmup": self.features.warmup,
        }


    def save(self, uri: str) -> str:
        from signalflow.model.store import save_model

        saved = save_model(self, uri)
        self._uri = saved
        return saved

    @classmethod
    def load(cls, uri: str) -> "ForecastModel":
        from signalflow.model.store import load_model

        return load_model(uri)

    def __repr__(self) -> str:
        state = "fitted" if self.is_fitted else "unfitted"
        return f"ForecastModel(backend={self.backend!r}, output={self.output!r}, {state})"
