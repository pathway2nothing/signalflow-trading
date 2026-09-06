"""ForecastModel - the trainable tier-1 model."""

import json
import time
import warnings
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import polars as pl
from loguru import logger
from sklearn.base import clone

from signalflow._hash import code_fingerprint, stable_hash
from signalflow._logging import frame_summary, step
from signalflow._time import bar_seconds
from signalflow.data.dataset import Dataset
from signalflow.errors import (
    DegenerateTargetError,
    FingerprintMismatch,
    FlowConfigError,
    PipelineError,
    UntrainedModelError,
)
from signalflow.model.cv import CVScheme, Rolling, build_cv
from signalflow.model.oos import build_fingerprint
from signalflow.sampler import Sampler, UniformSampler
from signalflow.target import LABEL_COL, Target
from signalflow.transform import FeaturePipeline
from signalflow.transform.base import ensure_sorted
from signalflow.transform.encode import WoE

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def _make_estimator(backend, params: dict):
    if not isinstance(backend, str):
        return clone(backend)
    b = backend.lower()
    if b == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            **{"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.05, "verbosity": -1, **params}
        )
    if b in ("logreg", "logistic"):
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=1000, **params)
    if b in ("rf", "random_forest"):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(**params)
    raise FlowConfigError(f"unknown backend {backend!r}")


def _mask_unready(p: np.ndarray, frame: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """NaN where any raw feature input is null/NaN (warmup or undefined), so no forecast fires there.

    Training never sees such rows (they are dropped before the fit), so a
    prediction on them would be an extrapolation from the encoder's missing bin.
    """
    present = [c for c in feature_cols if c in frame.columns]
    if not present or p.size == 0:
        return p
    checks = []
    for c in present:
        expr = pl.col(c).is_null()
        if frame.schema[c].is_float():
            expr = expr | pl.col(c).is_nan()
        checks.append(expr)
    unready = frame.select(pl.any_horizontal(checks)).to_series().to_numpy()
    out = np.asarray(p, dtype=float).copy()
    out[unready] = np.nan
    return out


def _tail_state(tail: FeaturePipeline | None) -> dict | None:
    """Portable fitted state of the first WoE in a fitted tail (the refit journal entry)."""
    if tail is None:
        return None
    for t in tail.transforms:
        if isinstance(t, WoE):
            return t.state_dict()
    return None


@dataclass
class ForecastModel:
    """Trainable continuous predictor; outputs one probability column.

    ``features`` is one :class:`FeaturePipeline`. Its stateless prefix is computed
    once over the data; its stateful tail (``requires_fit=True`` transforms such as
    ``Scaler``, ``WoE`` and ``IVSelector``) is refitted on the training rows of every
    fold, so a target encoder can never see its own test rows. A pipeline without a
    stateful tail trains on the raw feature columns.

    ``cv`` is the walk-forward scheme (:class:`~signalflow.model.cv.Rolling` or
    :class:`~signalflow.model.cv.KFold`) that produces the out-of-fold predictions.
    The fitted tail of the final production stack lives on ``tail_``.
    """

    backend: object = "lightgbm"
    target: Target | None = None
    features: FeaturePipeline | None = None
    sampler: Sampler | None = None
    backend_params: dict = field(default_factory=dict)
    output: str = "p_rise"
    cv: CVScheme = field(default_factory=Rolling)
    min_train_rows: int = 50

    def __post_init__(self) -> None:
        if self.features is None:
            self.features = FeaturePipeline()
        if not isinstance(self.features, FeaturePipeline):
            self.features = FeaturePipeline(self.features)
        self.cv = build_cv(self.cv)
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return getattr(self, "_fitted", False)

    @property
    def prefix(self) -> FeaturePipeline:
        """The stateless part of ``features`` (computed once, cacheable)."""
        return self.features.split()[0]

    @property
    def tail(self) -> FeaturePipeline | None:
        """The stateful part of ``features`` (refitted inside every fold), or ``None``."""
        tail = self.features.split()[1]
        return tail if tail.transforms else None

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise UntrainedModelError(f"ForecastModel(output={self.output!r}) is not fitted")

    def fit(self, data: Dataset, sampler: Sampler | None = None, cache=None, feature_store=None) -> "ForecastModel":
        """Train the model and compute leak-free out-of-fold predictions.

        Fits embargoed walk-forward folds (embargo width = the target horizon), stores
        the stitched out-of-fold predictions on ``oos_``, then fits the final production
        stack on all data. The folds come from ``cv``; the pipeline's stateful tail is
        refitted inside every fold. Passing an ``ArtifactCache`` reuses unchanged folds
        and recomputes only new ones.

        Returns:
            The fitted model (``self``).
        """
        if self.target is None:
            raise FlowConfigError("ForecastModel requires a target to fit")
        sampler = sampler or self.sampler or UniformSampler()
        t_fit = time.perf_counter()
        prefix, tail = self.features.split()
        tail = tail if tail.transforms else None
        backend_name = self.backend if isinstance(self.backend, str) else type(self.backend).__name__
        logger.debug(
            f"ForecastModel.fit({self.output}): backend={backend_name} target={self.target.name} "
            f"features={len(prefix.outputs)} tail={[t.name for t in tail.transforms] if tail else 'none'} "
            f"cv={self.cv.to_config()} sampler={getattr(sampler, 'name', type(sampler).__name__)} "
            f"data: {frame_summary(data.frame)}"
        )

        with step("ForecastModel.fit: features", store="yes" if feature_store is not None else "no") as log:
            feat = feature_store.compute(prefix, data) if feature_store is not None else prefix.compute(data.frame)
            log["rows"] = f"{feat.height:,}"
        with step("ForecastModel.fit: sampler") as log:
            ss = sampler.sample(data)
            log["selected"] = f"{ss.index.height:,}/{data.height:,}"
            log["weighted"] = "yes" if ss.weights is not None else "no"
        idx = ss.index
        if ss.weights is not None:
            idx = idx.with_columns(ss.weights.alias("_w"))
        with step("ForecastModel.fit: labels", target=self.target.name) as log:
            labels = self.target.labels(data, at=ss.index)
            lab = labels.get_column(LABEL_COL).drop_nulls()
            log["labeled"] = f"{lab.len():,}/{labels.height:,}"
            if lab.len():
                log["mean"] = f"{float(lab.cast(pl.Float64).mean()):.3f}"
                log["unique"] = lab.n_unique()

        raw_cols = list(prefix.outputs)
        base = (
            idx.join(feat, on=["pair", "ts"], how="inner")
            .join(labels, on=["pair", "ts"], how="left")
            .drop_nulls(subset=[LABEL_COL, *raw_cols])
            .sort("ts")
        )
        float_cols = [c for c in raw_cols if base.schema[c] in (pl.Float32, pl.Float64)]
        if float_cols:
            nan_frac = {c: base.get_column(c).is_nan().mean() for c in float_cols}
            dead = [c for c, frac in nan_frac.items() if frac == 1.0]
            if dead:
                raise PipelineError(
                    f"feature columns are entirely NaN over the training set: {dead}; check feature warmup and inputs"
                )
            before = base.height
            base = base.filter(~pl.any_horizontal([pl.col(c).is_nan() for c in float_cols]))
            if before and base.height < before * 0.5:
                logger.warning(
                    f"ForecastModel.fit: NaN filtering dropped {before - base.height} of {before} rows; "
                    f"NaN fractions: {nan_frac}"
                )
        if base.height < self.min_train_rows:
            raise DegenerateTargetError(f"not enough labeled samples to fit ({base.height})")

        ts_unique = base.get_column("ts").unique().sort().to_list()
        horizon_bars = self.target.horizon_bars(data)
        embargo = timedelta(seconds=horizon_bars * bar_seconds(ts_unique))
        folds = self.cv.folds(ts_unique, embargo)
        logger.debug(
            f"ForecastModel.fit: training set rows={base.height:,} features={len(raw_cols)} "
            f"span={ts_unique[0]}..{ts_unique[-1]}; folds={len(folds)} horizon_bars={horizon_bars} embargo={embargo}"
        )
        oos_parts: list[pl.DataFrame] = []
        self.refits_: list[dict] = []
        target_cfg = self.target.to_config()
        stack_fp = self._stack_fingerprint(data) if cache is not None else None
        n_cached = 0
        for i, fold in enumerate(folds, 1):
            t_fold = time.perf_counter()
            train = base.filter(pl.col("ts") < (fold.test_start - embargo))
            if fold.train_start is not None:
                train = train.filter(pl.col("ts") >= fold.train_start)
            test = base.filter((pl.col("ts") >= fold.test_start) & (pl.col("ts") <= fold.test_end))
            if train.height < self.min_train_rows or test.height == 0:
                logger.debug(
                    f"ForecastModel.fit: fold {i}/{len(folds)} skipped "
                    f"(train rows={train.height}, test rows={test.height})"
                )
                continue
            cached = self._load_fold(cache, stack_fp, fold, embargo) if cache is not None else None
            if cached is not None:
                preds, state = cached
                n_cached += 1
            else:
                preds, fold_tail, kept = self._fit_fold_predict(train, test, fold=fold)
                if preds is None:
                    continue
                state = _tail_state(fold_tail)
                if cache is not None:
                    self._store_fold(cache, stack_fp, fold, embargo, preds, state)
            logger.debug(
                f"ForecastModel.fit: fold {i}/{len(folds)} "
                f"{'cached' if cached is not None else f'fitted kept={kept}/{len(raw_cols)}'}: "
                f"train rows={train.height:,} (..{fold.test_start - embargo}) "
                f"test rows={test.height:,} ({fold.test_start}..{fold.test_end}) "
                f"({time.perf_counter() - t_fold:.2f}s)"
            )
            oos_parts.append(preds)
            if state is not None:
                self.refits_.append(
                    {
                        "test_start": fold.test_start,
                        "train_start": fold.train_start,
                        "train_end": fold.test_start - embargo,
                        "target": target_cfg,
                        "state": state,
                    }
                )

        self.oos_ = (
            pl.concat(oos_parts).unique(subset=["pair", "ts"], keep="first").sort(["pair", "ts"])
            if oos_parts
            else pl.DataFrame(schema={"pair": pl.Utf8, "ts": base.schema["ts"], self.output: pl.Float64})
        )
        if self.oos_.height > 0 and self.oos_.get_column(self.output).n_unique() == 1:
            constant = self.oos_.get_column(self.output)[0]
            logger.warning(
                f"ForecastModel(output={self.output!r}): OOS predictions are constant ({constant}); "
                f"the model likely learned nothing"
            )

        with step("ForecastModel.fit: final production stack", rows=f"{base.height:,}"):
            self.tail_, self.model_ = self._fit_stack(base)

        self._build_fingerprint(data, ts_unique, n_folds_effective=len(folds))
        self._fitted = True
        kept = len(getattr(self.model_, "_sf_cols", []) or [])
        logger.info(
            f"ForecastModel.fit({self.output}): {len(oos_parts)}/{len(folds)} folds ({n_cached} cached), "
            f"oos rows={self.oos_.height:,}, features kept {kept}/{len(raw_cols)} "
            f"({'encoder: ' + ', '.join(t.name for t in tail.transforms) if tail else 'raw features, no encoder'}), "
            f"backend={backend_name} ({time.perf_counter() - t_fit:.2f}s)"
        )
        return self

    def _fit_stack(self, train: pl.DataFrame, fold=None):
        """Fit a fresh copy of the pipeline's stateful tail on ``train``, then the estimator on its outputs."""
        train = ensure_sorted(train)  # the tail's compute sorts by (pair, ts); keep X, y and weights aligned
        y = train.get_column(LABEL_COL)
        est = _make_estimator(self.backend, self.backend_params)
        prefix, tail = self.features.split()
        raw_cols = list(prefix.outputs)
        if len(np.unique(y.to_numpy())) < 2:
            if fold is None:
                raise DegenerateTargetError(
                    f"ForecastModel(output={self.output!r}) final production fit has a single-class "
                    f"target (value={float(y.mean())}); refusing to store a constant predictor."
                )
            logger.warning(
                f"ForecastModel(output={self.output!r}) inner fold "
                f"[{fold.train_start}..{fold.test_start}] has a single-class target; "
                f"storing a constant predictor for this fold only."
            )
            est._sf_degenerate = float(y.mean())
            est._sf_cols = raw_cols
            return None, est
        fitted_tail = None
        frame = train
        cols = raw_cols
        if tail.transforms:
            fitted_tail = tail.clone().fit(train, y)
            frame = fitted_tail.compute(train)
            cols = fitted_tail.outputs
            missing = [c for c in cols if c not in frame.columns]
            if missing:
                raise PipelineError(
                    f"pipeline tail declares outputs {missing} that its compute did not produce; "
                    f"columns present: {frame.columns}"
                )
        if not cols:
            raise PipelineError("the feature pipeline produced no model input columns")
        X = frame.select(cols).fill_null(0.0).to_numpy()
        w = train.get_column("_w").to_numpy() if "_w" in train.columns else None
        est._sf_cols = cols
        est.fit(X, y.to_numpy(), sample_weight=w)
        return fitted_tail, est

    def _fit_fold_predict(self, train: pl.DataFrame, test: pl.DataFrame, fold=None):
        """Fit one fold's stack and score its test rows; also report how many columns the stack kept."""
        fitted_tail, est = self._fit_stack(train, fold=fold)
        test = ensure_sorted(test)
        p = self._predict_stack(fitted_tail, est, test)
        kept = len(getattr(est, "_sf_cols", None) or [])
        return test.select(["pair", "ts"]).with_columns(pl.Series(self.output, p)), fitted_tail, kept

    def _stack_fingerprint(self, data: Dataset) -> str:
        """Identity of the fold-producing stack: configs + code + dataset."""
        return stable_hash(
            {
                "features": self.features.to_config(),
                "target": self.target.to_config(),
                "backend": self.backend if isinstance(self.backend, str) else type(self.backend).__name__,
                "backend_params": self.backend_params,
                "output": self.output,
                "cv": self.cv.to_config(),
                "dataset": data.source_params,
                "model_code": code_fingerprint(type(self)),
                "transform_code": [code_fingerprint(type(t)) for t in self.features.transforms],
                "target_code": code_fingerprint(type(self.target)),
            }
        )

    def _fold_keys(self, cache, stack_fp: str, fold, embargo: timedelta):
        parts = {
            "stack": stack_fp,
            "train_start": str(fold.train_start),
            "train_end": str(fold.test_start - embargo),
            "test_start": str(fold.test_start),
            "test_end": str(fold.test_end),
        }
        return (
            cache.key(parts, kind="oos_for_training"),
            cache.key({**parts, "artifact": "woe_state"}, kind="oos_for_training"),
        )

    def _load_fold(self, cache, stack_fp: str, fold, embargo: timedelta):
        okey, skey = self._fold_keys(cache, stack_fp, fold, embargo)
        preds = cache.get(okey)
        state_df = cache.get(skey)
        if preds is None or state_df is None:
            return None
        raw = state_df.get_column("state")[0]
        return preds, (None if raw == "null" else json.loads(raw))

    def _store_fold(self, cache, stack_fp: str, fold, embargo: timedelta, preds, state) -> None:
        okey, skey = self._fold_keys(cache, stack_fp, fold, embargo)
        cache.put(okey, preds)
        cache.put(skey, pl.DataFrame({"state": [json.dumps(state) if state is not None else "null"]}))

    def _predict_stack(self, fitted_tail, est, frame: pl.DataFrame) -> np.ndarray:
        """Score ``frame`` (prefix outputs present, (pair, ts)-sorted) through the fitted tail and the estimator."""
        frame = ensure_sorted(frame)
        raw_cols = list(self.features.split()[0].outputs)
        cols = getattr(est, "_sf_cols", None) or raw_cols
        scored = fitted_tail.compute(frame) if fitted_tail is not None else frame
        X = scored.select(cols).fill_null(0.0).to_numpy()
        if hasattr(est, "_sf_degenerate"):
            p = np.full(X.shape[0], est._sf_degenerate, dtype=float)
        else:
            p = est.predict_proba(X)[:, 1]
        return _mask_unready(p, frame, raw_cols)

    def predict(self, data: Dataset, feature_store=None) -> pl.DataFrame:
        """Production prediction (in-sample on history - never feed to training).

        Null wherever a raw feature input is null/NaN (the warmup window).
        """
        self._check_fitted()
        prefix = self.features.split()[0]
        with step(f"ForecastModel.predict({self.output})") as log:
            feat = feature_store.compute(prefix, data) if feature_store is not None else prefix.compute(data.frame)
            p = self._predict_stack(self.tail_, self.model_, feat)
            log["rows"] = f"{feat.height:,}"
        return feat.select(["pair", "ts"]).with_columns(pl.Series(self.output, p).fill_nan(None))

    def predict_oos(self, data: Dataset, strict: bool = False) -> pl.DataFrame:
        """Leak-free out-of-fold predictions over the training span.

        With ``strict``, rows outside the cached OOS span raise
        :class:`FingerprintMismatch` instead of warning and returning nulls.
        """
        self._check_fitted()
        want = data.index()
        out = want.join(self.oos_, on=["pair", "ts"], how="left")
        missing = out.get_column(self.output).null_count()
        logger.debug(f"ForecastModel.predict_oos({self.output}): rows={out.height:,} covered={out.height - missing:,}")
        if missing:
            if strict:
                raise FingerprintMismatch(
                    f"predict_oos: {missing} of {out.height} requested rows fall outside the cached OOS "
                    f"span for ForecastModel(output={self.output!r}); refit or widen the span"
                )
            logger.warning(f"predict_oos: {missing} rows outside cached OOS span (null)")
        return out

    def operating_point(self, data: Dataset, quantile: float, column: str | None = None, oos: bool = False) -> float:
        """Score quantile as a firing threshold; caller must pass train-window data for leak safety."""
        self._check_fitted()
        preds = self.predict_oos(data) if oos else self.predict(data)
        col = self._score_column(preds, column)
        return float(preds.get_column(col).drop_nulls().quantile(quantile))

    def _score_column(self, preds: pl.DataFrame, column: str | None) -> str:
        scores = [c for c in preds.columns if c not in ("pair", "ts", "_w")]
        if column is not None:
            if column not in preds.columns:
                raise ValueError(f"operating_point: column {column!r} not found; available score columns: {scores}")
            return column
        if len(scores) != 1:
            raise ValueError(f"operating_point: {len(scores)} score columns {scores}; pass column= to disambiguate")
        return scores[0]

    def woe_history(self) -> list[dict]:
        """Per-refit WoE state (bin edges + WoE table + IV) across the walk-forward."""
        self._check_fitted()
        return getattr(self, "refits_", [])

    def dump_woe_history(self, path: str) -> str:
        """Serialize the refit timeline (binning + statistics) to portable JSON."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.woe_history(), fh, default=str)
        return path

    def _build_fingerprint(self, data: Dataset, ts_unique: list, n_folds_effective: "int | None" = None) -> None:
        prefix, tail = self.features.split()
        tail_cfg = tail.to_config() if tail.transforms else None
        self.fingerprint = build_fingerprint(
            backend=self.backend if isinstance(self.backend, str) else type(self.backend).__name__,
            backend_params=self.backend_params,
            target_cfg=self.target.to_config(),
            features_cfg=self.features.to_config(),
            tail_cfg=tail_cfg,
            dataset_params=data.source_params,
            cv={
                **self.cv.to_config(),
                "n_folds_effective": n_folds_effective,
                "purge": self.target.horizon_bars(data),
                "embargo": self.target.horizon_bars(data),
                "span": [str(ts_unique[0]), str(ts_unique[-1])] if ts_unique else None,
            },
            output=self.output,
        )
        self.feature_signature = {
            "features": self.features.to_config(),
            "raw_columns": list(prefix.outputs),
            "model_columns": list(getattr(self.model_, "_sf_cols", []) or []),
            "output": self.output,
            "warmup": self.features.warmup,
        }

    def save(self, uri: str) -> str:
        from signalflow.model.store import save_model

        saved = save_model(self, uri)
        self._uri = saved
        return saved

    @classmethod
    def load(cls, uri: str, trust_remote: bool = False) -> "ForecastModel":
        """Load a saved model; ``hf://`` artifacts unpickle remote code and need ``trust_remote=True``."""
        from signalflow.model.store import load_model

        return load_model(uri, trust_remote=trust_remote)

    def __repr__(self) -> str:
        state = "fitted" if self.is_fitted else "unfitted"
        return f"ForecastModel(backend={self.backend!r}, output={self.output!r}, {state})"
