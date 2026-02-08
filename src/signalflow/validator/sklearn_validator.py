# IMPORTANT

from dataclasses import dataclass
from typing import Any
from pathlib import Path
import pickle

import numpy as np
import polars as pl

from signalflow.core import sf_component, Signals
from signalflow.utils import import_model_class, build_optuna_params
from signalflow.validator.base import SignalValidator

SKLEARN_MODELS: dict[str, dict[str, Any]] = {
    "lightgbm": {
        "class": "lightgbm.LGBMClassifier",
        "default_params": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        },
        "tune_space": {
            "n_estimators": ("int", 50, 500),
            "max_depth": ("int", 3, 12),
            "learning_rate": ("log_float", 0.01, 0.3),
            "num_leaves": ("int", 15, 127),
            "min_child_samples": ("int", 5, 100),
            "subsample": ("float", 0.6, 1.0),
            "colsample_bytree": ("float", 0.6, 1.0),
        },
    },
    "xgboost": {
        "class": "xgboost.XGBClassifier",
        "default_params": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
            "use_label_encoder": False,
            "eval_metric": "logloss",
        },
        "tune_space": {
            "n_estimators": ("int", 50, 500),
            "max_depth": ("int", 3, 12),
            "learning_rate": ("log_float", 0.01, 0.3),
            "subsample": ("float", 0.6, 1.0),
            "colsample_bytree": ("float", 0.6, 1.0),
            "min_child_weight": ("int", 1, 10),
            "gamma": ("float", 0, 0.5),
        },
    },
    "random_forest": {
        "class": "sklearn.ensemble.RandomForestClassifier",
        "default_params": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        },
        "tune_space": {
            "n_estimators": ("int", 50, 300),
            "max_depth": ("int", 5, 30),
            "min_samples_split": ("int", 2, 20),
            "min_samples_leaf": ("int", 1, 10),
        },
    },
    "logistic_regression": {
        "class": "sklearn.linear_model.LogisticRegression",
        "default_params": {
            "C": 1.0,
            "max_iter": 1000,
            "random_state": 42,
            "n_jobs": -1,
        },
        "tune_space": {
            "C": ("log_float", 1e-4, 100),
            "penalty": ("categorical", ["l1", "l2"]),
            "solver": ("categorical", ["saga"]),
        },
    },
    "svm": {
        "class": "sklearn.svm.SVC",
        "default_params": {
            "C": 1.0,
            "kernel": "rbf",
            "probability": True,
            "random_state": 42,
        },
        "tune_space": {
            "C": ("log_float", 1e-3, 100),
            "kernel": ("categorical", ["rbf", "linear", "poly"]),
            "gamma": ("categorical", ["scale", "auto"]),
        },
    },
}

AUTO_SELECT_MODELS = ["lightgbm", "xgboost", "random_forest", "logistic_regression"]


@dataclass
@sf_component(name="sklearn")
class SklearnSignalValidator(SignalValidator):
    """Sklearn-based signal validator.

    Supports:
    - Multiple sklearn-compatible models (LightGBM, XGBoost, RF, etc.)
    - Automatic model selection via cross-validation
    - Hyperparameter tuning with Optuna
    - Early stopping for boosting models

    Note: Filter data to active signals (not NONE) BEFORE calling fit().
    This gives you full control over data preparation.

    Example:
        >>> # Prepare data - filter to active signals
        >>> df = df.filter(pl.col("signal_type") != "none")
        >>>
        >>> validator = SklearnSignalValidator(model_type="lightgbm")
        >>> validator.fit(
        ...     train_df.select(["pair", "timestamp"] + feature_cols),
        ...     train_df.select("label"),
        ... )
        >>>
        >>> # validate_signals returns Signals object
        >>> validated = validator.validate_signals(
        ...     Signals(test_df.select(signal_cols)),
        ...     test_df.select(["pair", "timestamp"] + feature_cols),
        ... )
        >>> validated.value.filter(pl.col("probability_rise") > 0.7)
    """

    auto_select_metric: str = "roc_auc"
    auto_select_cv_folds: int = 5

    def __post_init__(self) -> None:
        if self.model_params is None:
            self.model_params = {}
        if self.train_params is None:
            self.train_params = {}
        if self.tune_params is None:
            self.tune_params = {"n_trials": 50, "cv_folds": 5, "timeout": 600}

    def _get_model_config(self, model_type: str) -> dict[str, Any]:
        """Get model configuration from catalog."""
        if model_type not in SKLEARN_MODELS:
            available = ", ".join(SKLEARN_MODELS.keys())
            raise ValueError(f"Unknown model_type: {model_type}. Available: {available}")
        return SKLEARN_MODELS[model_type]

    def _create_model(self, model_type: str, params: dict | None = None) -> Any:
        """Create model instance."""
        config = self._get_model_config(model_type)
        model_class = import_model_class(config["class"])

        final_params = {**config["default_params"]}
        if params:
            final_params.update(params)

        return model_class(**final_params)

    def _extract_features(
        self,
        X: pl.DataFrame,
        fit_mode: bool = False,
    ) -> np.ndarray:
        """Extract feature matrix from DataFrame.

        Args:
            X: Input DataFrame
            fit_mode: If True, infer and store feature columns

        Returns:
            Feature matrix as numpy array
        """
        exclude_cols = {self.pair_col, self.ts_col}

        if fit_mode:
            self.feature_columns = [c for c in X.columns if c not in exclude_cols]

        if self.feature_columns is None:
            raise ValueError("feature_columns not set. Call fit() first.")

        missing = set(self.feature_columns) - set(X.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")

        return X.select(self.feature_columns).to_numpy()

    def _extract_labels(self, y: pl.DataFrame | pl.Series) -> np.ndarray:
        """Extract label array."""
        if isinstance(y, pl.DataFrame):
            if y.width == 1:
                return y.to_numpy().ravel()
            elif "label" in y.columns:
                return y["label"].to_numpy()
            else:
                raise ValueError("y DataFrame must have single column or 'label' column")
        return y.to_numpy()

    def _auto_select_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[str, dict]:
        """Select best model using cross-validation.

        Returns:
            Tuple of (best_model_type, best_params)
        """
        from sklearn.model_selection import cross_val_score

        best_score = -np.inf
        best_model_type = None
        best_params = None

        for model_type in AUTO_SELECT_MODELS:
            try:
                config = self._get_model_config(model_type)
                model = self._create_model(model_type)

                scores = cross_val_score(
                    model,
                    X,
                    y,
                    cv=self.auto_select_cv_folds,
                    scoring=self.auto_select_metric,
                    n_jobs=-1,
                )
                mean_score = scores.mean()

                if mean_score > best_score:
                    best_score = mean_score
                    best_model_type = model_type
                    best_params = config["default_params"].copy()

            except ImportError:
                continue
            except Exception:
                continue

        if best_model_type is None:
            raise RuntimeError("No suitable model found. Install lightgbm, xgboost, or scikit-learn.")

        return best_model_type, best_params

    def fit(
        self,
        X_train: pl.DataFrame,
        y_train: pl.DataFrame | pl.Series,
        X_val: pl.DataFrame | None = None,
        y_val: pl.DataFrame | pl.Series | None = None,
    ) -> "SklearnSignalValidator":
        """Train the validator.

        Note: Filter to active signals BEFORE calling this method.

        For boosting models with validation data, early stopping is applied.

        Args:
            X_train: Training features (already filtered to active signals)
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Self for method chaining
        """
        X_np = self._extract_features(X_train, fit_mode=True)
        y_np = self._extract_labels(y_train)

        if self.model_type == "auto" or self.model_type is None:
            self.model_type, self.model_params = self._auto_select_model(X_np, y_np)

        self.model = self._create_model(self.model_type, self.model_params)

        fit_kwargs: dict[str, Any] = {}

        if X_val is not None and y_val is not None:
            X_val_np = self._extract_features(X_val)
            y_val_np = self._extract_labels(y_val)

            if self.model_type in ("lightgbm", "xgboost"):
                early_stopping = self.train_params.get("early_stopping_rounds", 50)

                if self.model_type == "lightgbm":
                    fit_kwargs["eval_set"] = [(X_val_np, y_val_np)]
                    fit_kwargs["callbacks"] = [__import__("lightgbm").early_stopping(early_stopping, verbose=False)]
                elif self.model_type == "xgboost":
                    fit_kwargs["eval_set"] = [(X_val_np, y_val_np)]
                    fit_kwargs["early_stopping_rounds"] = early_stopping
                    fit_kwargs["verbose"] = False

        self.model.fit(X_np, y_np, **fit_kwargs)

        return self

    def tune(
        self,
        X_train: pl.DataFrame,
        y_train: pl.DataFrame | pl.Series,
        X_val: pl.DataFrame | None = None,
        y_val: pl.DataFrame | pl.Series | None = None,
    ) -> dict[str, Any]:
        """Tune hyperparameters using Optuna.

        Note: Filter to active signals BEFORE calling this method.

        Returns:
            Best parameters found
        """
        import optuna
        from sklearn.model_selection import cross_val_score

        if self.model_type is None or self.model_type == "auto":
            raise ValueError("Set model_type before tuning (not 'auto')")

        config = self._get_model_config(self.model_type)
        tune_space = config["tune_space"]

        X_np = self._extract_features(X_train, fit_mode=True)
        y_np = self._extract_labels(y_train)

        n_trials = self.tune_params.get("n_trials", 50)
        cv_folds = self.tune_params.get("cv_folds", 5)
        timeout = self.tune_params.get("timeout", 600)

        def objective(trial: optuna.Trial) -> float:
            params = build_optuna_params(trial, tune_space)
            params.update(config["default_params"])  # Base params

            model = self._create_model(self.model_type, params)

            scores = cross_val_score(
                model,
                X_np,
                y_np,
                cv=cv_folds,
                scoring=self.auto_select_metric,
                n_jobs=-1,
            )
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        best_params = {**config["default_params"], **study.best_params}
        self.model_params = best_params

        return best_params

    def predict(self, signals: Signals, X: pl.DataFrame) -> Signals:
        """Predict class labels and return updated Signals.

        Args:
            signals: Input signals container
            X: Features DataFrame with (pair, timestamp) + feature columns

        Returns:
            New Signals with 'validation_pred' column added
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        signals_df = signals.value

        # Join features to signals by keys
        X_matched = signals_df.select([self.pair_col, self.ts_col]).join(
            X,
            on=[self.pair_col, self.ts_col],
            how="left",
        )

        X_np = self._extract_features(X_matched)
        predictions = self.model.predict(X_np)

        result_df = signals_df.with_columns(pl.Series(name="validation_pred", values=predictions))

        return Signals(result_df)

    def predict_proba(self, signals: Signals, X: pl.DataFrame) -> Signals:
        """Predict class probabilities and return updated Signals.

        Args:
            signals: Input signals container
            X: Features DataFrame with (pair, timestamp) + feature columns

        Returns:
            New Signals with probability columns (probability_none, probability_rise, probability_fall)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        signals_df = signals.value
        classes = self._get_class_labels()

        # Join features to signals by keys
        X_matched = signals_df.select([self.pair_col, self.ts_col]).join(
            X,
            on=[self.pair_col, self.ts_col],
            how="left",
        )

        X_np = self._extract_features(X_matched)
        probas = self.model.predict_proba(X_np)

        # Add probability columns
        result_df = signals_df
        for i, class_label in enumerate(classes):
            col_name = f"probability_{class_label}"
            result_df = result_df.with_columns(pl.Series(name=col_name, values=probas[:, i]))

        return Signals(result_df)

    def validate_signals(
        self,
        signals: Signals,
        features: pl.DataFrame,
        prefix: str = "probability_",
    ) -> Signals:
        """Add validation probabilities to signals.

        Adds probability columns for each class:
        - probability_none: P(signal is noise / not actionable)
        - probability_rise: P(signal leads to price rise)
        - probability_fall: P(signal leads to price fall)

        Args:
            signals: Input Signals container
            features: Features DataFrame with (pair, timestamp) + features
            prefix: Prefix for probability columns (default: "probability_")

        Returns:
            New Signals with probability columns added.

        Example:
            >>> validated = validator.validate_signals(signals, features)
            >>> df = validated.value
            >>> confident_rise = df.filter(
            ...     (pl.col("signal_type") == "rise") &
            ...     (pl.col("probability_rise") > 0.7)
            ... )
        """
        return self.predict_proba(signals, features)

    def _get_class_labels(self) -> list[str]:
        """Get class labels for probability columns.

        Dynamically reads class labels from the fitted model. Falls back to
        a legacy mapping for backward compatibility with models trained on
        numeric labels (0/1/2).
        """
        if self.model is None:
            raise ValueError("Model not fitted.")

        classes = getattr(self.model, "classes_", None)
        if classes is None:
            return ["none", "rise", "fall"]

        # Legacy numeric class mapping (backward compat for old models)
        _legacy_map = {0: "none", 1: "rise", 2: "fall"}

        return [_legacy_map.get(c, str(c)) for c in classes]

    def save(self, path: str | Path) -> None:
        """Save validator to file."""
        path = Path(path)

        state = {
            "model": self.model,
            "model_type": self.model_type,
            "model_params": self.model_params,
            "train_params": self.train_params,
            "tune_params": self.tune_params,
            "feature_columns": self.feature_columns,
            "pair_col": self.pair_col,
            "ts_col": self.ts_col,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> "SklearnSignalValidator":
        """Load validator from file."""
        path = Path(path)

        with open(path, "rb") as f:
            state = pickle.load(f)

        validator = cls(
            model=state["model"],
            model_type=state["model_type"],
            model_params=state["model_params"],
            train_params=state["train_params"],
            tune_params=state["tune_params"],
            feature_columns=state["feature_columns"],
            pair_col=state.get("pair_col", "pair"),
            ts_col=state.get("ts_col", "timestamp"),
        )

        return validator
