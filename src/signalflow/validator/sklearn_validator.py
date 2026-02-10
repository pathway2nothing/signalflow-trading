"""Sklearn-based signal validator implementations.

Provides specialized validators for different sklearn-compatible models:
- LightGBMValidator: Gradient boosting (fast, good defaults)
- XGBoostValidator: Gradient boosting (robust)
- RandomForestValidator: Ensemble of decision trees
- LogisticRegressionValidator: Linear classifier
- SVMValidator: Support vector machine

Each validator has model-specific defaults and tuning spaces.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar
from pathlib import Path
import pickle

import numpy as np
import polars as pl

from signalflow.core import sf_component, Signals
from signalflow.utils import import_model_class, build_optuna_params
from signalflow.validator.base import SignalValidator


# ---------------------------------------------------------------------------
# Base class for sklearn-compatible validators
# ---------------------------------------------------------------------------


@dataclass
class SklearnValidatorBase(SignalValidator):
    """Base class for sklearn-compatible signal validators.

    Provides common functionality for feature extraction, fitting,
    prediction, and serialization. Subclasses define model-specific
    configuration via class variables.

    Class Variables (override in subclasses):
        _model_class: Import path for the model class (e.g., "lightgbm.LGBMClassifier")
        _default_params: Default model parameters
        _tune_space: Optuna tuning space definition
        _supports_early_stopping: Whether the model supports early stopping
    """

    # Class variables - override in subclasses
    _model_class: ClassVar[str] = ""
    _default_params: ClassVar[dict[str, Any]] = {}
    _tune_space: ClassVar[dict[str, tuple]] = {}
    _supports_early_stopping: ClassVar[bool] = False

    # Tuning configuration
    tune_metric: str = "roc_auc"
    tune_cv_folds: int = 5
    tune_n_trials: int = 50
    tune_timeout: int = 600

    # Early stopping (for boosting models)
    early_stopping_rounds: int = 50

    def __post_init__(self) -> None:
        if self.model_params is None:
            self.model_params = {}
        if self.train_params is None:
            self.train_params = {}
        if self.tune_params is None:
            self.tune_params = {
                "n_trials": self.tune_n_trials,
                "cv_folds": self.tune_cv_folds,
                "timeout": self.tune_timeout,
            }

    def _create_model(self, params: dict | None = None) -> Any:
        """Create model instance with given parameters."""
        model_class = import_model_class(self._model_class)

        final_params = {**self._default_params}
        if self.model_params:
            final_params.update(self.model_params)
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

    def _get_early_stopping_kwargs(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, Any]:
        """Get early stopping fit kwargs. Override in subclasses."""
        return {}

    def fit(
        self,
        X_train: pl.DataFrame,
        y_train: pl.DataFrame | pl.Series,
        X_val: pl.DataFrame | None = None,
        y_val: pl.DataFrame | pl.Series | None = None,
    ) -> "SklearnValidatorBase":
        """Train the validator.

        Note: Filter to active signals BEFORE calling this method.

        Args:
            X_train: Training features (already filtered to active signals)
            y_train: Training labels
            X_val: Validation features (optional, for early stopping)
            y_val: Validation labels (optional)

        Returns:
            Self for method chaining
        """
        X_np = self._extract_features(X_train, fit_mode=True)
        y_np = self._extract_labels(y_train)

        self.model = self._create_model()

        fit_kwargs: dict[str, Any] = {}

        if X_val is not None and y_val is not None and self._supports_early_stopping:
            X_val_np = self._extract_features(X_val)
            y_val_np = self._extract_labels(y_val)
            fit_kwargs = self._get_early_stopping_kwargs(X_val_np, y_val_np)

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

        X_np = self._extract_features(X_train, fit_mode=True)
        y_np = self._extract_labels(y_train)

        n_trials = self.tune_params.get("n_trials", self.tune_n_trials)
        cv_folds = self.tune_params.get("cv_folds", self.tune_cv_folds)
        timeout = self.tune_params.get("timeout", self.tune_timeout)

        def objective(trial: optuna.Trial) -> float:
            params = build_optuna_params(trial, self._tune_space)
            model = self._create_model(params)

            scores = cross_val_score(
                model,
                X_np,
                y_np,
                cv=cv_folds,
                scoring=self.tune_metric,
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

        best_params = {**self._default_params, **study.best_params}
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
            New Signals with probability columns added
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        signals_df = signals.value
        classes = self._get_class_labels()

        X_matched = signals_df.select([self.pair_col, self.ts_col]).join(
            X,
            on=[self.pair_col, self.ts_col],
            how="left",
        )

        X_np = self._extract_features(X_matched)
        probas = self.model.predict_proba(X_np)

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

        Args:
            signals: Input Signals container
            features: Features DataFrame with (pair, timestamp) + features
            prefix: Prefix for probability columns (default: "probability_")

        Returns:
            New Signals with probability columns added.
        """
        return self.predict_proba(signals, features)

    def _get_class_labels(self) -> list[str]:
        """Get class labels for probability columns."""
        if self.model is None:
            raise ValueError("Model not fitted.")

        classes = getattr(self.model, "classes_", None)
        if classes is None:
            return ["none", "rise", "fall"]

        _legacy_map = {0: "none", 1: "rise", 2: "fall"}
        return [_legacy_map.get(c, str(c)) for c in classes]

    def save(self, path: str | Path) -> None:
        """Save validator to file."""
        path = Path(path)

        state = {
            "validator_class": self.__class__.__name__,
            "model": self.model,
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
    def load(cls, path: str | Path) -> "SklearnValidatorBase":
        """Load validator from file."""
        path = Path(path)

        with open(path, "rb") as f:
            state = pickle.load(f)

        validator = cls(
            model=state["model"],
            model_params=state["model_params"],
            train_params=state["train_params"],
            tune_params=state["tune_params"],
            feature_columns=state["feature_columns"],
            pair_col=state.get("pair_col", "pair"),
            ts_col=state.get("ts_col", "timestamp"),
        )

        return validator


# ---------------------------------------------------------------------------
# LightGBM Validator
# ---------------------------------------------------------------------------


@dataclass
@sf_component(name="validator/lightgbm")
class LightGBMValidator(SklearnValidatorBase):
    """LightGBM-based signal validator.

    Gradient boosting model optimized for speed and performance.
    Supports early stopping with validation data.

    Example:
        >>> validator = LightGBMValidator(n_estimators=200)
        >>> validator.fit(X_train, y_train, X_val, y_val)
        >>> validated = validator.validate_signals(signals, features)
    """

    _model_class: ClassVar[str] = "lightgbm.LGBMClassifier"
    _default_params: ClassVar[dict[str, Any]] = {
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
    }
    _tune_space: ClassVar[dict[str, tuple]] = {
        "n_estimators": ("int", 50, 500),
        "max_depth": ("int", 3, 12),
        "learning_rate": ("log_float", 0.01, 0.3),
        "num_leaves": ("int", 15, 127),
        "min_child_samples": ("int", 5, 100),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0),
    }
    _supports_early_stopping: ClassVar[bool] = True

    # Model-specific parameters (can be overridden at init)
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    num_leaves: int = 31

    def __post_init__(self) -> None:
        super().__post_init__()
        # Merge instance params into model_params
        self.model_params = {
            **self.model_params,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
        }

    def _get_early_stopping_kwargs(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, Any]:
        import lightgbm

        return {
            "eval_set": [(X_val, y_val)],
            "callbacks": [lightgbm.early_stopping(self.early_stopping_rounds, verbose=False)],
        }


# ---------------------------------------------------------------------------
# XGBoost Validator
# ---------------------------------------------------------------------------


@dataclass
@sf_component(name="validator/xgboost")
class XGBoostValidator(SklearnValidatorBase):
    """XGBoost-based signal validator.

    Robust gradient boosting with regularization.
    Supports early stopping with validation data.

    Example:
        >>> validator = XGBoostValidator(n_estimators=200)
        >>> validator.fit(X_train, y_train, X_val, y_val)
        >>> validated = validator.validate_signals(signals, features)
    """

    _model_class: ClassVar[str] = "xgboost.XGBClassifier"
    _default_params: ClassVar[dict[str, Any]] = {
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
    }
    _tune_space: ClassVar[dict[str, tuple]] = {
        "n_estimators": ("int", 50, 500),
        "max_depth": ("int", 3, 12),
        "learning_rate": ("log_float", 0.01, 0.3),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0),
        "min_child_weight": ("int", 1, 10),
        "gamma": ("float", 0, 0.5),
    }
    _supports_early_stopping: ClassVar[bool] = True

    # Model-specific parameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        self.model_params = {
            **self.model_params,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
        }

    def _get_early_stopping_kwargs(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "eval_set": [(X_val, y_val)],
            "early_stopping_rounds": self.early_stopping_rounds,
            "verbose": False,
        }


# ---------------------------------------------------------------------------
# Random Forest Validator
# ---------------------------------------------------------------------------


@dataclass
@sf_component(name="validator/random_forest")
class RandomForestValidator(SklearnValidatorBase):
    """Random Forest-based signal validator.

    Ensemble of decision trees with bagging.

    Example:
        >>> validator = RandomForestValidator(n_estimators=200)
        >>> validator.fit(X_train, y_train)
        >>> validated = validator.validate_signals(signals, features)
    """

    _model_class: ClassVar[str] = "sklearn.ensemble.RandomForestClassifier"
    _default_params: ClassVar[dict[str, Any]] = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    }
    _tune_space: ClassVar[dict[str, tuple]] = {
        "n_estimators": ("int", 50, 300),
        "max_depth": ("int", 5, 30),
        "min_samples_split": ("int", 2, 20),
        "min_samples_leaf": ("int", 1, 10),
    }
    _supports_early_stopping: ClassVar[bool] = False

    # Model-specific parameters
    n_estimators: int = 100
    max_depth: int = 10

    def __post_init__(self) -> None:
        super().__post_init__()
        self.model_params = {
            **self.model_params,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
        }


# ---------------------------------------------------------------------------
# Logistic Regression Validator
# ---------------------------------------------------------------------------


@dataclass
@sf_component(name="validator/logistic_regression")
class LogisticRegressionValidator(SklearnValidatorBase):
    """Logistic Regression-based signal validator.

    Linear classifier with regularization.

    Example:
        >>> validator = LogisticRegressionValidator(C=0.1)
        >>> validator.fit(X_train, y_train)
        >>> validated = validator.validate_signals(signals, features)
    """

    _model_class: ClassVar[str] = "sklearn.linear_model.LogisticRegression"
    _default_params: ClassVar[dict[str, Any]] = {
        "C": 1.0,
        "max_iter": 1000,
        "random_state": 42,
        "n_jobs": -1,
    }
    _tune_space: ClassVar[dict[str, tuple]] = {
        "C": ("log_float", 1e-4, 100),
        "penalty": ("categorical", ["l1", "l2"]),
        "solver": ("categorical", ["saga"]),
    }
    _supports_early_stopping: ClassVar[bool] = False

    # Model-specific parameters
    C: float = 1.0
    max_iter: int = 1000

    def __post_init__(self) -> None:
        super().__post_init__()
        self.model_params = {
            **self.model_params,
            "C": self.C,
            "max_iter": self.max_iter,
        }


# ---------------------------------------------------------------------------
# SVM Validator
# ---------------------------------------------------------------------------


@dataclass
@sf_component(name="validator/svm")
class SVMValidator(SklearnValidatorBase):
    """Support Vector Machine-based signal validator.

    SVM classifier with RBF kernel by default.

    Example:
        >>> validator = SVMValidator(C=10.0, kernel="rbf")
        >>> validator.fit(X_train, y_train)
        >>> validated = validator.validate_signals(signals, features)
    """

    _model_class: ClassVar[str] = "sklearn.svm.SVC"
    _default_params: ClassVar[dict[str, Any]] = {
        "C": 1.0,
        "kernel": "rbf",
        "probability": True,
        "random_state": 42,
    }
    _tune_space: ClassVar[dict[str, tuple]] = {
        "C": ("log_float", 1e-3, 100),
        "kernel": ("categorical", ["rbf", "linear", "poly"]),
        "gamma": ("categorical", ["scale", "auto"]),
    }
    _supports_early_stopping: ClassVar[bool] = False

    # Model-specific parameters
    C: float = 1.0
    kernel: str = "rbf"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.model_params = {
            **self.model_params,
            "C": self.C,
            "kernel": self.kernel,
        }


# ---------------------------------------------------------------------------
# Auto-Select Validator (backward compatibility)
# ---------------------------------------------------------------------------

AUTO_SELECT_VALIDATORS = [
    LightGBMValidator,
    XGBoostValidator,
    RandomForestValidator,
    LogisticRegressionValidator,
]


@dataclass
@sf_component(name="validator/auto")
class AutoSelectValidator(SklearnValidatorBase):
    """Auto-selecting signal validator.

    Automatically selects the best model via cross-validation.
    Tests LightGBM, XGBoost, Random Forest, and Logistic Regression.

    Example:
        >>> validator = AutoSelectValidator()
        >>> validator.fit(X_train, y_train)  # Selects best model
        >>> print(validator.selected_validator)  # Shows which was selected
    """

    _model_class: ClassVar[str] = ""
    _default_params: ClassVar[dict[str, Any]] = {}
    _tune_space: ClassVar[dict[str, tuple]] = {}
    _supports_early_stopping: ClassVar[bool] = False

    auto_select_metric: str = "roc_auc"
    auto_select_cv_folds: int = 5

    selected_validator: SklearnValidatorBase | None = field(default=None, repr=False)

    def fit(
        self,
        X_train: pl.DataFrame,
        y_train: pl.DataFrame | pl.Series,
        X_val: pl.DataFrame | None = None,
        y_val: pl.DataFrame | pl.Series | None = None,
    ) -> "AutoSelectValidator":
        """Train the validator, auto-selecting the best model."""
        from sklearn.model_selection import cross_val_score

        X_np = self._extract_features(X_train, fit_mode=True)
        y_np = self._extract_labels(y_train)

        best_score = -np.inf
        best_validator_cls = None

        for validator_cls in AUTO_SELECT_VALIDATORS:
            try:
                validator = validator_cls()
                model = validator._create_model()

                scores = cross_val_score(
                    model,
                    X_np,
                    y_np,
                    cv=self.auto_select_cv_folds,
                    scoring=self.auto_select_metric,
                    n_jobs=-1,
                )
                mean_score = scores.mean()

                if mean_score > best_score:
                    best_score = mean_score
                    best_validator_cls = validator_cls

            except ImportError:
                continue
            except Exception:
                continue

        if best_validator_cls is None:
            raise RuntimeError("No suitable model found. Install lightgbm, xgboost, or scikit-learn.")

        # Create and fit the selected validator
        self.selected_validator = best_validator_cls(feature_columns=self.feature_columns)
        self.selected_validator.fit(X_train, y_train, X_val, y_val)
        self.model = self.selected_validator.model

        return self

    def predict(self, signals: Signals, X: pl.DataFrame) -> Signals:
        if self.selected_validator is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.selected_validator.predict(signals, X)

    def predict_proba(self, signals: Signals, X: pl.DataFrame) -> Signals:
        if self.selected_validator is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.selected_validator.predict_proba(signals, X)

    def tune(self, *args, **kwargs) -> dict[str, Any]:
        raise NotImplementedError("AutoSelectValidator does not support tune(). Use a specific validator.")


# ---------------------------------------------------------------------------
# Backward compatibility alias
# ---------------------------------------------------------------------------

# Keep SklearnSignalValidator as alias for backward compatibility
SklearnSignalValidator = AutoSelectValidator
