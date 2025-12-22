"""
Base Signal Validator for SignalFlow.

Signal validators (meta-labelers) predict the quality/risk of trading signals.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar
from pathlib import Path

import polars as pl
import numpy as np

from signalflow.core import SfComponentType, Signals


@dataclass
class SignalValidator:
    """Base class for signal validators (meta-labelers).
    
    Validates trading signals by predicting their risk/quality.
    In De Prado's terminology - this is a meta-labeler.
    
    Note: Filtering to active signals (RISE/FALL only) should be done
    BEFORE passing data to fit. This keeps the validator simple
    and gives users full control over data preparation.
    
    Attributes:
        model: The trained model instance
        model_type: String identifier for model type (e.g., "lightgbm", "xgboost")
        model_params: Parameters for model initialization
        train_params: Parameters for training (e.g., early stopping)
        tune_enabled: Whether hyperparameter tuning is enabled
        tune_params: Parameters for tuning (e.g., n_trials, cv_folds)
        feature_columns: List of feature column names (set after fit)
    """
    
    component_type: ClassVar[SfComponentType] = SfComponentType.VALIDATOR
    
    model: Any | None = None
    model_type: str | None = None
    model_params: dict | None = None
    
    train_params: dict | None = None
    tune_enabled: bool = False
    tune_params: dict | None = None
    
    feature_columns: list[str] | None = field(default=None, repr=False)
    pair_col: str = "pair"
    ts_col: str = "timestamp"

    def fit(
        self, 
        X_train: pl.DataFrame, 
        y_train: pl.DataFrame | pl.Series,
        X_val: pl.DataFrame | None = None, 
        y_val: pl.DataFrame | pl.Series | None = None,
    ) -> "SignalValidator":
        """Train the validator model.
        
        Args:
            X_train: Training features (Polars DataFrame)
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Self for method chaining
        """
        raise NotImplementedError("Subclasses must implement fit()")

    def tune(
        self, 
        X_train: pl.DataFrame, 
        y_train: pl.DataFrame | pl.Series,
        X_val: pl.DataFrame | None = None, 
        y_val: pl.DataFrame | pl.Series | None = None,
    ) -> dict[str, Any]:
        """Tune hyperparameters.
        
        Returns:
            Best parameters found
        """
        if not self.tune_enabled:
            raise ValueError("Tuning is not enabled for this validator")
        raise NotImplementedError("Subclasses must implement tune()")

    def predict(self, signals: Signals, X: pl.DataFrame) -> Signals:
        """Predict class labels and return updated Signals.
        
        Args:
            signals: Input signals container
            X: Features (Polars DataFrame) with (pair, timestamp) + feature columns
            
        Returns:
            New Signals with prediction column added
        """
        raise NotImplementedError("Subclasses must implement predict()")
    
    def predict_proba(self, signals: Signals, X: pl.DataFrame) -> Signals:
        """Predict class probabilities and return updated Signals.
        
        Args:
            signals: Input signals container  
            X: Features (Polars DataFrame)
            
        Returns:
            New Signals with probability columns added
        """
        raise NotImplementedError("Subclasses must implement predict_proba()")
    
    def validate_signals(
        self, 
        signals: Signals, 
        features: pl.DataFrame,
        prefix: str = "probability_",
    ) -> Signals:
        """Add validation predictions to signals.
        
        Convenience method - calls predict_proba internally.
        
        Args:
            signals: Input signals container
            features: Features DataFrame with (pair, timestamp) + feature columns
            prefix: Prefix for probability columns
            
        Returns:
            Signals with added validation columns
        """
        raise NotImplementedError("Subclasses must implement validate_signals()")
    
    def save(self, path: str | Path) -> None:
        """Save model to file."""
        raise NotImplementedError("Subclasses must implement save()")
    
    @classmethod
    def load(cls, path: str | Path) -> "SignalValidator":
        """Load model from file."""
        raise NotImplementedError("Subclasses must implement load()")