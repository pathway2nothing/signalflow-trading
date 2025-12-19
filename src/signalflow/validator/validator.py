from signalflow.core.containers.raw_data import RawData
from signalflow.core.signals import Signals
import polars as pl


class SignalValidator:
    def fit(signals: Signals, X: pl.DataFrame, y: pl.DataFrame) -> None:
        pass

    def predict(signals: Signals, X: pl.DataFrame) -> Signals:
        pass
