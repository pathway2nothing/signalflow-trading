

from dataclasses import dataclass
from signalflow.feature.pandasta.pandas_ta_extractor import PandasTaExtractor


@dataclass
class PandasTaRsiExtractor(PandasTaExtractor):
    length: int = 14

    def __post_init__(self) -> None:
        self.indicator = "rsi"
        self.params = {"length": int(self.length)}
        self.input_column = "close"
        self.additional_inputs = {}
        self.feature_prefix = "rsi"
        super().__post_init__()


@dataclass
class PandasTaBbandsExtractor(PandasTaExtractor):
    length: int = 20
    std: float = 2.0

    def __post_init__(self) -> None:
        self.indicator = "bbands"
        self.params = {"length": int(self.length), "std": float(self.std)}
        self.input_column = "close"
        self.additional_inputs = {}
        self.feature_prefix = f"bbands_{int(self.length)}_{float(self.std)}"
        super().__post_init__()


@dataclass
class PandasTaMacdExtractor(PandasTaExtractor):
    fast: int = 12
    slow: int = 26
    signal: int = 9

    def __post_init__(self) -> None:
        self.indicator = "macd"
        self.params = {"fast": int(self.fast), "slow": int(self.slow), "signal": int(self.signal)}
        self.input_column = "close"
        self.additional_inputs = {}
        self.feature_prefix = f"macd_{int(self.fast)}_{int(self.slow)}_{int(self.signal)}"
        super().__post_init__()


@dataclass
class PandasTaAtrExtractor(PandasTaExtractor):
    length: int = 14

    def __post_init__(self) -> None:
        self.indicator = "atr"
        self.params = {"length": int(self.length)}
        self.input_column = "high"
        self.additional_inputs = {"low": "low", "close": "close"}
        self.feature_prefix = f"atr_{int(self.length)}"
        super().__post_init__()
