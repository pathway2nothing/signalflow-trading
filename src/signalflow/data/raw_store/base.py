from abc import ABC, abstractmethod

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, ClassVar
from signalflow.core import SfComponentType
import polars as pl
import pandas as pd

@dataclass
class RawDataStore(ABC):
    component_type: ClassVar[SfComponentType] = SfComponentType.RAW_DATA_STORE
    
    @abstractmethod
    def load(self, pair: str, hours: Optional[int] = None, start: Optional[datetime] = None, end: Optional[datetime] = None) -> pl.DataFrame:
        pass

    @abstractmethod
    def load_many(self, pairs: list[str], hours: Optional[int] = None, start: Optional[datetime] = None, end: Optional[datetime] = None) -> pl.DataFrame:
        pass

    @abstractmethod
    def load_many_pandas(self, pairs: list[str], start: Optional[datetime] = None, end: Optional[datetime] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def close(self) -> None:
        pass