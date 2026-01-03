from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, ClassVar
from signalflow.core import SfComponentType
import polars as pl

@dataclass
class RawDataSource(ABC):
    component_type: ClassVar[SfComponentType] = SfComponentType.RAW_DATA_SOURCE


class  RawDataLoader(ABC):
    component_type: ClassVar[SfComponentType] = SfComponentType.RAW_DATA_LOADER

    @abstractmethod
    def download(self, **kwargs) :
        pass

    @abstractmethod
    def sync(self, **kwargs):
        pass