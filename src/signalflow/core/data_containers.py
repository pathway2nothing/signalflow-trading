import polars as pl
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RawData:
    datetime_start: datetime
    datetime_end: datetime    
    pairs: list[str] = field(default_factory=list)
    
    spot: pl.DataFrame
    
    #TODO: add other data sources


@dataclass
class Signals:
    values: pl.DataFrame
    strength: pl.DataFrame

    def __add__(self, other):
        pass   