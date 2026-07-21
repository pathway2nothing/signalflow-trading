"""Content-addressed on-disk cache for computed feature frames."""

from pathlib import Path

import polars as pl

from signalflow._hash import code_fingerprint, stable_hash
from signalflow.data.dataset import Dataset
from signalflow.transform.pipe import FeaturePipe


class FeatureStore:
    """Reuses computed feature frames keyed by pipe config + code + data identity."""

    def __init__(self, root: "str | Path") -> None:
        self.root = Path(root)

    def key(self, features: FeaturePipe, data: Dataset) -> str:
        frame = data.frame
        return stable_hash(
            {
                "pipe": features.to_config(),
                "code": [code_fingerprint(type(t)) for t in features.transforms],
                "source": data.source_params,
                "span": [str(frame.get_column("ts").min()), str(frame.get_column("ts").max())],
                "rows": frame.height,
                "pairs": sorted(frame.get_column("pair").unique().to_list()),
            }
        )

    def compute(self, features: FeaturePipe, data: Dataset) -> pl.DataFrame:
        path = self.root / f"{self.key(features, data).replace(':', '_')}.parquet"
        if path.exists():
            return pl.read_parquet(path)
        frame = features.compute(data.frame)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        frame.write_parquet(tmp)
        tmp.replace(path)
        return frame
