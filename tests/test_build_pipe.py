"""build_pipe registry-name -> validated FeaturePipe tests."""

import polars as pl
import pytest
from loguru import logger

from signalflow.data import data
from signalflow.decorators import feature
from signalflow.errors import PipeError, UnknownComponentError
from signalflow.transform import FeaturePipe, build_pipe
from signalflow.transform.base import Feature


@feature("boom_feature")
class _BoomFeature(Feature):
    """Deliberately-broken feature that raises on compute (for drop/raise tests)."""

    @property
    def outputs(self) -> list[str]:
        return ["boom"]

    def exprs(self) -> list[pl.Expr]:
        raise RuntimeError("boom feature cannot compute")


@pytest.fixture(scope="module")
def sample():
    return data("memory", pairs=["BTCUSDT"], start="2023-01-01", interval="1h")


def test_builds_from_sma():
    pipe = build_pipe(["sma"])
    assert isinstance(pipe, FeaturePipe)
    assert "sma_20" in pipe.outputs


def test_unknown_name_raises():
    with pytest.raises(UnknownComponentError):
        build_pipe(["definitely_not_a_feature"])


def test_broken_transform_raises_by_default(sample):
    with pytest.raises(PipeError):
        build_pipe(["sma", "boom_feature"], data=sample)


def test_broken_transform_dropped_with_warning(sample):
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        pipe = build_pipe(["sma", "boom_feature"], data=sample, on_error="drop")
    finally:
        logger.remove(sink_id)
    assert "sma_20" in pipe.outputs
    assert "boom" not in pipe.outputs
    assert any("boom_feature" in m for m in messages)
