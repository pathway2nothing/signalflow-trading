"""Every registered target declares its forward look-ahead explicitly (no attribute-name heuristics)."""

import dataclasses

import pytest

import signalflow as sf
from signalflow.target.labeler import Labeler


def _declared(target) -> int:
    field_name = type(target).horizon_field
    if field_name is None:
        return 1
    value = getattr(target, field_name)
    return int(max(value)) if isinstance(value, (tuple, list)) else int(value)


@pytest.fixture(scope="module")
def ds():
    return sf.data("synthetic", pairs=["BTCUSDT"], start="2024-01-01", end="2024-01-08", interval="1h")


@pytest.mark.parametrize("name", sf.registry.list(sf.ComponentType.TARGET))
def test_registered_target_declares_horizon(name, ds):
    cls = sf.registry.get(sf.ComponentType.TARGET, name)
    target = cls()
    if isinstance(target, Labeler):
        field_name = type(target).horizon_field
        assert field_name is None or field_name in {f.name for f in dataclasses.fields(target)}, (
            f"{cls.__name__}.horizon_field={field_name!r} names no dataclass field"
        )
        assert target.horizon_bars(ds) == _declared(target)
    else:
        assert target.horizon_bars(ds) >= 1


def test_misdeclared_horizon_field_fails_loudly():
    @dataclasses.dataclass
    class Broken(Labeler):
        horizon_field = "bars_ahead"

        def compute_group(self, group_df, data_context=None):
            return group_df

    with pytest.raises(TypeError, match="names no field"):
        _ = Broken().horizon
