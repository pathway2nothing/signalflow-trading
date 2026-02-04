"""Tests for signalflow.utils.kwargs_mixin.KwargsTolerantMixin."""

import pytest
from dataclasses import dataclass

from signalflow.utils.kwargs_mixin import KwargsTolerantMixin


# The mixin wraps __init__ in __init_subclass__ and uses dataclass fields()
# to identify known vs unknown kwargs. It requires:
#   1. An explicit __init__ in the class (so __init_subclass__ can wrap it)
#   2. The class to be a dataclass (so fields() works)
#
# For @dataclass classes, __init_subclass__ fires BEFORE @dataclass generates
# __init__, so direct dataclass subclasses don't get wrapping.
#
# For non-dataclass classes with explicit __init__ + **kwargs, the mixin's
# fields() call raises TypeError, falling through to original __init__.


class TolerantWithKwargs(KwargsTolerantMixin):
    """Non-dataclass with **kwargs - mixin falls through to orig __init__."""
    def __init__(self, x: int = 0, y: str = "hello", **kwargs):
        self.x = x
        self.y = y


@dataclass
class DataclassChild(KwargsTolerantMixin):
    """Direct @dataclass - wrapping doesn't apply (decorator order)."""
    x: int = 0
    y: str = "hello"


class TestKwargsTolerantMixinWithExplicitInit:
    def test_known_kwargs(self):
        obj = TolerantWithKwargs(x=42, y="world")
        assert obj.x == 42
        assert obj.y == "world"

    def test_unknown_kwargs_absorbed_by_explicit_kwargs(self):
        """With explicit **kwargs in __init__, unknown params are absorbed by Python."""
        obj = TolerantWithKwargs(x=1, unknown_param=999)
        assert obj.x == 1
        assert not hasattr(obj, "unknown_param")

    def test_no_kwargs(self):
        obj = TolerantWithKwargs()
        assert obj.x == 0
        assert obj.y == "hello"


class TestDataclassDirectChild:
    def test_known_kwargs(self):
        obj = DataclassChild(x=42, y="world")
        assert obj.x == 42
        assert obj.y == "world"

    def test_defaults(self):
        obj = DataclassChild()
        assert obj.x == 0
        assert obj.y == "hello"

    def test_unknown_kwargs_raise(self):
        """Direct @dataclass child: no mixin wrapping, standard TypeError."""
        with pytest.raises(TypeError):
            DataclassChild(x=1, unknown=999)


class TestMixinClassAttributes:
    def test_default_flags(self):
        assert KwargsTolerantMixin.__ignore_unknown_kwargs__ is True
        assert KwargsTolerantMixin.__log_unknown_kwargs__ is True
        assert KwargsTolerantMixin.__strict_unknown_kwargs__ is False
