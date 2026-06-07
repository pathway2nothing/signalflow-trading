"""Tests for ModelRef parsing/validation, MlflowResolver lazy load, and caching."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from signalflow.models import (
    CachingModelRegistry,
    MlflowResolver,
    ModelRef,
)

# ── ModelRef parsing ──────────────────────────────────────────────────────


def test_parse_mlflow_uri() -> None:
    ref = ModelRef.parse("models:/revert/3")
    assert ref == ModelRef(name="revert", version="3", source="mlflow")
    assert ref.uri == "models:/revert/3"


def test_parse_at_spec() -> None:
    ref = ModelRef.parse("revert@3")
    assert ref.name == "revert"
    assert ref.version == "3"
    assert ref.source == "mlflow"


def test_parse_at_spec_with_source() -> None:
    ref = ModelRef.parse("revert@3", source="hf")
    assert ref.source == "hf"


@pytest.mark.parametrize("spec", ["", "   ", "garbage", "models:/onlyname", "name@", "@3"])
def test_parse_rejects_malformed(spec: str) -> None:
    with pytest.raises(ValueError):
        ModelRef.parse(spec)


# ── ModelRef validation ───────────────────────────────────────────────────


def test_empty_name_raises() -> None:
    with pytest.raises(ValueError, match="name"):
        ModelRef(name="", version="3")


def test_empty_version_raises() -> None:
    with pytest.raises(ValueError, match="version"):
        ModelRef(name="revert", version="")


def test_invalid_source_raises() -> None:
    with pytest.raises(ValueError, match="source"):
        ModelRef(name="revert", version="3", source="s3")


def test_int_version_allowed() -> None:
    ref = ModelRef(name="revert", version=3)
    assert ref.uri == "models:/revert/3"


# ── latest gating ─────────────────────────────────────────────────────────


def test_latest_forbidden_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SF_ALLOW_LATEST", raising=False)
    with pytest.raises(ValueError, match=r"parity|latest"):
        ModelRef(name="revert", version="latest")


def test_latest_allowed_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SF_ALLOW_LATEST", "1")
    ref = ModelRef(name="revert", version="latest")
    assert ref.version == "latest"


def test_latest_parsed_forbidden_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SF_ALLOW_LATEST", raising=False)
    with pytest.raises(ValueError):
        ModelRef.parse("models:/revert/latest")


# ── frozen / hashable ─────────────────────────────────────────────────────


def test_frozen() -> None:
    ref = ModelRef(name="revert", version="3")
    with pytest.raises(FrozenInstanceError):
        ref.name = "other"  # type: ignore[misc]


def test_hashable_dict_key() -> None:
    a = ModelRef(name="revert", version="3")
    b = ModelRef(name="revert", version="3")
    c = ModelRef(name="turb", version="1")
    d: dict[ModelRef, str] = {a: "x"}
    assert d[b] == "x"  # equal refs collide
    d[c] = "y"
    assert len(d) == 2


# ── CachingModelRegistry ──────────────────────────────────────────────────


class _CountingResolver:
    """Fake Resolver returning a marker; counts resolve calls."""

    def __init__(self) -> None:
        self.calls = 0
        self.marker = object()

    def resolve(self, ref: ModelRef) -> object:
        self.calls += 1
        return self.marker


def test_caching_registry_resolves_once() -> None:
    resolver = _CountingResolver()
    reg = CachingModelRegistry(resolver)
    ref = ModelRef(name="revert", version="3")

    assert reg.has(ref) is False
    first = reg.get(ref)
    second = reg.get(ref)

    assert first is resolver.marker
    assert second is resolver.marker
    assert resolver.calls == 1  # resolved once, cached for the 2nd get
    assert reg.has(ref) is True


def test_caching_registry_distinct_refs() -> None:
    resolver = _CountingResolver()
    reg = CachingModelRegistry(resolver)
    reg.get(ModelRef(name="revert", version="3"))
    reg.get(ModelRef(name="turb", version="1"))
    assert resolver.calls == 2


# ── MlflowResolver (no real mlflow) ───────────────────────────────────────


def test_mlflow_resolver_builds_uri_and_is_lazy() -> None:
    captured: list[str] = []
    sentinel = object()

    class _StubResolver(MlflowResolver):
        def _load(self, uri: str) -> object:
            captured.append(uri)
            return sentinel

    resolver = _StubResolver()
    ref = ModelRef(name="revert", version="3")
    result = resolver.resolve(ref)

    assert result is sentinel
    assert captured == ["models:/revert/3"]


def test_mlflow_resolver_rejects_non_mlflow_source() -> None:
    class _StubResolver(MlflowResolver):
        def _load(self, uri: str) -> object:  # pragma: no cover - must not run
            raise AssertionError("should not load")

    resolver = _StubResolver()
    with pytest.raises(ValueError, match="mlflow"):
        resolver.resolve(ModelRef(name="m", version="1", source="hf"))


def test_import_does_not_require_mlflow() -> None:
    # If this module imported, signalflow.models imported cleanly already.
    import signalflow.models  # noqa: F401
