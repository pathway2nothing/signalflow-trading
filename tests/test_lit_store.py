"""The Lightning AI model store (``lit://``): URI parsing and the artifact round-trip."""

import shutil
import sys
import types
from pathlib import Path

import pytest

import signalflow as sf

# --- lit:// (Lightning AI model store) -------------------------------------------------


def test_lit_uri_parsing_and_teamspace(monkeypatch):
    from signalflow.model.store import resolve_uri
    from signalflow.model.store.lit_store import _parse_location

    assert resolve_uri("lit://models/m1") == ("lit", "models/m1")
    monkeypatch.setenv("LIGHTNING_TEAMSPACE", "my-team")
    assert _parse_location("models/m1") == ("my-team/m1", None)
    assert _parse_location("m1@4") == ("my-team/m1", "4")
    assert _parse_location("models/other/m1") == ("other/m1", None)  # explicit teamspace wins
    monkeypatch.delenv("LIGHTNING_TEAMSPACE")
    with pytest.raises(sf.ArtifactError, match="teamspace"):
        _parse_location("models/m1")


def test_lit_store_round_trips_the_artifact_layout(monkeypatch, tmp_path, fitted_forecast):
    """save/load go through litmodels; the layout written and read is the framework's own."""
    from signalflow.model.store import lit_store

    monkeypatch.setenv("LIGHTNING_TEAMSPACE", "my-team")
    uploaded: dict = {}

    class _Info:
        version = "7"

    def fake_upload(name, path, **kw):
        uploaded["name"] = name
        shutil.copytree(path, tmp_path / "remote", dirs_exist_ok=True)
        return _Info()

    def fake_download(name, download_dir, **kw):
        uploaded["loaded"] = name
        shutil.copytree(tmp_path / "remote", Path(download_dir) / "nested", dirs_exist_ok=True)

    fake = types.SimpleNamespace(upload_model_files=fake_upload, download_model=fake_download)
    monkeypatch.setattr(lit_store, "_litmodels", lambda: fake)

    uri = lit_store.save(fitted_forecast, "models/exp002_rise_202401")
    assert uri == "lit://models/my-team/exp002_rise_202401@7"
    assert uploaded["name"] == "my-team/exp002_rise_202401"
    assert (tmp_path / "remote" / "model.pkl").exists()

    restored = lit_store.load("models/exp002_rise_202401@7")
    assert uploaded["loaded"] == "my-team/exp002_rise_202401:7"
    assert restored.output == fitted_forecast.output
    assert restored.is_fitted


def test_lit_store_without_litmodels(monkeypatch):
    from signalflow.model.store import lit_store

    monkeypatch.setitem(sys.modules, "litmodels", None)
    with pytest.raises(sf.ArtifactError, match="litmodels"):
        lit_store._litmodels()
