"""
Hugging Face Hub backend (real huggingface_hub).

URI form (``hf://`` already stripped to ``location``)::

<repo_id>[@revision]

Save uploads the artifact layout to a model repo (creating it if needed). Load
downloads a snapshot and reads the layout. Uploads require an ``HF_TOKEN`` in the
environment; without it we raise a clear ArtifactError.
"""


import os
import tempfile

from signalflow.errors import ArtifactError
from signalflow.model.store._layout import read_layout, write_layout


def _parse_location(location: str) -> tuple[str, str | None]:
    repo_id, sep, revision = location.strip().strip("/").partition("@")
    if not repo_id:
        raise ArtifactError(f"could not parse repo_id from hf location {location!r}")
    return repo_id, (revision if sep else None)


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def save(model, location: str) -> str:
    import huggingface_hub

    repo_id, revision = _parse_location(location)
    token = _token()
    if not token:
        raise ArtifactError(
            "uploading to Hugging Face requires an HF_TOKEN (or "
            "HUGGING_FACE_HUB_TOKEN) environment variable with write access."
        )

    try:
        huggingface_hub.create_repo(
            repo_id=repo_id, repo_type="model", token=token, exist_ok=True
        )
        with tempfile.TemporaryDirectory() as tmp:
            write_layout(model, tmp)
            huggingface_hub.upload_folder(
                repo_id=repo_id,
                folder_path=tmp,
                repo_type="model",
                token=token,
                revision=revision,
            )
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError(f"hf save failed for {location!r}: {exc}") from exc

    suffix = f"@{revision}" if revision else ""
    return f"hf://{repo_id}{suffix}"


def load(location: str):
    import huggingface_hub

    repo_id, revision = _parse_location(location)
    try:
        local = huggingface_hub.snapshot_download(
            repo_id=repo_id,
            revision=revision,
            repo_type="model",
            token=_token(),
        )
    except Exception as exc:
        raise ArtifactError(f"hf load failed for {location!r}: {exc}") from exc

    return read_layout(local)
