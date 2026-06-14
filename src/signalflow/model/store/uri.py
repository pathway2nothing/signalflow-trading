"""URI resolution for model artifacts."""


from signalflow.errors import ArtifactError

_SCHEMES = ("file", "mlflow", "hf")


def resolve_uri(uri: str) -> tuple[str, str]:
    """Return ``(scheme, location)`` for a model artifact URI."""
    if not isinstance(uri, str) or not uri.strip():
        raise ArtifactError(f"invalid artifact uri: {uri!r}")
    raw = uri.strip()

    if "://" in raw:
        scheme, _, location = raw.partition("://")
        scheme = scheme.lower()
        if scheme not in _SCHEMES:
            raise ArtifactError(
                f"unknown artifact scheme {scheme!r} in {uri!r}; "
                f"expected one of {_SCHEMES} or a bare local path"
            )
        if not location:
            raise ArtifactError(f"empty location in artifact uri {uri!r}")
        return scheme, location


    return "file", raw
