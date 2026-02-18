"""Artifact caching for Flow execution.

Provides MD5-based caching for intermediate artifacts following RawDataLazy pattern.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

if TYPE_CHECKING:
    from signalflow.config.dag import Node
    from signalflow.core import Signals

CacheMode = Literal["memory", "disk", "none"]


def _serialize_config(config: dict[str, Any]) -> str:
    """Serialize config dict to deterministic string."""
    return json.dumps(config, sort_keys=True, default=str)


@dataclass
class ArtifactCache:
    """Cache for intermediate artifacts in Flow execution.

    Uses MD5-based cache keys following RawDataLazy pattern.
    Supports memory and disk (parquet) caching modes.

    Attributes:
        cache_mode: "memory", "disk", or "none"
        cache_dir: Directory for disk cache (auto-created if None)
        max_memory_mb: Maximum memory cache size in MB (0 = unlimited)

    Example:
        >>> cache = ArtifactCache(cache_mode="disk", cache_dir="./cache")
        >>> result = flow.run(cache=cache)
        >>> # Second run uses cached artifacts
        >>> result2 = flow.run(cache=cache)
    """

    cache_mode: CacheMode = "memory"
    cache_dir: Path | None = None
    max_memory_mb: int = 0

    _memory_cache: dict[str, Any] = field(default_factory=dict, repr=False)
    _memory_size: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """Initialize cache directory."""
        if self.cache_mode == "disk" and self.cache_dir is None:
            import tempfile

            self.cache_dir = Path(tempfile.mkdtemp(prefix="signalflow_artifact_cache_"))
        elif self.cache_dir:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_key(self, node: Node, inputs: dict[str, Any]) -> str:
        """Generate MD5 cache key for a node execution.

        Key components:
        - Node ID
        - Node type
        - Node config (serialized)
        - Input artifact hashes

        Args:
            node: The node being executed
            inputs: Input artifacts to the node

        Returns:
            16-character hex cache key
        """
        key_parts = [
            node.id,
            node.type,
            node.name,
            _serialize_config(node.config),
        ]

        # Add input artifact signatures
        for input_name in sorted(inputs.keys()):
            input_data = inputs[input_name]
            key_parts.append(f"{input_name}:{self._data_signature(input_data)}")

        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    @staticmethod
    def _data_signature(data: Any) -> str:
        """Generate signature for input data."""
        if isinstance(data, pl.DataFrame):
            # Use shape + column names + first/last rows hash
            sig_parts = [
                str(data.shape),
                ",".join(data.columns),
            ]
            if data.height > 0:
                first = str(data.head(1).to_dicts())
                last = str(data.tail(1).to_dicts())
                sig_parts.extend([first, last])
            return hashlib.md5("|".join(sig_parts).encode()).hexdigest()[:8]

        if hasattr(data, "value"):  # Signals
            return ArtifactCache._data_signature(data.value)

        # Fallback: string representation
        return hashlib.md5(str(data).encode()).hexdigest()[:8]

    def get(self, node: Node, inputs: dict[str, Any]) -> dict[str, Any] | None:
        """Get cached outputs for a node.

        Args:
            node: The node being executed
            inputs: Input artifacts to the node

        Returns:
            Cached outputs dict or None if not cached
        """
        if self.cache_mode == "none":
            return None

        key = self.cache_key(node, inputs)

        # Check memory cache
        if self.cache_mode == "memory":
            return self._memory_cache.get(key)

        # Check disk cache
        if self.cache_mode == "disk" and self.cache_dir:
            cache_path = self.cache_dir / key
            if cache_path.exists():
                return self._load_from_disk(cache_path)

        return None

    def put(self, node: Node, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        """Cache outputs for a node.

        Args:
            node: The executed node
            inputs: Input artifacts to the node
            outputs: Output artifacts from the node
        """
        if self.cache_mode == "none":
            return

        key = self.cache_key(node, inputs)

        if self.cache_mode == "memory":
            self._memory_cache[key] = outputs

        elif self.cache_mode == "disk" and self.cache_dir:
            cache_path = self.cache_dir / key
            self._save_to_disk(cache_path, outputs)

    def _save_to_disk(self, path: Path, outputs: dict[str, Any]) -> None:
        """Save outputs to disk cache."""
        from signalflow.core import Signals

        path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {"outputs": list(outputs.keys())}
        (path / "metadata.json").write_text(json.dumps(metadata))

        # Save each output
        for name, data in outputs.items():
            if isinstance(data, pl.DataFrame):
                data.write_parquet(path / f"{name}.parquet")
            elif isinstance(data, Signals):
                data.value.write_parquet(path / f"{name}.signals.parquet")
            elif isinstance(data, dict):
                (path / f"{name}.json").write_text(json.dumps(data, default=str))

    def _load_from_disk(self, path: Path) -> dict[str, Any]:
        """Load outputs from disk cache."""
        from signalflow.core import Signals

        outputs: dict[str, Any] = {}

        metadata_path = path / "metadata.json"
        if not metadata_path.exists():
            return outputs

        metadata = json.loads(metadata_path.read_text())

        for name in metadata.get("outputs", []):
            parquet_path = path / f"{name}.parquet"
            signals_path = path / f"{name}.signals.parquet"
            json_path = path / f"{name}.json"

            if parquet_path.exists():
                outputs[name] = pl.read_parquet(parquet_path)
            elif signals_path.exists():
                outputs[name] = Signals(pl.read_parquet(signals_path))
            elif json_path.exists():
                outputs[name] = json.loads(json_path.read_text())

        return outputs

    def clear(self) -> None:
        """Clear all cached data."""
        self._memory_cache.clear()
        self._memory_size = 0

        if self.cache_mode == "disk" and self.cache_dir and self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def invalidate(self, node_id: str) -> int:
        """Invalidate cache entries for a specific node.

        Args:
            node_id: Node ID to invalidate

        Returns:
            Number of entries invalidated
        """
        count = 0

        if self.cache_mode == "memory":
            # Cache keys start with node_id
            keys_to_remove = [k for k in self._memory_cache if node_id in k]
            for key in keys_to_remove:
                del self._memory_cache[key]
                count += 1

        return count

    @property
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if self.cache_mode == "memory":
            return {
                "mode": "memory",
                "entries": len(self._memory_cache),
                "size_mb": self._memory_size / (1024 * 1024),
            }
        if self.cache_mode == "disk" and self.cache_dir:
            entries = list(self.cache_dir.iterdir()) if self.cache_dir.exists() else []
            return {
                "mode": "disk",
                "entries": len(entries),
                "path": str(self.cache_dir),
            }
        return {"mode": "none"}
