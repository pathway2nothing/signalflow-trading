"""Configuration loading utilities.

Provides unified config loading for SignalFlow flows.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries. Override values take precedence.

    Args:
        base: Base dictionary
        override: Dictionary with override values

    Returns:
        Merged dictionary

    Example:
        >>> deep_merge({"a": 1, "b": {"c": 2}}, {"b": {"d": 3}})
        {"a": 1, "b": {"c": 2, "d": 3}}
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Parsed YAML as dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_conf_path(conf_path: Path | str | None) -> Path:
    """Resolve conf path.

    Priority:
    1. Explicit conf_path argument
    2. SF_CONF_PATH environment variable
    3. Current working directory / conf / base

    Args:
        conf_path: Optional explicit path

    Returns:
        Resolved Path object
    """
    if conf_path is not None:
        return Path(conf_path)

    env_path = os.environ.get("SF_CONF_PATH")
    if env_path:
        return Path(env_path)

    return Path.cwd() / "conf" / "base"


def _resolve_env_vars(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve environment variables in config values.

    Replaces ${VAR_NAME} patterns with environment variable values.

    Args:
        config: Configuration dictionary

    Returns:
        Config with resolved environment variables
    """
    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = _resolve_env_vars(value)
        elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            var_name = value[2:-1]
            result[key] = os.environ.get(var_name, value)
        else:
            result[key] = value
    return result


def load_flow_config(
    flow_id: str,
    conf_path: Path | str | None = None,
    *,
    resolve_env: bool = True,
) -> dict[str, Any]:
    """Load flow configuration by ID.

    Merges common defaults with flow-specific config.

    Args:
        flow_id: Flow identifier (e.g., 'grid_sma', 'baseline_sma')
        conf_path: Path to conf directory (contains parameters/, flows/)
        resolve_env: Whether to resolve ${VAR} environment variables

    Returns:
        Merged configuration dictionary

    Raises:
        FileNotFoundError: If flow config doesn't exist

    Example:
        >>> config = load_flow_config("grid_sma", conf_path="./conf/base")
        >>> config["flow_id"]
        'grid_sma'
    """
    conf_path = _resolve_conf_path(conf_path)
    logger.debug(f"Loading flow config from: {conf_path}")

    # Load common config
    common_path = conf_path / "parameters" / "common.yml"
    common_config: dict[str, Any] = {}
    if common_path.exists():
        common_config = load_yaml(common_path)
        logger.debug(f"Loaded common config from {common_path}")

    # Load flow-specific config
    flow_path = conf_path / "flows" / f"{flow_id}.yml"
    if not flow_path.exists():
        raise FileNotFoundError(f"Flow config not found: {flow_path}")

    flow_config = load_yaml(flow_path)
    logger.info(f"Loaded flow config: {flow_id}")

    # Merge configs: flow overrides common defaults
    defaults = common_config.get("defaults", {})
    merged = deep_merge(defaults, flow_config)

    # Add telegram config if present
    if "telegram" in common_config:
        merged["telegram"] = common_config["telegram"]

    # Resolve output paths with flow_id
    output = common_config.get("output", {})
    if output:
        merged["output"] = {
            "signals": output.get("signals", "data/08_reporting/{flow_id}/signals").format(flow_id=flow_id),
            "strategy": output.get("strategy", "data/08_reporting/{flow_id}/strategy").format(flow_id=flow_id),
            "db": output.get("db", "data/07_model_output/strategy_{flow_id}.duckdb").format(flow_id=flow_id),
        }

    # Ensure flow_id is set
    merged["flow_id"] = flow_id

    # Resolve environment variables
    if resolve_env:
        merged = _resolve_env_vars(merged)

    return merged


def list_flows(conf_path: Path | str | None = None) -> list[str]:
    """List available flow configurations.

    Args:
        conf_path: Path to conf directory

    Returns:
        List of flow IDs (sorted alphabetically)

    Example:
        >>> list_flows("./conf/base")
        ['baseline_sma', 'grid_rsi', 'grid_sma', ...]
    """
    conf_path = _resolve_conf_path(conf_path)
    flows_dir = conf_path / "flows"

    if not flows_dir.exists():
        logger.warning(f"Flows directory not found: {flows_dir}")
        return []

    flows = sorted(f.stem for f in flows_dir.glob("*.yml"))
    logger.debug(f"Found {len(flows)} flows in {flows_dir}")
    return flows


def get_flow_info(flow_id: str, conf_path: Path | str | None = None) -> dict[str, str]:
    """Get flow metadata (name, description).

    Args:
        flow_id: Flow identifier
        conf_path: Path to conf directory

    Returns:
        Dict with flow_id, flow_name, description
    """
    config = load_flow_config(flow_id, conf_path, resolve_env=False)
    return {
        "flow_id": config.get("flow_id", flow_id),
        "flow_name": config.get("flow_name", flow_id),
        "description": config.get("description", ""),
    }
