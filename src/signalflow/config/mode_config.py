"""Mode-specific configuration handling.

This module implements the SF Flow specification for mode-specific configurations:
- Single file format with train/optimize/backtest/live sections
- Modular format with extends inheritance
- Deep merge with special operators (!, +)
- Parameter resolution (${VAR} and ${param:default})

Resolution Algorithm:
1. Detect format (single file vs modular)
2. Load base configuration
3. Load mode-specific overlay
4. Merge with special operators
5. Resolve parameters
6. Validate

Example:
    >>> from signalflow.config import resolve_flow_config, FlowMode
    >>> config = resolve_flow_config("flows/grid_sma.yml", mode=FlowMode.BACKTEST)
    >>> # Or for modular:
    >>> config = resolve_flow_config("flows/grid_sma/backtest.yml")
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


class FlowMode(StrEnum):
    """Flow execution modes."""

    BACKTEST = "backtest"
    TRAIN = "train"
    OPTIMIZE = "optimize"
    LIVE = "live"


class ConfigFormat(StrEnum):
    """Configuration file format."""

    SINGLE_FILE = "single_file"  # Has mode sections (train/backtest/etc)
    MODULAR_CHILD = "modular_child"  # Has 'extends' field
    MODULAR_BASE = "modular_base"  # Directory with flow.yml + mode files


# Mode sections recognized in single-file format
MODE_SECTIONS = {"train", "optimize", "backtest", "live"}

# Regex for parameter resolution
PARAM_PATTERN = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


@dataclass
class ModeConfig:
    """Mode-specific configuration overlay.

    Attributes:
        mode: Execution mode
        data: Mode-specific data config (dates, splits)
        capital: Initial capital (backtest/live)
        costs: Trading costs (backtest)
        models: Model loading/saving config
        params: Parameter file or values
        labeling: Labeling config (train)
        output: Output directory config
        search_space: Optimization search space
        optimizer: Optimizer config
        objective: Optimization objective
        exchange: Exchange connection (live)
        risk: Risk management (live)
        hooks: Event hooks (live)
        state: State persistence (live)
        signal_metrics: Signal metrics to compute
        strategy_metrics: Strategy metrics to compute
        validators: Validator training config (train)
    """

    mode: FlowMode
    data: dict[str, Any] = field(default_factory=dict)
    capital: dict[str, Any] = field(default_factory=dict)
    costs: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    labeling: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    search_space: dict[str, Any] = field(default_factory=dict)
    optimizer: dict[str, Any] = field(default_factory=dict)
    objective: dict[str, Any] = field(default_factory=dict)
    exchange: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    hooks: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    signal_metrics: list[dict[str, Any]] = field(default_factory=list)
    strategy_metrics: list[dict[str, Any]] = field(default_factory=list)
    validators: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, mode: FlowMode, data: dict[str, Any]) -> ModeConfig:
        """Create ModeConfig from dict."""
        return cls(
            mode=mode,
            data=data.get("data", {}),
            capital=data.get("capital", {}),
            costs=data.get("costs", {}),
            models=data.get("models", {}),
            params=data.get("params", {}),
            labeling=data.get("labeling", {}),
            output=data.get("output", {}),
            search_space=data.get("search_space", {}),
            optimizer=data.get("optimizer", {}),
            objective=data.get("objective", {}),
            exchange=data.get("exchange", {}),
            risk=data.get("risk", {}),
            hooks=data.get("hooks", {}),
            state=data.get("state", {}),
            signal_metrics=data.get("signal_metrics", []),
            strategy_metrics=data.get("strategy_metrics", []),
            validators=data.get("validators", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict."""
        result: dict[str, Any] = {}
        if self.data:
            result["data"] = self.data
        if self.capital:
            result["capital"] = self.capital
        if self.costs:
            result["costs"] = self.costs
        if self.models:
            result["models"] = self.models
        if self.params:
            result["params"] = self.params
        if self.labeling:
            result["labeling"] = self.labeling
        if self.output:
            result["output"] = self.output
        if self.search_space:
            result["search_space"] = self.search_space
        if self.optimizer:
            result["optimizer"] = self.optimizer
        if self.objective:
            result["objective"] = self.objective
        if self.exchange:
            result["exchange"] = self.exchange
        if self.risk:
            result["risk"] = self.risk
        if self.hooks:
            result["hooks"] = self.hooks
        if self.state:
            result["state"] = self.state
        if self.signal_metrics:
            result["signal_metrics"] = self.signal_metrics
        if self.strategy_metrics:
            result["strategy_metrics"] = self.strategy_metrics
        if self.validators:
            result["validators"] = self.validators
        return result


def detect_format(config: dict[str, Any], path: Path | None = None) -> ConfigFormat:
    """Detect configuration file format.

    Args:
        config: Parsed YAML config
        path: Optional path for directory detection

    Returns:
        ConfigFormat enum value
    """
    # Check for 'extends' field (modular child)
    if "extends" in config:
        return ConfigFormat.MODULAR_CHILD

    # Check for mode sections (single file)
    if any(section in config for section in MODE_SECTIONS):
        return ConfigFormat.SINGLE_FILE

    # Check if path is directory with flow.yml (modular base)
    if path and path.is_dir() and (path / "flow.yml").exists():
        return ConfigFormat.MODULAR_BASE

    # Default to single file (base without mode sections)
    return ConfigFormat.SINGLE_FILE


def deep_merge_with_operators(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Deep merge with special operators.

    Operators:
    - key!: Replace (don't merge)
    - key+: Append (for arrays)

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration
    """
    result = base.copy()

    for key, value in override.items():
        # Check for replace operator (!)
        if key.endswith("!"):
            actual_key = key[:-1]
            result[actual_key] = value
            continue

        # Check for append operator (+)
        if key.endswith("+"):
            actual_key = key[:-1]
            if actual_key in result and isinstance(result[actual_key], list):
                result[actual_key] = result[actual_key] + value
            else:
                result[actual_key] = value
            continue

        # Normal merge
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge_with_operators(result[key], value)
            else:
                result[key] = value
        else:
            result[key] = value

    return result


def resolve_params(
    config: dict[str, Any],
    params: dict[str, Any] | None = None,
    env_vars: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve parameters in configuration.

    Patterns:
    - ${VAR}: Environment variable
    - ${param:default}: Parameter with default value

    Args:
        config: Configuration dictionary
        params: Optional parameter overrides
        env_vars: Optional environment variable overrides

    Returns:
        Config with resolved parameters
    """
    params = params or {}
    env_vars = env_vars or dict(os.environ)

    def resolve_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: resolve_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve_value(v) for v in value]
        if isinstance(value, str):
            return resolve_string(value)
        return value

    def resolve_string(s: str) -> Any:
        match = PARAM_PATTERN.fullmatch(s)
        if match:
            param_name = match.group(1)
            default_value = match.group(2)

            # Check params first
            if param_name in params:
                return params[param_name]

            # Check environment variables
            if param_name in env_vars:
                return env_vars[param_name]

            # Use default if provided
            if default_value is not None:
                # Try to parse as number
                try:
                    if "." in default_value:
                        return float(default_value)
                    return int(default_value)
                except ValueError:
                    return default_value

            # Return original if no resolution
            return s

        # Handle multiple params in string
        def replace_param(match: re.Match[str]) -> str:
            param_name = match.group(1)
            default_value = match.group(2)

            if param_name in params:
                return str(params[param_name])
            if param_name in env_vars:
                return env_vars[param_name]
            if default_value is not None:
                return default_value
            return match.group(0)

        return PARAM_PATTERN.sub(replace_param, s)

    return resolve_value(config)


def load_params_from_file(params_path: Path | str) -> dict[str, Any]:
    """Load parameters from a YAML file.

    Args:
        params_path: Path to parameters file

    Returns:
        Parameters dictionary
    """
    path = Path(params_path)
    if not path.exists():
        logger.warning(f"Parameters file not found: {path}")
        return {}

    with open(path) as f:
        return yaml.safe_load(f) or {}


def extract_base_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract base configuration (non-mode sections).

    Args:
        config: Full configuration

    Returns:
        Base configuration without mode sections
    """
    return {k: v for k, v in config.items() if k not in MODE_SECTIONS}


def extract_mode_config(config: dict[str, Any], mode: FlowMode) -> dict[str, Any]:
    """Extract mode-specific configuration section.

    Args:
        config: Full configuration
        mode: Target mode

    Returns:
        Mode-specific configuration
    """
    return config.get(mode.value, {})


def resolve_extends_chain(
    config: dict[str, Any],
    base_path: Path,
    visited: set[str] | None = None,
) -> dict[str, Any]:
    """Resolve extends chain for modular configs.

    Args:
        config: Child configuration
        base_path: Base directory for relative paths
        visited: Set of visited paths (cycle detection)

    Returns:
        Merged configuration with all parents
    """
    visited = visited or set()
    extends = config.get("extends")

    if not extends:
        return config

    # Resolve extends path
    extends_path = base_path / extends
    extends_key = str(extends_path.resolve())

    if extends_key in visited:
        raise ValueError(f"Circular extends detected: {extends_key}")

    visited.add(extends_key)

    # Load parent config
    if not extends_path.exists():
        raise FileNotFoundError(f"Extends file not found: {extends_path}")

    with open(extends_path) as f:
        parent_config = yaml.safe_load(f) or {}

    # Recursively resolve parent's extends
    parent_config = resolve_extends_chain(
        parent_config,
        extends_path.parent,
        visited,
    )

    # Remove extends from child before merge
    child_config = {k: v for k, v in config.items() if k != "extends"}

    # Merge: child overrides parent
    return deep_merge_with_operators(parent_config, child_config)


def resolve_flow_config(
    path: str | Path,
    mode: FlowMode | str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve flow configuration with mode-specific overlay.

    This implements the Resolution Algorithm from SFFLOW.md:
    1. Detect format (single file vs modular)
    2. Load base configuration
    3. Load mode-specific overlay
    4. Merge with special operators
    5. Resolve parameters
    6. (Validation happens separately)

    Args:
        path: Path to flow configuration file or directory
        mode: Execution mode (optional for modular format)
        params: Optional parameter overrides

    Returns:
        Fully resolved configuration dictionary

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If mode required but not specified

    Example:
        >>> # Single file with mode
        >>> config = resolve_flow_config("flows/grid_sma.yml", mode="backtest")

        >>> # Modular format (mode inferred from filename)
        >>> config = resolve_flow_config("flows/grid_sma/backtest.yml")

        >>> # Directory with explicit mode
        >>> config = resolve_flow_config("flows/grid_sma/", mode="train")
    """
    path = Path(path)
    if isinstance(mode, str):
        mode = FlowMode(mode)

    # Step 1: Detect format
    if path.is_dir():
        # Modular format: directory
        return _resolve_modular_directory(path, mode, params)

    if not path.exists():
        raise FileNotFoundError(f"Flow config not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f) or {}

    fmt = detect_format(config, path)

    if fmt == ConfigFormat.MODULAR_CHILD:
        # Modular format: child file with extends
        return _resolve_modular_child(config, path, params)

    # Single file format
    return _resolve_single_file(config, mode, params)


def _resolve_single_file(
    config: dict[str, Any],
    mode: FlowMode | None,
    params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve single-file format configuration.

    Args:
        config: Loaded configuration
        mode: Target mode
        params: Parameter overrides

    Returns:
        Resolved configuration
    """
    # Step 2: Extract base config
    base = extract_base_config(config)

    # Step 3: Extract mode-specific overlay
    if mode:
        mode_overlay = extract_mode_config(config, mode)
        # Step 4: Merge
        result = deep_merge_with_operators(base, mode_overlay)
        result["_mode"] = mode.value
    else:
        result = base

    # Load params from file if specified
    all_params = params or {}
    if "params" in result and "from" in result["params"]:
        file_params = load_params_from_file(result["params"]["from"])
        all_params = {**file_params, **all_params}

    # Step 5: Resolve parameters
    return resolve_params(result, all_params)


def _resolve_modular_directory(
    directory: Path,
    mode: FlowMode | None,
    params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve modular format from directory.

    Args:
        directory: Flow directory
        mode: Target mode
        params: Parameter overrides

    Returns:
        Resolved configuration
    """
    if not mode:
        raise ValueError("Mode required for directory-based modular format")

    # Load base flow.yml
    base_path = directory / "flow.yml"
    if not base_path.exists():
        raise FileNotFoundError(f"Base flow.yml not found in {directory}")

    with open(base_path) as f:
        base_config = yaml.safe_load(f) or {}

    # Load mode-specific file
    mode_path = directory / f"{mode.value}.yml"
    if mode_path.exists():
        with open(mode_path) as f:
            mode_config = yaml.safe_load(f) or {}

        # Resolve extends if present
        if "extends" in mode_config:
            mode_config = resolve_extends_chain(mode_config, directory)

        # Merge
        result = deep_merge_with_operators(base_config, mode_config)
    else:
        result = base_config

    result["_mode"] = mode.value

    # Load params from file if specified
    all_params = params or {}
    if "params" in result and "from" in result["params"]:
        params_path = directory / result["params"]["from"]
        if params_path.exists():
            file_params = load_params_from_file(params_path)
            all_params = {**file_params, **all_params}

    return resolve_params(result, all_params)


def _resolve_modular_child(
    config: dict[str, Any],
    path: Path,
    params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve modular format from child file.

    Args:
        config: Child configuration
        path: Path to child file
        params: Parameter overrides

    Returns:
        Resolved configuration
    """
    # Resolve extends chain
    result = resolve_extends_chain(config, path.parent)

    # Infer mode from filename
    mode_name = path.stem
    if mode_name in [m.value for m in FlowMode]:
        result["_mode"] = mode_name

    # Load params from file if specified
    all_params = params or {}
    if "params" in result and "from" in result["params"]:
        params_path = path.parent / result["params"]["from"]
        if params_path.exists():
            file_params = load_params_from_file(params_path)
            all_params = {**file_params, **all_params}

    return resolve_params(result, all_params)


@dataclass
class ResolvedFlowConfig:
    """Fully resolved flow configuration.

    This is the result of the resolution algorithm, ready for execution.
    """

    flow_id: str
    flow_name: str
    mode: FlowMode | None
    version: str = "1.0.0"

    # Core configuration
    data: dict[str, Any] = field(default_factory=dict)
    detectors: list[dict[str, Any]] = field(default_factory=list)
    validators: list[dict[str, Any]] = field(default_factory=list)
    strategy: dict[str, Any] = field(default_factory=dict)

    # Mode-specific configuration
    mode_config: ModeConfig | None = None

    # Auxiliary
    signal_metrics: list[dict[str, Any]] = field(default_factory=list)
    strategy_metrics: list[dict[str, Any]] = field(default_factory=list)
    hooks: dict[str, Any] = field(default_factory=dict)

    # Raw configuration
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> ResolvedFlowConfig:
        """Create from resolved config dict."""
        mode_str = config.get("_mode")
        mode = FlowMode(mode_str) if mode_str else None

        # Extract mode config
        mode_config = None
        if mode:
            mode_config = ModeConfig.from_dict(mode, config)

        return cls(
            flow_id=config.get("flow_id", "unknown"),
            flow_name=config.get("flow_name", config.get("flow_id", "unknown")),
            mode=mode,
            version=config.get("version", "1.0.0"),
            data=config.get("data", {}),
            detectors=config.get("detectors", []),
            validators=config.get("validators", []),
            strategy=config.get("strategy", {}),
            mode_config=mode_config,
            signal_metrics=config.get("signal_metrics", []),
            strategy_metrics=config.get("strategy_metrics", []),
            hooks=config.get("hooks", {}),
            raw=config,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        mode: FlowMode | str | None = None,
        params: dict[str, Any] | None = None,
    ) -> ResolvedFlowConfig:
        """Load and resolve flow configuration.

        Args:
            path: Path to config file or directory
            mode: Execution mode
            params: Parameter overrides

        Returns:
            Resolved configuration
        """
        config = resolve_flow_config(path, mode, params)
        return cls.from_dict(config)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict."""
        return self.raw
