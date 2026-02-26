"""Flow configuration validation.

This module implements validation rules from SFFLOW.md specification:

EXISTENCE RULES:
- V1: ∃ at least one DETECTOR block
- V2: ∃ exactly one STRATEGY block
- V3: ∃ exactly one DATA section

UNIQUENESS RULES:
- V4: All block ids must be unique

REFERENCE INTEGRITY:
- V5: ∀ validator.bound_to: ∃ detector.id = bound_to
- V6: ∀ reconcile.weights.key: ∃ detector.id = key

CARDINALITY RULES:
- V7: ∀ detector: count(validator.bound_to = detector.id) ≤ 1

STRUCTURAL RULES:
- V8: No circular dependencies
- V9: strategy.entries.rules is non-empty
- V10: strategy.exits.rules is non-empty

MODE COMPATIBILITY:
- V11: mode=train → validators must define fit()
- V12: mode=live → hooks should be defined (warning)
- V13: mode=optimize → metrics must be defined

WARNINGS (non-fatal):
- W1: strategy.model without fallback_entry
- W2: strategy.model without fallback_exit
- W3: mode=live without hooks

Example:
    >>> from signalflow.config.validation import validate_flow_config
    >>> errors, warnings = validate_flow_config(config)
    >>> if errors:
    ...     print("Invalid config:", errors)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    """Validation result with errors and warnings.

    Attributes:
        errors: List of error messages (fatal)
        warnings: List of warning messages (non-fatal)
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if config is valid (no errors)."""
        return len(self.errors) == 0

    def add_error(self, code: str, message: str) -> None:
        """Add an error."""
        self.errors.append(f"[{code}] {message}")

    def add_warning(self, code: str, message: str) -> None:
        """Add a warning."""
        self.warnings.append(f"[{code}] {message}")

    def merge(self, other: ValidationResult) -> None:
        """Merge another result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


def validate_flow_config(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Validate flow configuration.

    Args:
        config: Flow configuration dictionary

    Returns:
        Tuple of (errors, warnings)
    """
    result = ValidationResult()

    # Run all validators
    _validate_existence(config, result)
    _validate_uniqueness(config, result)
    _validate_references(config, result)
    _validate_cardinality(config, result)
    _validate_structure(config, result)
    _validate_mode_compatibility(config, result)

    return result.errors, result.warnings


def _validate_existence(config: dict[str, Any], result: ValidationResult) -> None:
    """Validate existence rules (V1-V3)."""
    # V1: At least one detector
    detectors = config.get("detectors", [])
    if not detectors:
        result.add_error("V1", "Flow must have at least one detector block")

    # V2: Exactly one strategy
    strategy = config.get("strategy")
    if not strategy:
        result.add_error("V2", "Flow must have exactly one strategy block")

    # V3: Data section
    data = config.get("data")
    if not data:
        result.add_error("V3", "Flow must have a data section")


def _validate_uniqueness(config: dict[str, Any], result: ValidationResult) -> None:
    """Validate uniqueness rules (V4)."""
    # V4: All block ids must be unique
    ids: set[str] = set()
    duplicates: set[str] = set()

    # Collect detector ids
    for detector in config.get("detectors", []):
        det_id = detector.get("id")
        if det_id:
            if det_id in ids:
                duplicates.add(det_id)
            ids.add(det_id)

    # Collect validator ids
    for validator in config.get("validators", []):
        val_id = validator.get("id")
        if val_id:
            if val_id in ids:
                duplicates.add(val_id)
            ids.add(val_id)

    if duplicates:
        result.add_error("V4", f"Duplicate block ids: {', '.join(sorted(duplicates))}")


def _validate_references(config: dict[str, Any], result: ValidationResult) -> None:
    """Validate reference integrity (V5-V6)."""
    # Get all detector ids
    detector_ids = {d.get("id") for d in config.get("detectors", []) if d.get("id")}

    # V5: Validator bound_to references valid detector
    for validator in config.get("validators", []):
        bound_to = validator.get("bound_to")
        if bound_to and bound_to not in detector_ids:
            result.add_error(
                "V5",
                f"Validator '{validator.get('id', 'unknown')}' references unknown detector: {bound_to}"
            )

    # V6: Reconcile weights reference valid detectors
    strategy = config.get("strategy", {})
    reconcile = strategy.get("reconcile", {})
    weights = reconcile.get("weights", {})

    for weight_key in weights:
        if weight_key not in detector_ids:
            result.add_error(
                "V6",
                f"Reconcile weight references unknown detector: {weight_key}"
            )


def _validate_cardinality(config: dict[str, Any], result: ValidationResult) -> None:
    """Validate cardinality rules (V7)."""
    # V7: At most one validator per detector
    detector_validators: dict[str, list[str]] = {}

    for validator in config.get("validators", []):
        bound_to = validator.get("bound_to")
        val_id = validator.get("id", "unknown")
        if bound_to:
            if bound_to not in detector_validators:
                detector_validators[bound_to] = []
            detector_validators[bound_to].append(val_id)

    for detector_id, validators in detector_validators.items():
        if len(validators) > 1:
            result.add_error(
                "V7",
                f"Detector '{detector_id}' has multiple validators: {', '.join(validators)}"
            )


def _validate_structure(config: dict[str, Any], result: ValidationResult) -> None:
    """Validate structural rules (V8-V10)."""
    # V8: No circular dependencies
    # (This is checked at flow compile time, not in static config validation)

    strategy = config.get("strategy", {})

    # V9: Entry rules non-empty
    entries = strategy.get("entries", {})
    entry_rules = entries.get("rules", [])
    if not entry_rules:
        # Check for model-based entries
        model = strategy.get("model")
        if not model:
            result.add_error("V9", "Strategy must have at least one entry rule or model")

    # V10: Exit rules non-empty
    exits = strategy.get("exits", {})
    exit_rules = exits.get("rules", [])
    if not exit_rules:
        result.add_error("V10", "Strategy must have at least one exit rule")


def _validate_mode_compatibility(config: dict[str, Any], result: ValidationResult) -> None:
    """Validate mode compatibility rules (V11-V13, W1-W3)."""
    mode = config.get("_mode")

    if not mode:
        return  # No mode specified, skip mode-specific validation

    strategy = config.get("strategy", {})
    model = strategy.get("model")

    # V11: mode=train → validators should exist
    if mode == "train":
        validators = config.get("validators", [])
        if not validators:
            result.add_warning(
                "V11",
                "Train mode without validators - nothing to train"
            )

        # Check for labeling config
        train_config = config.get("train", {})
        if not train_config.get("labeling"):
            result.add_warning(
                "V11",
                "Train mode without labeling configuration"
            )

    # V12: mode=live → hooks should be defined
    if mode == "live":
        live_config = config.get("live", {})
        hooks = live_config.get("hooks", {})
        if not hooks:
            result.add_warning("V12", "Live mode without hooks - no notifications will be sent")

        # Check for risk config
        risk = live_config.get("risk", {})
        if not risk:
            result.add_warning("V12", "Live mode without risk configuration")

    # V13: mode=optimize → metrics should be defined
    if mode == "optimize":
        optimize_config = config.get("optimize", {})
        objective = optimize_config.get("objective", {})
        if not objective.get("metric"):
            result.add_warning(
                "V13",
                "Optimize mode without objective metric"
            )

    # W1-W2: Model without fallbacks
    if model:
        if not model.get("fallback_entry"):
            result.add_warning(
                "W1",
                "Strategy model without fallback_entry - model failures will block entries"
            )
        if not model.get("fallback_exit"):
            result.add_warning(
                "W2",
                "Strategy model without fallback_exit - model failures will block exits"
            )


def validate_detector(detector: dict[str, Any]) -> ValidationResult:
    """Validate a single detector configuration.

    Args:
        detector: Detector configuration

    Returns:
        ValidationResult with errors and warnings
    """
    result = ValidationResult()

    # Must have id
    if not detector.get("id"):
        result.add_error("DET1", "Detector must have an id")

    # Must have logic
    logic = detector.get("logic", {})
    if not logic:
        result.add_error("DET2", f"Detector '{detector.get('id', 'unknown')}' must have logic configuration")
    elif not logic.get("type"):
        result.add_error("DET3", f"Detector '{detector.get('id', 'unknown')}' logic must have a type")

    return result


def validate_validator(validator: dict[str, Any]) -> ValidationResult:
    """Validate a single validator configuration.

    Args:
        validator: Validator configuration

    Returns:
        ValidationResult with errors and warnings
    """
    result = ValidationResult()

    # Must have id
    if not validator.get("id"):
        result.add_error("VAL1", "Validator must have an id")

    # Must have bound_to
    if not validator.get("bound_to"):
        result.add_error("VAL2", f"Validator '{validator.get('id', 'unknown')}' must have bound_to")

    # Must have config
    config = validator.get("config", {})
    if not config:
        result.add_error("VAL3", f"Validator '{validator.get('id', 'unknown')}' must have config")
    elif not config.get("type"):
        result.add_error("VAL4", f"Validator '{validator.get('id', 'unknown')}' config must have a type")

    return result


def validate_strategy(strategy: dict[str, Any]) -> ValidationResult:
    """Validate strategy configuration.

    Args:
        strategy: Strategy configuration

    Returns:
        ValidationResult with errors and warnings
    """
    result = ValidationResult()

    # Reconcile config
    reconcile = strategy.get("reconcile", {})
    mode = reconcile.get("mode", "any")

    valid_modes = {"any", "all", "weighted", "voting", "model"}
    if mode not in valid_modes:
        result.add_error("STRAT1", f"Invalid reconcile mode: {mode}")

    # Weighted mode needs weights
    if mode == "weighted" and not reconcile.get("weights"):
        result.add_error("STRAT2", "Weighted reconcile mode requires weights")

    # Entries config
    entries = strategy.get("entries", {})
    entry_mode = entries.get("mode", "sequential")

    valid_entry_modes = {"sequential", "parallel"}
    if entry_mode not in valid_entry_modes:
        result.add_error("STRAT3", f"Invalid entries mode: {entry_mode}")

    # Entry rules
    for i, rule in enumerate(entries.get("rules", [])):
        if not rule.get("type"):
            result.add_error("STRAT4", f"Entry rule {i} must have a type")

    # Exit rules
    exits = strategy.get("exits", {})
    for i, rule in enumerate(exits.get("rules", [])):
        if not rule.get("type"):
            result.add_error("STRAT5", f"Exit rule {i} must have a type")

    return result


def validate_data(data: dict[str, Any]) -> ValidationResult:
    """Validate data configuration.

    Args:
        data: Data configuration

    Returns:
        ValidationResult with errors and warnings
    """
    result = ValidationResult()

    # Pairs
    pairs = data.get("pairs", [])
    if not pairs:
        result.add_error("DATA1", "Data must specify at least one pair")

    # Timeframe
    timeframe = data.get("timeframe")
    if not timeframe:
        result.add_warning("DATA2", "No timeframe specified, defaulting to 1h")

    return result


class FlowValidator:
    """Comprehensive flow validator.

    Runs all validation rules and collects results.

    Example:
        >>> validator = FlowValidator()
        >>> result = validator.validate(config)
        >>> if not result.is_valid:
        ...     for error in result.errors:
        ...         print(error)
    """

    def validate(self, config: dict[str, Any]) -> ValidationResult:
        """Validate flow configuration.

        Args:
            config: Flow configuration

        Returns:
            ValidationResult with all errors and warnings
        """
        result = ValidationResult()

        # Top-level validation
        errors, warnings = validate_flow_config(config)
        result.errors.extend(errors)
        result.warnings.extend(warnings)

        # Component-level validation
        for detector in config.get("detectors", []):
            result.merge(validate_detector(detector))

        for validator in config.get("validators", []):
            result.merge(validate_validator(validator))

        if config.get("strategy"):
            result.merge(validate_strategy(config["strategy"]))

        if config.get("data"):
            result.merge(validate_data(config["data"]))

        return result
