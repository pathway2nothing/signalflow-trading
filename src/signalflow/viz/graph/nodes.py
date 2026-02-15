"""Node types for pipeline visualization graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(Enum):
    """Types of nodes in a pipeline graph."""

    DATA_SOURCE = "data_source"
    FEATURE = "feature"
    DETECTOR = "detector"
    LABELER = "labeler"
    VALIDATOR = "validator"
    RUNNER = "runner"
    ENTRY_RULE = "entry_rule"
    EXIT_RULE = "exit_rule"


@dataclass(frozen=True)
class Node:
    """Base node in pipeline graph."""

    id: str
    name: str
    node_type: NodeType
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.node_type.value,
            "metadata": self.metadata,
        }

    def to_cytoscape(self) -> dict[str, Any]:
        """Convert to Cytoscape.js node format."""
        return {
            "data": {
                "id": self.id,
                "label": self.name,
                "type": self.node_type.value,
                **self.metadata,
            },
            "classes": self.node_type.value,
        }


@dataclass(frozen=True)
class DataSourceNode(Node):
    """Data source node (RawData, exchange)."""

    exchange: str | None = None
    data_type: str = "spot"
    pairs: tuple[str, ...] = field(default_factory=tuple)
    columns: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # Ensure node_type is DATA_SOURCE
        if self.node_type != NodeType.DATA_SOURCE:
            object.__setattr__(self, "node_type", NodeType.DATA_SOURCE)

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "exchange": self.exchange,
                "data_type": self.data_type,
                "pairs": list(self.pairs),
                "columns": sorted(self.columns),
            }
        )
        return base

    def to_cytoscape(self) -> dict[str, Any]:
        result = super().to_cytoscape()
        result["data"].update(
            {
                "exchange": self.exchange,
                "data_type": self.data_type,
                "pairs": list(self.pairs),
                "column_count": len(self.columns),
            }
        )
        return result


@dataclass(frozen=True)
class FeatureNode(Node):
    """Feature computation node."""

    feature_class: str = ""
    requires: tuple[str, ...] = field(default_factory=tuple)
    outputs: tuple[str, ...] = field(default_factory=tuple)
    params: dict[str, Any] = field(default_factory=dict)
    is_global: bool = False

    def __post_init__(self) -> None:
        if self.node_type != NodeType.FEATURE:
            object.__setattr__(self, "node_type", NodeType.FEATURE)

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "feature_class": self.feature_class,
                "requires": list(self.requires),
                "outputs": list(self.outputs),
                "params": self.params,
                "is_global": self.is_global,
            }
        )
        return base

    def to_cytoscape(self) -> dict[str, Any]:
        result = super().to_cytoscape()
        result["data"].update(
            {
                "feature_class": self.feature_class,
                "requires": list(self.requires),
                "outputs": list(self.outputs),
                "is_global": self.is_global,
            }
        )
        if self.is_global:
            result["classes"] = f"{result['classes']} global"
        return result


@dataclass(frozen=True)
class DetectorNode(Node):
    """Signal detector node."""

    detector_class: str = ""
    signal_category: str = "PRICE_DIRECTION"
    raw_data_type: str = "spot"

    def __post_init__(self) -> None:
        if self.node_type != NodeType.DETECTOR:
            object.__setattr__(self, "node_type", NodeType.DETECTOR)

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "detector_class": self.detector_class,
                "signal_category": self.signal_category,
                "raw_data_type": self.raw_data_type,
            }
        )
        return base


@dataclass(frozen=True)
class LabelerNode(Node):
    """Target labeler node."""

    labeler_class: str = ""
    out_col: str = "label"

    def __post_init__(self) -> None:
        if self.node_type != NodeType.LABELER:
            object.__setattr__(self, "node_type", NodeType.LABELER)

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "labeler_class": self.labeler_class,
                "out_col": self.out_col,
            }
        )
        return base


@dataclass(frozen=True)
class ValidatorNode(Node):
    """Signal validator node."""

    validator_class: str = ""
    model_type: str | None = None

    def __post_init__(self) -> None:
        if self.node_type != NodeType.VALIDATOR:
            object.__setattr__(self, "node_type", NodeType.VALIDATOR)

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "validator_class": self.validator_class,
                "model_type": self.model_type,
            }
        )
        return base


@dataclass(frozen=True)
class RunnerNode(Node):
    """Strategy runner node."""

    runner_class: str = ""
    entry_rules: tuple[str, ...] = field(default_factory=tuple)
    exit_rules: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.node_type != NodeType.RUNNER:
            object.__setattr__(self, "node_type", NodeType.RUNNER)

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "runner_class": self.runner_class,
                "entry_rules": list(self.entry_rules),
                "exit_rules": list(self.exit_rules),
            }
        )
        return base


@dataclass(frozen=True)
class RuleNode(Node):
    """Entry or exit rule node."""

    rule_class: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "rule_class": self.rule_class,
                "params": self.params,
            }
        )
        return base
