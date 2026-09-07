"""Feature configuration as a tree (or forest) of transforms.

A feature belongs to exactly one consumer, so the dependency structure of a
pipeline is a forest: each node names the transform it runs and nests the nodes
that produce its inputs. Nesting *is* the edge - no identifiers to wire up, no
cycles to detect - and the execution order is the post-order walk (producers
before the consumer).

```yaml
pipeline:
  woe_sma:                      # node name, free-form, used in errors and the graph
    registry: woe               # component name in the signalflow registry
    cls: WoE                    # class name; written on save, validated on load
    fit: true                   # stateful (refit inside every fold); validated too
    params: {binning: quantile, max_bins: 15}
    inputs:                     # the nodes that produce what this one reads
      - {registry: sma, params: {length: 10}}
      - {registry: sma, params: {length: 50}}
```

A node without ``inputs`` reads the raw dataset columns. ``cls`` and ``fit`` are
derived from the registry and only need to be written by hand when you want the
file to state them; when present they are checked, so a config can never quietly
disagree with the code it names. A plain list of nodes is also accepted and keeps
the old meaning: every step sees everything produced before it.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from signalflow.enums import ComponentType
from signalflow.errors import PipelineError

_NODE_KEYS = {"registry", "transform", "cls", "fit", "params", "inputs", "name"}


def _registry_name(spec: Mapping, where: str) -> str:
    name = spec.get("registry") or spec.get("transform")
    if not name:
        raise PipelineError(f"{where}: node has no 'registry' (the component name in the signalflow registry)")
    return str(name)


def _build_one(spec: Mapping, where: str, scope: "list[str] | None" = None):
    """Instantiate the transform a node names, checking its declared class and fit flag.

    ``scope`` is what this node's inputs produce: when the transform takes a
    ``columns`` parameter and the config did not set one, it is bound to that scope.
    """
    import dataclasses

    from signalflow.registry import registry

    unknown = set(spec) - _NODE_KEYS
    if unknown:
        raise PipelineError(f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(_NODE_KEYS)}")
    name = _registry_name(spec, where)
    try:
        cls = registry.get(ComponentType.TRANSFORM, name)
    except Exception as exc:
        raise PipelineError(f"{where}: unknown transform {name!r}: {exc}") from exc

    declared_cls = spec.get("cls")
    if declared_cls and declared_cls != cls.__name__:
        raise PipelineError(
            f"{where}: config says cls={declared_cls!r} but {name!r} is {cls.__name__!r}; "
            f"the file disagrees with the code"
        )
    params = dict(spec.get("params") or {})
    takes_columns = dataclasses.is_dataclass(cls) and any(f.name == "columns" for f in dataclasses.fields(cls))
    if scope and "columns" not in params and takes_columns:
        params["columns"] = list(scope)
    try:
        node = cls(**params)
    except TypeError as exc:
        raise PipelineError(f"{where}: {name!r} rejects params {sorted(params)}: {exc}") from exc

    declared_fit = spec.get("fit")
    if declared_fit is not None and bool(declared_fit) != bool(node.requires_fit):
        raise PipelineError(
            f"{where}: config says fit={bool(declared_fit)} but {name!r} "
            f"{'requires' if node.requires_fit else 'does not require'} fitting"
        )
    return node


def _walk(name: str, spec: Any, out: list, path: str) -> None:
    """Post-order: every producer is appended before the node that consumes it.

    The nesting also *scopes* the consumer: a transform that takes a ``columns``
    parameter (WoE, Scaler, IVSelector) is bound to exactly what its inputs
    produce, so one encoder per feature group is the natural thing to write and
    "encode everything at once" stops being the default.
    """
    where = f"{path}{name}" if path else name
    if not isinstance(spec, Mapping):
        raise PipelineError(f"{where}: a node must be a mapping with 'registry', got {type(spec).__name__}")
    inputs = spec.get("inputs") or []
    if isinstance(inputs, Mapping):
        inputs = [{"name": k, **v} if isinstance(v, Mapping) else v for k, v in inputs.items()]
    if isinstance(inputs, str) or not isinstance(inputs, Sequence):
        raise PipelineError(f"{where}: 'inputs' must be a list of nodes, got {type(inputs).__name__}")
    produced: list[str] = []
    for i, child in enumerate(inputs):
        child_name = child.get("name") if isinstance(child, Mapping) else None
        before = len(out)
        _walk(child_name or f"inputs[{i}]", child, out, f"{where}.")
        for _, node in out[before:]:
            produced.extend(c for c in node.outputs if c not in produced)
    out.append((where, _build_one(spec, where, scope=produced)))


def parse_nodes(spec: Any) -> "list[tuple[str, Any]]":
    """``(node name, transform)`` in execution order, from a forest, a list, or a single node."""
    nodes: list[tuple[str, Any]] = []
    if isinstance(spec, Mapping):
        # a forest: name -> node; or a single node written inline
        if spec.keys() & {"registry", "transform"}:
            _walk(spec.get("name") or _registry_name(spec, "pipeline"), spec, nodes, "")
        else:
            for name, node in spec.items():
                _walk(str(name), node, nodes, "")
    elif isinstance(spec, Sequence) and not isinstance(spec, str):
        for i, node in enumerate(spec):
            name = node.get("name") if isinstance(node, Mapping) else None
            _walk(name or f"[{i}]", node, nodes, "")
    else:
        raise PipelineError(f"pipeline config must be a mapping or a list of nodes, got {type(spec).__name__}")
    if not nodes:
        raise PipelineError("pipeline config is empty")
    return nodes


def node_config(node: Any, name: str, inputs: "list[dict] | None" = None) -> dict:
    """One node as it is written to YAML: registry, cls, fit, params, and nested inputs."""
    cfg = node.to_config()
    out: dict[str, Any] = {
        "registry": cfg.get("transform", node.name),
        "cls": type(node).__name__,
        "fit": bool(node.requires_fit),
    }
    params = cfg.get("params") or {}
    if params:
        out["params"] = params
    if inputs:
        out["inputs"] = inputs
    return {name: out}


__all__ = ["node_config", "parse_nodes"]
