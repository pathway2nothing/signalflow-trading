"""Component help and introspection API.

Provides user-friendly access to component documentation, parameter schemas,
and a built-in glossary of trading/ML terminology.

Usage::

    import signalflow as sf

    sf.help("sharpe")           # → prints metric explanation
    sf.help("sma_cross")        # → detector docs + params
    sf.help.metrics()           # → table of all strategy metrics
    sf.help.detectors()         # → table of all detectors
    sf.help.search("momentum")  # → fuzzy search across components + glossary
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from signalflow.core import SfComponentType, default_registry
from signalflow.help_glossary import GLOSSARY

if TYPE_CHECKING:
    pass

# ── User-friendly type aliases ─────────────────────────────────────────

_TYPE_ALIASES: dict[str, SfComponentType] = {
    "detector": SfComponentType.DETECTOR,
    "detectors": SfComponentType.DETECTOR,
    "feature": SfComponentType.FEATURE,
    "features": SfComponentType.FEATURE,
    "labeler": SfComponentType.LABELER,
    "labelers": SfComponentType.LABELER,
    "label": SfComponentType.LABELER,
    "validator": SfComponentType.VALIDATOR,
    "validators": SfComponentType.VALIDATOR,
    "entry": SfComponentType.STRATEGY_ENTRY_RULE,
    "entry_rule": SfComponentType.STRATEGY_ENTRY_RULE,
    "exit": SfComponentType.STRATEGY_EXIT_RULE,
    "exit_rule": SfComponentType.STRATEGY_EXIT_RULE,
    "metric": SfComponentType.STRATEGY_METRIC,
    "metrics": SfComponentType.STRATEGY_METRIC,
    "data_source": SfComponentType.RAW_DATA_SOURCE,
    "source": SfComponentType.RAW_DATA_SOURCE,
    "data_store": SfComponentType.RAW_DATA_STORE,
    "store": SfComponentType.RAW_DATA_STORE,
    "executor": SfComponentType.STRATEGY_EXECUTOR,
    "runner": SfComponentType.STRATEGY_EXECUTOR,
    "risk": SfComponentType.STRATEGY_RISK,
    "alert": SfComponentType.STRATEGY_ALERT,
    "signal_feature": SfComponentType.SIGNAL_FEATURE,
    "signal_metric": SfComponentType.SIGNAL_METRIC,
}


def _resolve_type(name: str) -> SfComponentType | None:
    """Resolve user-friendly type name to SfComponentType."""
    low = name.lower().strip()
    if low in _TYPE_ALIASES:
        return _TYPE_ALIASES[low]
    for ct in SfComponentType:
        if ct.value == low or ct.name.lower() == low:
            return ct
    return None


# ── Formatting helpers ─────────────────────────────────────────────────


def _format_glossary_entry(term: str, entry: dict[str, str]) -> str:
    """Format a glossary entry for terminal display."""
    lines: list[str] = []
    lines.append(f"  {term}")
    lines.append(f"  {'=' * len(term)}")
    if "definition" in entry:
        lines.append(f"  {entry['definition']}")
    if "formula" in entry:
        lines.append(f"\n  Formula: {entry['formula']}")
    if "interpretation" in entry:
        lines.append(f"\n  Interpretation: {entry['interpretation']}")
    if "good_range" in entry:
        lines.append(f"  Good range: {entry['good_range']}")
    if "category" in entry:
        lines.append(f"  Category: {entry['category']}")
    return "\n".join(lines)


def _format_component(name: str, component_type: SfComponentType) -> str:
    """Format component info for terminal display."""
    lines: list[str] = []

    try:
        info = default_registry.get_info(component_type, name)
    except KeyError:
        return f"  Component '{name}' not found in {component_type.value}"

    lines.append(f"  {name}  ({component_type.value})")
    lines.append(f"  {'=' * (len(name) + len(component_type.value) + 4)}")

    if info.summary:
        lines.append(f"  {info.summary}")

    if info.module:
        lines.append(f"\n  Module: {info.module}")

    # Parameters
    try:
        schema = default_registry.get_schema(component_type, name)
        params = schema.get("parameters", [])
        if params:
            lines.append("\n  Parameters:")
            for p in params:
                req = " (required)" if p.get("required") else ""
                default = f" = {p['default']}" if "default" in p and not p.get("required") else ""
                lines.append(f"    {p['name']}: {p.get('type', 'Any')}{default}{req}")
    except Exception:
        pass

    # Full docstring (if longer than summary)
    if info.docstring and info.docstring != info.summary:
        lines.append(f"\n  {info.docstring}")

    return "\n".join(lines)


def _format_type_table(component_type: SfComponentType) -> str:
    """Format a table listing all components of a type."""
    names = default_registry.list(component_type)
    if not names:
        return f"  No {component_type.value} components registered."

    lines: list[str] = []
    type_label = component_type.value
    lines.append(f"  {type_label} ({len(names)} registered)")
    lines.append(f"  {'-' * 60}")

    for name in names:
        try:
            info = default_registry.get_info(component_type, name)
            summary = info.summary[:55] + "..." if len(info.summary) > 58 else info.summary
            lines.append(f"  {name:<30} {summary}")
        except KeyError:
            lines.append(f"  {name:<30}")

    return "\n".join(lines)


# ── Search ─────────────────────────────────────────────────────────────


def _search(query: str) -> str:
    """Search across all components and glossary terms."""
    q = query.lower().strip()
    results: list[str] = []

    # Search glossary
    for term, entry in GLOSSARY.items():
        if q in term.lower() or q in entry.get("definition", "").lower():
            results.append(f"  [glossary] {term}: {entry.get('definition', '')[:60]}...")

    # Search components
    for ct in SfComponentType:
        for name in default_registry.list(ct):
            if q in name.lower():
                try:
                    info = default_registry.get_info(ct, name)
                    results.append(f"  [{ct.value}] {name}: {info.summary[:60]}")
                except KeyError:
                    results.append(f"  [{ct.value}] {name}")
                continue
            # Search in docstring
            try:
                info = default_registry.get_info(ct, name)
                if q in info.docstring.lower():
                    results.append(f"  [{ct.value}] {name}: {info.summary[:60]}")
            except KeyError:
                pass

    if not results:
        return f'  No results for "{query}".'
    return f'  Search results for "{query}" ({len(results)} found):\n\n' + "\n".join(results)


# ── Main help class ────────────────────────────────────────────────────


class _HelpSystem:
    """Callable help system with method access for category listings.

    Used as ``sf.help("term")`` or ``sf.help.detectors()``.
    """

    def __call__(self, term: str = "") -> None:
        """Look up a term in glossary, components, or type categories.

        Args:
            term: A glossary term (e.g. "sharpe"), component name
                  (e.g. "sma_cross"), or category name (e.g. "detectors").
                  Empty string shows an overview.
        """
        if not term:
            self._print_overview()
            return

        low = term.lower().strip()

        # 1. Check if it's a type alias → list components of that type
        ct = _resolve_type(low)
        if ct is not None:
            print(_format_type_table(ct))
            return

        # 2. Check glossary
        for gterm, entry in GLOSSARY.items():
            if low == gterm.lower():
                print(_format_glossary_entry(gterm, entry))
                return

        # 3. Search registry across all types (exact match or suffix match)
        for comp_type in SfComponentType:
            names = default_registry.list(comp_type)
            # Exact match first
            for n in names:
                if n.lower() == low:
                    print(_format_component(n, comp_type))
                    return
            # Suffix match: "sma_cross" matches "example/sma_cross"
            for n in names:
                if n.lower().endswith("/" + low):
                    print(_format_component(n, comp_type))
                    return

        # 4. Fallback: fuzzy search
        print(_search(term))

    def _print_overview(self) -> None:
        """Print overview of component types and counts."""
        lines: list[str] = []
        lines.append("  SignalFlow Help")
        lines.append("  ===============")
        lines.append("")
        lines.append("  Component Types:")
        for ct in SfComponentType:
            count = len(default_registry.list(ct))
            if count > 0:
                lines.append(f"    {ct.value:<30} {count:>3} registered")
        lines.append("")
        lines.append(f"  Glossary terms: {len(GLOSSARY)}")
        lines.append("")
        lines.append("  Usage:")
        lines.append('    sf.help("sharpe_ratio")      # metric explanation')
        lines.append('    sf.help("sma_cross")          # component docs + params')
        lines.append('    sf.help("detectors")          # list all detectors')
        lines.append('    sf.help.search("momentum")    # search everything')
        lines.append('    sf.help.metrics()             # list all metrics')
        lines.append('    sf.help.detectors()           # list all detectors')
        print("\n".join(lines))

    def search(self, query: str) -> None:
        """Search across all components and glossary terms.

        Args:
            query: Search string to match against names, docstrings, definitions.
        """
        print(_search(query))

    def detectors(self) -> None:
        """List all registered detectors."""
        print(_format_type_table(SfComponentType.DETECTOR))

    def features(self) -> None:
        """List all registered features."""
        print(_format_type_table(SfComponentType.FEATURE))

    def metrics(self) -> None:
        """List all registered strategy metrics."""
        print(_format_type_table(SfComponentType.STRATEGY_METRIC))

    def labelers(self) -> None:
        """List all registered labelers."""
        print(_format_type_table(SfComponentType.LABELER))

    def validators(self) -> None:
        """List all registered validators."""
        print(_format_type_table(SfComponentType.VALIDATOR))

    def entries(self) -> None:
        """List all registered entry rules."""
        print(_format_type_table(SfComponentType.STRATEGY_ENTRY_RULE))

    def exits(self) -> None:
        """List all registered exit rules."""
        print(_format_type_table(SfComponentType.STRATEGY_EXIT_RULE))

    def data_sources(self) -> None:
        """List all registered data sources."""
        print(_format_type_table(SfComponentType.RAW_DATA_SOURCE))

    def glossary(self) -> None:
        """Print the full glossary."""
        lines: list[str] = []
        lines.append(f"  SignalFlow Glossary ({len(GLOSSARY)} terms)")
        lines.append(f"  {'-' * 60}")
        for term, entry in sorted(GLOSSARY.items()):
            defn = entry.get("definition", "")
            cat = entry.get("category", "")
            cat_str = f" [{cat}]" if cat else ""
            lines.append(f"  {term:<30}{cat_str} {defn[:50]}")
        print("\n".join(lines))

    def schema(self, component_type: str, name: str) -> dict[str, Any]:
        """Return raw JSON-serializable schema for a component.

        Args:
            component_type: Type alias (e.g. "detector") or enum value.
            name: Component name.

        Returns:
            Schema dict with parameters, docstring, module.

        Raises:
            KeyError: If component_type or name not found.
        """
        ct = _resolve_type(component_type)
        if ct is None:
            msg = f"Unknown component type: {component_type!r}"
            raise KeyError(msg)
        return default_registry.get_schema(ct, name)

    def export_all(self) -> dict[str, Any]:
        """Export all schemas + glossary as JSON-serializable dict.

        Returns:
            Dict with "components" (from registry) and "glossary" keys.
        """
        return {
            "components": default_registry.export_schemas(),
            "glossary": dict(GLOSSARY),
        }


help_system = _HelpSystem()
