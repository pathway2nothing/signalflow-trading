"""
SignalFlow Pipeline Visualization.

Provides interactive DAG visualizations of SignalFlow pipelines,
similar to Kedro-Viz.

Example:
    >>> import signalflow as sf

    >>> # Visualize a backtest pipeline
    >>> builder = sf.Backtest("test").data(...).detector(...)
    >>> sf.viz.pipeline(builder)  # Opens interactive HTML

    >>> # Visualize feature dependencies
    >>> from signalflow.feature import FeaturePipeline
    >>> pipe = FeaturePipeline([RsiFeature(14), SmaFeature(20)])
    >>> sf.viz.features(pipe)

    >>> # Export to Mermaid
    >>> sf.viz.pipeline(builder, format="mermaid")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Union

from signalflow.viz.graph import PipelineGraph
from signalflow.viz.renderers import HtmlRenderer, MermaidRenderer

if TYPE_CHECKING:
    from signalflow.api.builder import BacktestBuilder
    from signalflow.core import RawData
    from signalflow.detector.base import SignalDetector
    from signalflow.feature import FeaturePipeline


def pipeline(
    source: Union["BacktestBuilder", "FeaturePipeline", "SignalDetector"],
    *,
    output: str | Path | None = None,
    format: Literal["html", "mermaid", "json"] = "html",
    show: bool = True,
) -> str | PipelineGraph:
    """
    Visualize a SignalFlow pipeline.

    Args:
        source: Pipeline source (BacktestBuilder, FeaturePipeline, or Detector)
        output: Output file path (optional)
        format: Output format ("html", "mermaid", "json")
        show: Open in browser (HTML only)

    Returns:
        Rendered output string or PipelineGraph (if format="json")

    Example:
        >>> import signalflow as sf
        >>> builder = sf.Backtest("test").data(...).detector(...)
        >>> sf.viz.pipeline(builder)  # Opens interactive HTML
        >>> sf.viz.pipeline(builder, format="mermaid")  # Returns Mermaid code
    """
    from signalflow.api.builder import BacktestBuilder
    from signalflow.detector.base import SignalDetector
    from signalflow.feature import FeaturePipeline
    from signalflow.viz.extractors import BacktestExtractor, FeaturePipelineExtractor

    # Extract graph
    if isinstance(source, BacktestBuilder):
        graph = BacktestExtractor(source).extract()
    elif isinstance(source, FeaturePipeline):
        graph = FeaturePipelineExtractor(source).extract()
    elif isinstance(source, SignalDetector):
        # Extract from detector's feature pipeline if it has one
        if hasattr(source, "features") and isinstance(source.features, FeaturePipeline):
            graph = FeaturePipelineExtractor(source.features).extract()
        else:
            # Minimal graph with just detector
            from signalflow.viz.graph import DetectorNode, NodeType

            graph = PipelineGraph()
            graph.add_node(
                DetectorNode(
                    id="detector",
                    name=source.__class__.__name__,
                    node_type=NodeType.DETECTOR,
                    detector_class=source.__class__.__name__,
                )
            )
    else:
        msg = f"Unsupported source type: {type(source).__name__}"
        raise TypeError(msg)

    # Render
    return _render(graph, output=output, format=format, show=show)


def features(
    pipeline: "FeaturePipeline",
    *,
    output: str | Path | None = None,
    format: Literal["html", "mermaid"] = "html",
    show: bool = True,
) -> str:
    """
    Visualize feature dependencies.

    Args:
        pipeline: FeaturePipeline to visualize
        output: Output file path (optional)
        format: Output format ("html" or "mermaid")
        show: Open in browser (HTML only)

    Returns:
        Rendered output string

    Example:
        >>> from signalflow.feature import FeaturePipeline
        >>> from signalflow.feature.examples import ExampleRsiFeature
        >>> pipe = FeaturePipeline([ExampleRsiFeature(14)])
        >>> sf.viz.features(pipe)
    """
    from signalflow.viz.extractors import FeaturePipelineExtractor

    graph = FeaturePipelineExtractor(pipeline).extract()
    result = _render(graph, output=output, format=format, show=show)
    if isinstance(result, PipelineGraph):
        return result.to_mermaid()
    return result


def data_flow(
    raw: "RawData",
    *,
    output: str | Path | None = None,
    format: Literal["html", "mermaid"] = "html",
    show: bool = True,
) -> str:
    """
    Visualize multi-source data flow.

    Args:
        raw: RawData container to visualize
        output: Output file path (optional)
        format: Output format ("html" or "mermaid")
        show: Open in browser (HTML only)

    Returns:
        Rendered output string

    Example:
        >>> raw = RawDataFactory.from_stores(
        ...     stores={"binance": s1, "okx": s2},
        ...     pairs=["BTCUSDT"], start=..., end=...
        ... )
        >>> sf.viz.data_flow(raw)
    """
    from signalflow.viz.extractors import MultiSourceExtractor

    graph = MultiSourceExtractor(raw).extract()
    result = _render(graph, output=output, format=format, show=show)
    if isinstance(result, PipelineGraph):
        return result.to_mermaid()
    return result


def _render(
    graph: PipelineGraph,
    *,
    output: str | Path | None = None,
    format: Literal["html", "mermaid", "json"] = "html",
    show: bool = True,
) -> str | PipelineGraph:
    """Internal render function."""
    if format == "json":
        return graph

    if format == "mermaid":
        result = MermaidRenderer(graph).render()
    else:  # html
        renderer = HtmlRenderer(graph)
        result = renderer.render(output_path=output)
        if show and output is None:
            renderer.show()

    if output and format != "html":
        Path(output).write_text(result, encoding="utf-8")

    return result


def serve(
    source,
    *,
    port: int = 4141,
    open_browser: bool = True,
):
    """
    Start local visualization server (like Kedro-Viz).

    Starts an HTTP server at localhost:{port} serving the pipeline
    visualization. Press Ctrl+C to stop.

    Args:
        source: Pipeline source (BacktestBuilder, FeaturePipeline, etc.)
        port: Server port (default: 4141)
        open_browser: Open browser automatically

    Example:
        >>> import signalflow as sf
        >>> builder = sf.Backtest("test").data(...).detector(...)
        >>> sf.viz.serve(builder)  # Opens http://localhost:4141
    """
    from signalflow.viz.server import serve as _serve

    graph = pipeline(source, format="json", show=False)
    _serve(graph, port=port, open_browser=open_browser, block=True)


__all__ = [
    "pipeline",
    "features",
    "data_flow",
    "serve",
    "PipelineGraph",
]
