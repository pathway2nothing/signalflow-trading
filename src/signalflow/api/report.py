"""Structured report generation from backtest/flow results.

Generates a :class:`Report` with typed sections that can be exported
to HTML (standalone, print-to-PDF) or a JSON-serializable dict for UI rendering.

Usage::

    result = sf.backtest(...)
    report = result.report()
    report.to_html("report.html")
    print(report.text_summary())
"""

from __future__ import annotations

import html as html_mod
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


# ── Section types ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReportSection:
    """Single section of a report.

    Attributes:
        title: Human-readable section heading.
        content_type: One of ``"summary"``, ``"metrics_table"``, ``"equity_svg"``,
            ``"drawdown_svg"``, ``"monthly_table"``, ``"trade_dist"``,
            ``"validation"``, ``"config"``.
        data: Section payload (dict, list, or string depending on type).
    """

    title: str
    content_type: str
    data: Any


# ── Report ─────────────────────────────────────────────────────────────


@dataclass
class Report:
    """Structured backtest/flow report.

    Built via :meth:`BacktestResult.report` or :func:`build_report`.

    Attributes:
        strategy_id: Strategy identifier.
        generated_at: Timestamp when report was generated.
        sections: Ordered list of report sections.
    """

    strategy_id: str
    generated_at: str
    sections: list[ReportSection] = field(default_factory=list)

    # ── Export ──────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Export report as JSON-serializable dict for UI rendering."""
        return {
            "strategy_id": self.strategy_id,
            "generated_at": self.generated_at,
            "sections": [{"title": s.title, "content_type": s.content_type, "data": s.data} for s in self.sections],
        }

    def text_summary(self) -> str:
        """Return plain-text summary of the report."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append(f"  Report: {self.strategy_id}")
        lines.append(f"  Generated: {self.generated_at}")
        lines.append("=" * 60)

        for section in self.sections:
            lines.append("")
            lines.append(f"  {section.title}")
            lines.append(f"  {'-' * len(section.title)}")

            if section.content_type == "metrics_table" and isinstance(section.data, dict):
                for k, v in section.data.items():
                    if isinstance(v, float):
                        lines.append(f"    {k:<28} {v:>12.4f}")
                    else:
                        lines.append(f"    {k:<28} {v!s:>12}")
            elif section.content_type in ("summary", "config") and isinstance(section.data, dict):
                for k, v in section.data.items():
                    lines.append(f"    {k}: {v}")
            elif section.content_type == "monthly_table" and isinstance(section.data, list):
                for row in section.data:
                    month = row.get("month", "")
                    ret = row.get("return", 0.0)
                    lines.append(f"    {month:<12} {ret:>+8.2%}")
            elif section.content_type == "trade_dist" and isinstance(section.data, dict):
                for k, v in section.data.items():
                    lines.append(f"    {k}: {v}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_html(self, path: str | Path | None = None) -> str:
        """Generate standalone HTML report.

        Args:
            path: Optional file path to write the HTML file.

        Returns:
            HTML string (always returned, even when written to file).
        """
        parts: list[str] = [_HTML_HEAD.replace("{{TITLE}}", html_mod.escape(self.strategy_id))]
        parts.append(f"<h1>{html_mod.escape(self.strategy_id)}</h1>")
        parts.append(f'<p class="meta">Generated: {html_mod.escape(self.generated_at)}</p>')

        for section in self.sections:
            parts.append(f"<h2>{html_mod.escape(section.title)}</h2>")
            parts.append(_render_section_html(section))

        parts.append("</div></body></html>")
        html_str = "\n".join(parts)

        if path is not None:
            Path(path).write_text(html_str, encoding="utf-8")

        return html_str


# ── Builder ────────────────────────────────────────────────────────────


def build_report(result: Any) -> Report:
    """Build a :class:`Report` from a BacktestResult or FlowResult.

    Args:
        result: Object with ``.metrics``, ``.trades``, ``.config``, etc.

    Returns:
        Fully populated Report instance.
    """
    strategy_id = _extract_name(result)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections: list[ReportSection] = []

    # 1. Summary
    sections.append(_build_summary_section(result))

    # 2. Metrics table
    metrics = _get_metrics(result)
    if metrics:
        sections.append(ReportSection(title="Performance Metrics", content_type="metrics_table", data=metrics))

    # 3. Equity curve SVG
    equity = _get_equity_data(result)
    if equity:
        sections.append(ReportSection(title="Equity Curve", content_type="equity_svg", data=equity))

    # 4. Drawdown SVG
    drawdown = _get_drawdown_data(result)
    if drawdown:
        sections.append(ReportSection(title="Drawdown", content_type="drawdown_svg", data=drawdown))

    # 5. Monthly returns
    monthly = _get_monthly_returns(result)
    if monthly:
        sections.append(ReportSection(title="Monthly Returns", content_type="monthly_table", data=monthly))

    # 6. Trade distribution
    trade_dist = _get_trade_distribution(result)
    if trade_dist:
        sections.append(ReportSection(title="Trade Distribution", content_type="trade_dist", data=trade_dist))

    # 7. Config snapshot
    config = _get_config(result)
    if config:
        sections.append(ReportSection(title="Configuration", content_type="config", data=config))

    return Report(strategy_id=strategy_id, generated_at=now, sections=sections)


# ── Private helpers ────────────────────────────────────────────────────


def _extract_name(result: Any) -> str:
    if hasattr(result, "strategy_id") and result.strategy_id:
        return str(result.strategy_id)
    if hasattr(result, "config") and isinstance(result.config, dict):
        return str(result.config.get("strategy_id", "Backtest Report"))
    return "Backtest Report"


def _get_metrics(result: Any) -> dict[str, float]:
    if hasattr(result, "metrics"):
        m = result.metrics
        if isinstance(m, dict):
            return {k: v for k, v in m.items() if isinstance(v, (int, float)) and not math.isnan(v)}
    return {}


def _build_summary_section(result: Any) -> ReportSection:
    data: dict[str, Any] = {}
    if hasattr(result, "n_trades"):
        data["trades"] = result.n_trades
    if hasattr(result, "total_return"):
        data["total_return"] = f"{result.total_return:+.2%}"
    if hasattr(result, "win_rate"):
        data["win_rate"] = f"{result.win_rate:.1%}"
    if hasattr(result, "initial_capital"):
        data["initial_capital"] = f"${result.initial_capital:,.2f}"
    if hasattr(result, "final_capital"):
        data["final_capital"] = f"${result.final_capital:,.2f}"
    metrics = _get_metrics(result)
    if "sharpe_ratio" in metrics:
        data["sharpe_ratio"] = f"{metrics['sharpe_ratio']:.2f}"
    if "max_drawdown" in metrics:
        data["max_drawdown"] = f"{metrics['max_drawdown']:.2%}"
    return ReportSection(title="Summary", content_type="summary", data=data)


def _get_equity_data(result: Any) -> list[dict[str, float]] | None:
    """Extract equity curve data points."""
    if not hasattr(result, "metrics_df") or result.metrics_df is None:
        return None
    df = result.metrics_df
    if df.height == 0 or "equity" not in df.columns:
        return None
    rows: list[dict[str, float]] = []
    for i in range(df.height):
        rows.append({"index": float(i), "equity": float(df["equity"][i])})
    return rows


def _get_drawdown_data(result: Any) -> list[dict[str, float]] | None:
    """Extract drawdown series data points."""
    if not hasattr(result, "metrics_df") or result.metrics_df is None:
        return None
    df = result.metrics_df
    if df.height == 0 or "current_drawdown" not in df.columns:
        return None
    rows: list[dict[str, float]] = []
    for i in range(df.height):
        rows.append({"index": float(i), "drawdown": float(df["current_drawdown"][i])})
    return rows


def _get_monthly_returns(result: Any) -> list[dict[str, Any]] | None:
    """Extract monthly return breakdown (stub if no time data)."""
    if not hasattr(result, "metrics_df") or result.metrics_df is None:
        return None
    df = result.metrics_df
    if df.height == 0 or "equity" not in df.columns:
        return None
    if "timestamp" not in df.columns and "ts" not in df.columns:
        return None
    return None  # Requires timestamp column; implemented when data is available


def _get_trade_distribution(result: Any) -> dict[str, Any] | None:
    """Summarize trade PnL distribution."""
    if not hasattr(result, "trades") or not result.trades:
        return None
    pnls: list[float] = []
    for t in result.trades:
        pnl = _trade_pnl(t)
        if pnl is not None:
            pnls.append(pnl)
    if not pnls:
        return None

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    return {
        "total_trades": len(pnls),
        "winners": len(wins),
        "losers": len(losses),
        "avg_win": sum(wins) / len(wins) if wins else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        "largest_win": max(wins) if wins else 0.0,
        "largest_loss": min(losses) if losses else 0.0,
        "avg_pnl": sum(pnls) / len(pnls),
    }


def _trade_pnl(trade: Any) -> float | None:
    if hasattr(trade, "pnl") and trade.pnl is not None:
        return float(trade.pnl)
    if hasattr(trade, "realized_pnl") and trade.realized_pnl is not None:
        return float(trade.realized_pnl)
    if isinstance(trade, dict):
        v = trade.get("pnl") or trade.get("realized_pnl")
        if v is not None:
            return float(v)
    return None


def _get_config(result: Any) -> dict[str, Any] | None:
    if hasattr(result, "config") and isinstance(result.config, dict) and result.config:
        return result.config
    return None


# ── HTML rendering ─────────────────────────────────────────────────────


_HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<style>
  :root { --bg: #0f172a; --card: #1e293b; --border: #334155; --text: #e2e8f0;
          --muted: #94a3b8; --accent: #818cf8; --green: #22c55e; --red: #ef4444; }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: var(--bg); color: var(--text); line-height: 1.6; }
  .container { max-width: 900px; margin: 40px auto; padding: 0 24px; }
  h1 { font-size: 24px; margin-bottom: 4px; }
  h2 { font-size: 16px; color: var(--accent); margin: 32px 0 12px; border-bottom: 1px solid var(--border);
       padding-bottom: 6px; }
  .meta { color: var(--muted); font-size: 13px; margin-bottom: 24px; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
  td.num { text-align: right; font-family: monospace; }
  .card { background: var(--card); border-radius: 8px; padding: 20px; margin-bottom: 16px; }
  .summary-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; }
  .summary-item label { display: block; color: var(--muted); font-size: 12px; text-transform: uppercase; }
  .summary-item .value { font-size: 20px; font-weight: 600; }
  svg { display: block; margin: 8px 0; }
  .positive { color: var(--green); }
  .negative { color: var(--red); }
</style>
</head>
<body><div class="container">"""


def _render_section_html(section: ReportSection) -> str:
    if section.content_type == "summary":
        return _html_summary(section.data)
    if section.content_type == "metrics_table":
        return _html_metrics_table(section.data)
    if section.content_type == "equity_svg":
        return _html_svg_line(section.data, "equity", "#818cf8", "Equity ($)")
    if section.content_type == "drawdown_svg":
        return _html_svg_line(section.data, "drawdown", "#ef4444", "Drawdown")
    if section.content_type == "monthly_table":
        return _html_monthly(section.data)
    if section.content_type == "trade_dist":
        return _html_trade_dist(section.data)
    if section.content_type == "config":
        return _html_config(section.data)
    return f"<pre>{html_mod.escape(str(section.data))}</pre>"


def _html_summary(data: dict[str, Any]) -> str:
    items = []
    for k, v in data.items():
        label = k.replace("_", " ").title()
        css = ""
        sv = str(v)
        if "+" in sv:
            css = ' class="positive"'
        elif sv.startswith("-"):
            css = ' class="negative"'
        items.append(
            f'<div class="summary-item"><label>{html_mod.escape(label)}</label>'
            f'<div class="value"{css}>{html_mod.escape(sv)}</div></div>'
        )
    return f'<div class="card"><div class="summary-grid">{"".join(items)}</div></div>'


def _html_metrics_table(data: dict[str, float]) -> str:
    rows = []
    for k, v in data.items():
        name = html_mod.escape(k)
        formatted = f"{v:.4f}" if isinstance(v, float) else str(v)
        rows.append(f"<tr><td>{name}</td><td class='num'>{formatted}</td></tr>")
    return (
        f'<div class="card"><table><thead><tr><th>Metric</th><th>Value</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _html_svg_line(data: list[dict[str, float]], key: str, color: str, label: str) -> str:
    if not data:
        return "<p>No data available.</p>"
    width, height = 850, 200
    values = [d[key] for d in data]
    vmin, vmax = min(values), max(values)
    vrange = vmax - vmin if vmax != vmin else 1.0
    n = len(values)

    points = []
    for i, v in enumerate(values):
        x = (i / max(n - 1, 1)) * width
        y = height - ((v - vmin) / vrange) * (height - 20) - 10
        points.append(f"{x:.1f},{y:.1f}")

    polyline = " ".join(points)
    return (
        f'<div class="card"><svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="1.5"/>'
        f'<text x="5" y="14" fill="#94a3b8" font-size="11">{html_mod.escape(label)}</text>'
        f'<text x="5" y="{height - 4}" fill="#94a3b8" font-size="10">{vmin:.2f}</text>'
        f'<text x="{width - 60}" y="14" fill="#94a3b8" font-size="10">{vmax:.2f}</text>'
        f"</svg></div>"
    )


def _html_monthly(data: list[dict[str, Any]]) -> str:
    rows = []
    for row in data:
        m = html_mod.escape(str(row.get("month", "")))
        ret = row.get("return", 0.0)
        css = "positive" if ret >= 0 else "negative"
        rows.append(f"<tr><td>{m}</td><td class='num {css}'>{ret:+.2%}</td></tr>")
    return (
        f'<div class="card"><table><thead><tr><th>Month</th><th>Return</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _html_trade_dist(data: dict[str, Any]) -> str:
    rows = []
    for k, v in data.items():
        label = html_mod.escape(k.replace("_", " ").title())
        formatted = f"{v:,.2f}" if isinstance(v, float) else str(v)
        rows.append(f"<tr><td>{label}</td><td class='num'>{formatted}</td></tr>")
    return (
        f'<div class="card"><table><thead><tr><th>Stat</th><th>Value</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _html_config(data: dict[str, Any]) -> str:
    rows = []
    for k, v in data.items():
        rows.append(f"<tr><td>{html_mod.escape(str(k))}</td><td>{html_mod.escape(str(v))}</td></tr>")
    return (
        f'<div class="card"><table><thead><tr><th>Key</th><th>Value</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )
