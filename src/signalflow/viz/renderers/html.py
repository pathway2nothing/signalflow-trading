"""Interactive HTML renderer using Cytoscape.js."""

from __future__ import annotations

import json
import tempfile
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from signalflow.viz.graph import PipelineGraph


# HTML template with embedded Cytoscape.js
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SignalFlow Pipeline</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
        }
        #header {
            background: #16213e;
            padding: 12px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #0f3460;
        }
        #header h1 {
            font-size: 18px;
            font-weight: 500;
            color: #e94560;
        }
        #header .meta {
            font-size: 13px;
            color: #888;
        }
        #cy {
            width: 100%;
            height: calc(100vh - 50px);
        }
        #tooltip {
            position: absolute;
            display: none;
            background: #16213e;
            border: 1px solid #0f3460;
            border-radius: 6px;
            padding: 12px 16px;
            font-size: 13px;
            max-width: 320px;
            z-index: 1000;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        }
        #tooltip .title {
            font-weight: 600;
            color: #e94560;
            margin-bottom: 8px;
        }
        #tooltip .row {
            display: flex;
            margin: 4px 0;
        }
        #tooltip .label {
            color: #888;
            width: 80px;
            flex-shrink: 0;
        }
        #tooltip .value {
            color: #eee;
            word-break: break-word;
        }
        #tooltip .columns {
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid #0f3460;
        }
        #tooltip .columns .badge {
            display: inline-block;
            background: #0f3460;
            color: #7ec8e3;
            padding: 2px 8px;
            border-radius: 4px;
            margin: 2px;
            font-size: 11px;
        }
    </style>
</head>
<body>
    <div id="header">
        <h1>SignalFlow Pipeline</h1>
        <span class="meta">{metadata}</span>
    </div>
    <div id="cy"></div>
    <div id="tooltip"></div>

    <script>
        const graphData = {graph_json};

        // Color scheme for node types
        const colors = {
            data_source: { bg: '#2d6a4f', border: '#40916c', text: '#ffffff' },
            feature: { bg: '#023e8a', border: '#0077b6', text: '#ffffff' },
            detector: { bg: '#7b2cbf', border: '#9d4edd', text: '#ffffff' },
            labeler: { bg: '#9d4edd', border: '#c77dff', text: '#ffffff' },
            validator: { bg: '#ff6d00', border: '#ff9500', text: '#ffffff' },
            runner: { bg: '#d00000', border: '#e85d04', text: '#ffffff' },
            entry_rule: { bg: '#0a9396', border: '#94d2bd', text: '#ffffff' },
            exit_rule: { bg: '#9b2226', border: '#bb3e03', text: '#ffffff' }
        };

        // Edge colors
        const edgeColors = {
            data_flow: '#6c757d',
            column: '#7ec8e3',
            signal: '#e94560',
            feature: '#0077b6'
        };

        // Build style array
        const style = [
            {
                selector: 'node',
                style: {
                    'label': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'font-size': '12px',
                    'font-weight': 500,
                    'width': 'label',
                    'height': 36,
                    'padding': '12px',
                    'shape': 'roundrectangle',
                    'text-wrap': 'wrap',
                    'text-max-width': '140px'
                }
            },
            {
                selector: 'edge',
                style: {
                    'width': 2,
                    'line-color': '#6c757d',
                    'target-arrow-color': '#6c757d',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier',
                    'arrow-scale': 0.8
                }
            },
            {
                selector: 'edge[label]',
                style: {
                    'label': 'data(label)',
                    'font-size': '10px',
                    'text-rotation': 'autorotate',
                    'text-margin-y': -10,
                    'color': '#888'
                }
            }
        ];

        // Add node type styles
        Object.entries(colors).forEach(([type, c]) => {
            style.push({
                selector: `node.${type}`,
                style: {
                    'background-color': c.bg,
                    'border-color': c.border,
                    'border-width': 2,
                    'color': c.text
                }
            });
        });

        // Add edge type styles
        Object.entries(edgeColors).forEach(([type, color]) => {
            style.push({
                selector: `edge.${type}`,
                style: {
                    'line-color': color,
                    'target-arrow-color': color
                }
            });
        });

        // Global feature style
        style.push({
            selector: 'node.global',
            style: {
                'border-style': 'dashed'
            }
        });

        // Initialize Cytoscape
        const cy = cytoscape({
            container: document.getElementById('cy'),
            elements: graphData.elements,
            style: style,
            layout: {
                name: 'dagre',
                rankDir: 'LR',
                nodeSep: 50,
                rankSep: 80,
                padding: 40
            },
            minZoom: 0.3,
            maxZoom: 2.5,
            wheelSensitivity: 0.3
        });

        // Tooltip handling
        const tooltip = document.getElementById('tooltip');

        cy.on('mouseover', 'node', (evt) => {
            const node = evt.target;
            const data = node.data();

            let html = `<div class="title">${data.label}</div>`;
            html += `<div class="row"><span class="label">Type:</span><span class="value">${data.type}</span></div>`;

            if (data.feature_class) {
                html += `<div class="row"><span class="label">Class:</span><span class="value">${data.feature_class}</span></div>`;
            }
            if (data.exchange) {
                html += `<div class="row"><span class="label">Exchange:</span><span class="value">${data.exchange}</span></div>`;
            }
            if (data.data_type) {
                html += `<div class="row"><span class="label">Data Type:</span><span class="value">${data.data_type}</span></div>`;
            }
            if (data.is_global !== undefined) {
                html += `<div class="row"><span class="label">Global:</span><span class="value">${data.is_global ? 'Yes' : 'No'}</span></div>`;
            }

            // Show requires/outputs for features
            if (data.requires && data.requires.length > 0) {
                html += '<div class="columns"><strong>Requires:</strong><br>';
                data.requires.forEach(col => {
                    html += `<span class="badge">${col}</span>`;
                });
                html += '</div>';
            }
            if (data.outputs && data.outputs.length > 0) {
                html += '<div class="columns"><strong>Outputs:</strong><br>';
                data.outputs.forEach(col => {
                    html += `<span class="badge">${col}</span>`;
                });
                html += '</div>';
            }

            tooltip.innerHTML = html;
            tooltip.style.display = 'block';
        });

        cy.on('mouseout', 'node', () => {
            tooltip.style.display = 'none';
        });

        cy.on('mousemove', (evt) => {
            const pos = evt.renderedPosition || evt.position;
            tooltip.style.left = (pos.x + 15) + 'px';
            tooltip.style.top = (pos.y + 15) + 'px';
        });

        // Fit to screen
        cy.fit(undefined, 40);
    </script>
</body>
</html>"""


class HtmlRenderer:
    """Render PipelineGraph to interactive HTML with Cytoscape.js."""

    def __init__(self, graph: "PipelineGraph"):
        self.graph = graph

    def render(self, output_path: str | Path | None = None) -> str:
        """
        Render to HTML string or file.

        Args:
            output_path: Optional file path to write HTML.

        Returns:
            HTML string.
        """
        graph_json = json.dumps(self.graph.to_cytoscape())

        # Build metadata string
        meta_parts = []
        if "strategy_id" in self.graph.metadata:
            meta_parts.append(f"Strategy: {self.graph.metadata['strategy_id']}")
        if "type" in self.graph.metadata:
            meta_parts.append(f"Type: {self.graph.metadata['type']}")
        meta_parts.append(f"Nodes: {len(self.graph.nodes)}")
        meta_parts.append(f"Edges: {len(self.graph.edges)}")
        metadata = " | ".join(meta_parts)

        html = HTML_TEMPLATE.replace("{graph_json}", graph_json)
        html = html.replace("{metadata}", metadata)

        if output_path:
            Path(output_path).write_text(html, encoding="utf-8")

        return html

    def show(self) -> None:
        """Open visualization in browser."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".html",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(self.render())
            webbrowser.open(f"file://{f.name}")
