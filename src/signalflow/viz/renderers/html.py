"""Interactive HTML renderer using D3.js."""

from __future__ import annotations

import json
import tempfile
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from signalflow.viz.graph import PipelineGraph


# Premium HTML template with D3.js + dagre - SignalFlow documentation style
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SignalFlow Viz</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
    <style>
        :root {
            /* SignalFlow brand colors - from docs */
            --sf-accent: #3b82f6;
            --sf-accent-light: #60a5fa;
            --sf-accent-glow: rgba(59, 130, 246, 0.4);

            /* Dark theme - slate from MkDocs Material */
            --sf-bg: #0f1115;
            --sf-bg-surface: #161920;
            --sf-bg-elevated: #1c2028;
            --sf-bg-hover: #242830;

            --sf-text-primary: #f8fafc;
            --sf-text-secondary: #cbd5e1;
            --sf-text-muted: #64748b;

            --sf-border: rgba(255, 255, 255, 0.08);
            --sf-border-active: rgba(255, 255, 255, 0.16);

            /* Node colors - vibrant but professional */
            --node-data: #2563eb;
            --node-feature: #0891b2;
            --node-detector: #7c3aed;
            --node-labeler: #db2777;
            --node-validator: #ca8a04;
            --node-runner: #dc2626;
            --node-entry: #059669;
            --node-exit: #ea580c;

            /* Gradients */
            --gradient-brand: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            --gradient-surface: linear-gradient(180deg, rgba(255,255,255,0.03) 0%, transparent 100%);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--sf-bg);
            color: var(--sf-text-primary);
            overflow: hidden;
            -webkit-font-smoothing: antialiased;
        }

        .app { display: flex; height: 100vh; }

        /* ============================================
           Sidebar - Premium glassmorphism
        ============================================ */
        .sidebar {
            width: 320px;
            background: var(--sf-bg-surface);
            border-right: 1px solid var(--sf-border);
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
            position: relative;
            z-index: 10;
        }

        .sidebar::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 200px;
            background: var(--gradient-surface);
            pointer-events: none;
        }

        /* Header */
        .sidebar-header {
            padding: 24px;
            border-bottom: 1px solid var(--sf-border);
            position: relative;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 24px;
        }

        .logo-icon {
            width: 44px;
            height: 44px;
            background: var(--gradient-brand);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 15px;
            color: white;
            box-shadow: 0 4px 12px var(--sf-accent-glow);
            letter-spacing: -0.5px;
        }

        .logo-text {
            display: flex;
            flex-direction: column;
        }

        .logo-title {
            font-size: 18px;
            font-weight: 700;
            color: var(--sf-text-primary);
            letter-spacing: -0.3px;
        }

        .logo-subtitle {
            font-size: 12px;
            color: var(--sf-text-muted);
            font-weight: 500;
            margin-top: 2px;
        }

        /* Stats */
        .stats {
            display: flex;
            gap: 32px;
        }

        .stat {
            display: flex;
            flex-direction: column;
        }

        .stat-value {
            font-size: 32px;
            font-weight: 800;
            color: var(--sf-text-primary);
            line-height: 1;
            letter-spacing: -1px;
        }

        .stat-label {
            font-size: 11px;
            color: var(--sf-text-muted);
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 600;
            margin-top: 6px;
        }

        /* Legend */
        .legend {
            padding: 20px 24px;
            border-bottom: 1px solid var(--sf-border);
        }

        .section-title {
            font-size: 10px;
            font-weight: 700;
            color: var(--sf-text-muted);
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 14px;
        }

        .legend-items {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            font-weight: 500;
            color: var(--sf-text-secondary);
            cursor: pointer;
            padding: 6px 12px;
            border-radius: 8px;
            border: 1px solid transparent;
            transition: all 0.2s ease;
            background: transparent;
        }

        .legend-item:hover {
            background: var(--sf-bg-hover);
            border-color: var(--sf-border-active);
            color: var(--sf-text-primary);
        }

        .legend-dot {
            width: 12px;
            height: 12px;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }

        /* Details Panel */
        .details {
            flex: 1;
            padding: 24px;
            overflow-y: auto;
        }

        .details::-webkit-scrollbar {
            width: 6px;
        }

        .details::-webkit-scrollbar-track {
            background: transparent;
        }

        .details::-webkit-scrollbar-thumb {
            background: var(--sf-border-active);
            border-radius: 3px;
        }

        .details-empty {
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: var(--sf-text-muted);
            text-align: center;
        }

        .details-empty-icon {
            width: 64px;
            height: 64px;
            margin-bottom: 16px;
            opacity: 0.3;
            background: var(--gradient-brand);
            -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='currentColor'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M13 10V3L4 14h7v7l9-11h-7z'/%3E%3C/svg%3E") center/contain no-repeat;
            mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='currentColor'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M13 10V3L4 14h7v7l9-11h-7z'/%3E%3C/svg%3E") center/contain no-repeat;
        }

        .details-empty-text {
            font-size: 14px;
            font-weight: 500;
        }

        .details-empty-hint {
            font-size: 12px;
            margin-top: 4px;
            opacity: 0.6;
        }

        .details-content { display: none; }
        .details-content.active { display: block; }

        .details-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            padding: 6px 12px;
            border-radius: 6px;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }

        .details-badge::before {
            content: '';
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: currentColor;
            opacity: 0.6;
        }

        .details-name {
            font-size: 22px;
            font-weight: 700;
            letter-spacing: -0.5px;
            margin-bottom: 24px;
            color: var(--sf-text-primary);
        }

        .details-section {
            margin-bottom: 24px;
        }

        .details-section-title {
            font-size: 10px;
            font-weight: 700;
            color: var(--sf-text-muted);
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .details-section-title::after {
            content: '';
            flex: 1;
            height: 1px;
            background: var(--sf-border);
        }

        .details-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid var(--sf-border);
            font-size: 13px;
        }

        .details-row:last-child {
            border-bottom: none;
        }

        .details-key {
            color: var(--sf-text-muted);
            font-weight: 500;
        }

        .details-value {
            color: var(--sf-text-primary);
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
        }

        .tag-list {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .tag {
            font-size: 11px;
            font-weight: 600;
            padding: 5px 10px;
            background: var(--sf-bg-elevated);
            border: 1px solid var(--sf-border);
            border-radius: 6px;
            color: var(--sf-text-secondary);
            font-family: 'JetBrains Mono', monospace;
            transition: all 0.15s ease;
        }

        .tag.input {
            border-color: var(--node-entry);
            color: var(--node-entry);
            background: rgba(5, 150, 105, 0.1);
        }

        .tag.output {
            border-color: var(--sf-accent);
            color: var(--sf-accent-light);
            background: rgba(59, 130, 246, 0.1);
        }

        /* ============================================
           Graph Container
        ============================================ */
        .graph-container {
            flex: 1;
            position: relative;
            background: var(--sf-bg);
            overflow: hidden;
        }

        /* Subtle grid pattern */
        .graph-container::before {
            content: '';
            position: absolute;
            inset: 0;
            background-image:
                radial-gradient(circle at 1px 1px, rgba(255,255,255,0.03) 1px, transparent 0);
            background-size: 40px 40px;
            pointer-events: none;
        }

        /* Gradient overlay */
        .graph-container::after {
            content: '';
            position: absolute;
            inset: 0;
            background: radial-gradient(ellipse at center, transparent 0%, var(--sf-bg) 100%);
            opacity: 0.4;
            pointer-events: none;
        }

        svg.graph {
            width: 100%;
            height: 100%;
            cursor: grab;
            position: relative;
            z-index: 1;
        }

        svg.graph:active { cursor: grabbing; }

        /* Node styling */
        .node {
            cursor: pointer;
            transition: transform 0.2s ease, filter 0.2s ease;
        }

        .node:hover {
            filter: brightness(1.15);
        }

        .node rect, .node ellipse, .node polygon {
            stroke-width: 2px;
            transition: all 0.2s ease;
        }

        .node.selected rect,
        .node.selected ellipse,
        .node.selected polygon {
            stroke: var(--sf-accent-light) !important;
            stroke-width: 3px;
            filter: drop-shadow(0 0 20px var(--sf-accent-glow));
        }

        .node.faded {
            opacity: 0.15;
        }

        .node text {
            font-family: 'Inter', sans-serif;
            font-size: 12px;
            font-weight: 600;
            fill: white;
            pointer-events: none;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
        }

        /* Edge styling */
        .edge path {
            fill: none;
            stroke: rgba(255, 255, 255, 0.2);
            stroke-width: 2px;
            transition: all 0.2s ease;
        }

        .edge.highlighted path {
            stroke: var(--sf-accent);
            stroke-width: 3px;
            filter: drop-shadow(0 0 6px var(--sf-accent-glow));
        }

        .edge.faded {
            opacity: 0.05;
        }

        .edge polygon {
            fill: rgba(255, 255, 255, 0.3);
            stroke: none;
            transition: fill 0.2s ease;
        }

        .edge.highlighted polygon {
            fill: var(--sf-accent);
        }

        /* Animated edge flow */
        @keyframes flowAnimation {
            0% { stroke-dashoffset: 24; }
            100% { stroke-dashoffset: 0; }
        }

        .edge.highlighted path {
            stroke-dasharray: 8 4;
            animation: flowAnimation 0.8s linear infinite;
        }

        /* ============================================
           Controls
        ============================================ */
        .controls {
            position: absolute;
            bottom: 24px;
            right: 24px;
            display: flex;
            gap: 8px;
            z-index: 10;
        }

        .control-btn {
            width: 44px;
            height: 44px;
            background: var(--sf-bg-surface);
            border: 1px solid var(--sf-border);
            border-radius: 10px;
            color: var(--sf-text-secondary);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            font-weight: 600;
            transition: all 0.2s ease;
            backdrop-filter: blur(8px);
        }

        .control-btn:hover {
            background: var(--sf-bg-hover);
            color: var(--sf-text-primary);
            border-color: var(--sf-accent);
            box-shadow: 0 4px 12px var(--sf-accent-glow);
            transform: translateY(-2px);
        }

        .control-btn:active {
            transform: translateY(0);
        }

        /* ============================================
           Watermark
        ============================================ */
        .watermark {
            position: absolute;
            bottom: 24px;
            left: 24px;
            font-size: 11px;
            color: var(--sf-text-muted);
            opacity: 0.5;
            z-index: 10;
            font-weight: 500;
            letter-spacing: 0.5px;
        }

        .watermark a {
            color: var(--sf-accent);
            text-decoration: none;
            font-weight: 600;
        }

        .watermark a:hover {
            text-decoration: underline;
        }

        /* ============================================
           Tooltips
        ============================================ */
        .tooltip {
            position: absolute;
            background: var(--sf-bg-elevated);
            border: 1px solid var(--sf-border-active);
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 12px;
            font-weight: 500;
            color: var(--sf-text-primary);
            pointer-events: none;
            opacity: 0;
            transform: translateY(4px);
            transition: all 0.15s ease;
            z-index: 100;
            box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        }

        .tooltip.visible {
            opacity: 1;
            transform: translateY(0);
        }
    </style>
</head>
<body>
    <div class="app">
        <aside class="sidebar">
            <div class="sidebar-header">
                <div class="logo">
                    <div class="logo-icon">SF</div>
                    <div class="logo-text">
                        <span class="logo-title">SignalFlow</span>
                        <span class="logo-subtitle">Pipeline Visualization</span>
                    </div>
                </div>
                <div class="stats">
                    <div class="stat">
                        <div class="stat-value" id="node-count">0</div>
                        <div class="stat-label">Nodes</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="edge-count">0</div>
                        <div class="stat-label">Edges</div>
                    </div>
                </div>
            </div>
            <div class="legend">
                <div class="section-title">Node Types</div>
                <div class="legend-items" id="legend"></div>
            </div>
            <div class="details">
                <div class="details-empty" id="details-empty">
                    <div class="details-empty-icon"></div>
                    <div class="details-empty-text">Select a node</div>
                    <div class="details-empty-hint">Click on any node to view details</div>
                </div>
                <div class="details-content" id="details"></div>
            </div>
        </aside>
        <main class="graph-container">
            <svg class="graph" id="graph"></svg>
            <div class="controls">
                <button class="control-btn" id="btn-fit" title="Fit to view">⊡</button>
                <button class="control-btn" id="btn-zoom-in" title="Zoom in">+</button>
                <button class="control-btn" id="btn-zoom-out" title="Zoom out">−</button>
            </div>
            <div class="watermark">
                Built with <a href="https://signalflow-trading.com" target="_blank">SignalFlow</a>
            </div>
        </main>
    </div>
    <div class="tooltip" id="tooltip"></div>

<script>
const graphData = {graph_json};

// Node colors matching SignalFlow brand
const nodeColors = {
    data_source: '#2563eb',
    feature: '#0891b2',
    detector: '#7c3aed',
    labeler: '#db2777',
    validator: '#ca8a04',
    runner: '#dc2626',
    entry_rule: '#059669',
    exit_rule: '#ea580c'
};

// Lighter variants for gradients
const nodeColorsLight = {
    data_source: '#3b82f6',
    feature: '#22d3ee',
    detector: '#a78bfa',
    labeler: '#f472b6',
    validator: '#fbbf24',
    runner: '#f87171',
    entry_rule: '#34d399',
    exit_rule: '#fb923c'
};

const nodeLabels = {
    data_source: 'Data Source',
    feature: 'Feature',
    detector: 'Detector',
    labeler: 'Labeler',
    validator: 'Validator',
    runner: 'Runner',
    entry_rule: 'Entry Rule',
    exit_rule: 'Exit Rule'
};

const nodeIcons = {
    data_source: '◉',
    feature: '⬡',
    detector: '◆',
    labeler: '▣',
    validator: '✓',
    runner: '▶',
    entry_rule: '↗',
    exit_rule: '↘'
};

// Parse graph data
const nodes = graphData.elements.filter(e => !e.data.source);
const edges = graphData.elements.filter(e => e.data.source);

// Update stats
document.getElementById('node-count').textContent = nodes.length;
document.getElementById('edge-count').textContent = edges.length;

// Build legend
const legend = document.getElementById('legend');
const usedTypes = new Set(nodes.map(n => n.data.type));
usedTypes.forEach(type => {
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = `<span class="legend-dot" style="background:${nodeColors[type] || '#666'}"></span>${nodeLabels[type] || type}`;
    item.onclick = () => highlightType(type);
    legend.appendChild(item);
});

// Create dagre graph for layout
const g = new dagre.graphlib.Graph();
g.setGraph({
    rankdir: 'LR',
    nodesep: 70,
    ranksep: 120,
    marginx: 60,
    marginy: 60
});
g.setDefaultEdgeLabel(() => ({}));

nodes.forEach(n => {
    const label = n.data.label || n.data.id;
    const width = Math.max(140, label.length * 9 + 40);
    g.setNode(n.data.id, { label, width, height: 48, data: n.data });
});

edges.forEach(e => {
    g.setEdge(e.data.source, e.data.target);
});

dagre.layout(g);

// Setup SVG
const svg = d3.select('#graph');
const container = svg.append('g').attr('class', 'container');

// Define gradients
const defs = svg.append('defs');
Object.keys(nodeColors).forEach(type => {
    const grad = defs.append('linearGradient')
        .attr('id', `grad-${type}`)
        .attr('x1', '0%').attr('y1', '0%')
        .attr('x2', '100%').attr('y2', '100%');
    grad.append('stop')
        .attr('offset', '0%')
        .attr('stop-color', nodeColorsLight[type] || nodeColors[type]);
    grad.append('stop')
        .attr('offset', '100%')
        .attr('stop-color', nodeColors[type]);
});

// Arrow marker
defs.append('marker')
    .attr('id', 'arrowhead')
    .attr('viewBox', '0 -5 10 10')
    .attr('refX', 8)
    .attr('refY', 0)
    .attr('markerWidth', 6)
    .attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', 'M0,-5L10,0L0,5')
    .attr('fill', 'rgba(255,255,255,0.3)');

defs.append('marker')
    .attr('id', 'arrowhead-active')
    .attr('viewBox', '0 -5 10 10')
    .attr('refX', 8)
    .attr('refY', 0)
    .attr('markerWidth', 6)
    .attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', 'M0,-5L10,0L0,5')
    .attr('fill', '#3b82f6');

// Zoom behavior
const zoom = d3.zoom()
    .scaleExtent([0.1, 4])
    .on('zoom', (e) => container.attr('transform', e.transform));
svg.call(zoom);

// Draw edges first (below nodes)
const edgeGroup = container.append('g').attr('class', 'edges');
g.edges().forEach(e => {
    const edge = g.edge(e);
    const path = d3.line().curve(d3.curveBasis)(edge.points.map(p => [p.x, p.y]));

    edgeGroup.append('g')
        .attr('class', 'edge')
        .attr('data-source', e.v)
        .attr('data-target', e.w)
        .append('path')
        .attr('d', path)
        .attr('marker-end', 'url(#arrowhead)');
});

// Draw nodes
const nodeGroup = container.append('g').attr('class', 'nodes');
g.nodes().forEach(nodeId => {
    const node = g.node(nodeId);
    const type = node.data.type;
    const color = nodeColors[type] || '#666';

    const group = nodeGroup.append('g')
        .attr('class', 'node')
        .attr('data-id', nodeId)
        .attr('data-type', type)
        .attr('transform', `translate(${node.x},${node.y})`)
        .on('click', (event) => {
            event.stopPropagation();
            selectNode(nodeId, node.data);
        })
        .on('mouseenter', function(event) {
            showTooltip(event, node.data);
        })
        .on('mouseleave', hideTooltip);

    // Shape based on type with gradient fill
    if (type === 'data_source') {
        // Ellipse for data sources
        group.append('ellipse')
            .attr('rx', node.width / 2)
            .attr('ry', node.height / 2)
            .attr('fill', `url(#grad-${type})`)
            .attr('stroke', nodeColorsLight[type] || color)
            .attr('stroke-opacity', 0.6);
    } else if (type === 'detector') {
        // Diamond for detector
        const w = node.width / 2, h = node.height / 2;
        group.append('polygon')
            .attr('points', `0,${-h} ${w},0 0,${h} ${-w},0`)
            .attr('fill', `url(#grad-${type})`)
            .attr('stroke', nodeColorsLight[type] || color)
            .attr('stroke-opacity', 0.6);
    } else if (type === 'runner') {
        // Hexagon for runner
        const w = node.width / 2, h = node.height / 2;
        const hw = w * 0.85;
        group.append('polygon')
            .attr('points', `${-hw},0 ${-w*0.5},${-h} ${w*0.5},${-h} ${hw},0 ${w*0.5},${h} ${-w*0.5},${h}`)
            .attr('fill', `url(#grad-${type})`)
            .attr('stroke', nodeColorsLight[type] || color)
            .attr('stroke-opacity', 0.6);
    } else {
        // Rounded rectangle for others
        group.append('rect')
            .attr('x', -node.width / 2)
            .attr('y', -node.height / 2)
            .attr('width', node.width)
            .attr('height', node.height)
            .attr('rx', 10)
            .attr('fill', `url(#grad-${type})`)
            .attr('stroke', nodeColorsLight[type] || color)
            .attr('stroke-opacity', 0.6);
    }

    // Label
    group.append('text')
        .attr('text-anchor', 'middle')
        .attr('dy', '0.35em')
        .text(node.label.length > 20 ? node.label.slice(0, 18) + '…' : node.label);
});

// Fit to view
function fitToView() {
    const bounds = container.node().getBBox();
    const parent = svg.node().getBoundingClientRect();
    const scale = Math.min(
        parent.width / (bounds.width + 120),
        parent.height / (bounds.height + 120),
        1.5
    );
    const tx = (parent.width - bounds.width * scale) / 2 - bounds.x * scale;
    const ty = (parent.height - bounds.height * scale) / 2 - bounds.y * scale;
    svg.transition().duration(600).ease(d3.easeCubicOut).call(
        zoom.transform,
        d3.zoomIdentity.translate(tx, ty).scale(scale)
    );
}

// Selection state
let selectedNode = null;

function selectNode(id, data) {
    // Clear previous selection
    d3.selectAll('.node').classed('selected', false).classed('faded', false);
    d3.selectAll('.edge').classed('highlighted', false).classed('faded', false);
    d3.selectAll('.edge path').attr('marker-end', 'url(#arrowhead)');

    if (selectedNode === id) {
        selectedNode = null;
        document.getElementById('details-empty').style.display = 'flex';
        document.getElementById('details').classList.remove('active');
        return;
    }

    selectedNode = id;

    // Find connected nodes
    const connectedNodes = new Set([id]);
    const connectedEdges = new Set();

    edges.forEach((e, i) => {
        if (e.data.source === id) {
            connectedNodes.add(e.data.target);
            connectedEdges.add(i);
        }
        if (e.data.target === id) {
            connectedNodes.add(e.data.source);
            connectedEdges.add(i);
        }
    });

    // Apply highlighting
    d3.selectAll('.node').each(function() {
        const nodeId = d3.select(this).attr('data-id');
        d3.select(this)
            .classed('selected', nodeId === id)
            .classed('faded', !connectedNodes.has(nodeId));
    });

    d3.selectAll('.edge').each(function() {
        const src = d3.select(this).attr('data-source');
        const tgt = d3.select(this).attr('data-target');
        const connected = src === id || tgt === id;
        d3.select(this)
            .classed('highlighted', connected)
            .classed('faded', !connected);
        d3.select(this).select('path')
            .attr('marker-end', connected ? 'url(#arrowhead-active)' : 'url(#arrowhead)');
    });

    showDetails(data);
}

function showDetails(data) {
    document.getElementById('details-empty').style.display = 'none';
    const details = document.getElementById('details');
    details.classList.add('active');

    const color = nodeColors[data.type] || '#666';
    const typeLabel = nodeLabels[data.type] || data.type;

    let html = `
        <span class="details-badge" style="background:${color}; color: white;">${typeLabel}</span>
        <div class="details-name">${data.label}</div>
        <div class="details-section">
            <div class="details-section-title">Properties</div>
    `;

    const props = [];
    if (data.feature_class) props.push(['Class', data.feature_class]);
    if (data.exchange) props.push(['Exchange', data.exchange]);
    if (data.data_type) props.push(['Data Type', data.data_type]);
    if (data.detector_class) props.push(['Detector', data.detector_class]);
    if (data.is_global !== undefined) props.push(['Global', data.is_global ? 'Yes' : 'No']);

    if (props.length === 0) {
        props.push(['ID', data.id]);
    }

    props.forEach(([key, val]) => {
        html += `<div class="details-row"><span class="details-key">${key}</span><span class="details-value">${val}</span></div>`;
    });

    html += '</div>';

    if (data.requires?.length) {
        html += `
            <div class="details-section">
                <div class="details-section-title">Inputs</div>
                <div class="tag-list">${data.requires.map(c => `<span class="tag input">${c}</span>`).join('')}</div>
            </div>
        `;
    }

    if (data.outputs?.length) {
        html += `
            <div class="details-section">
                <div class="details-section-title">Outputs</div>
                <div class="tag-list">${data.outputs.map(c => `<span class="tag output">${c}</span>`).join('')}</div>
            </div>
        `;
    }

    details.innerHTML = html;
}

function highlightType(type) {
    // Flash nodes of this type
    d3.selectAll('.node').each(function() {
        const nodeType = d3.select(this).attr('data-type');
        d3.select(this).classed('faded', nodeType !== type);
    });
    d3.selectAll('.edge').classed('faded', true);

    setTimeout(() => {
        if (!selectedNode) {
            d3.selectAll('.node, .edge').classed('faded', false);
        }
    }, 1500);
}

// Tooltip
const tooltip = document.getElementById('tooltip');

function showTooltip(event, data) {
    if (selectedNode) return;
    const typeLabel = nodeLabels[data.type] || data.type;
    tooltip.textContent = `${typeLabel}: ${data.label}`;
    tooltip.style.left = (event.pageX + 12) + 'px';
    tooltip.style.top = (event.pageY - 28) + 'px';
    tooltip.classList.add('visible');
}

function hideTooltip() {
    tooltip.classList.remove('visible');
}

// Controls
document.getElementById('btn-fit').onclick = fitToView;
document.getElementById('btn-zoom-in').onclick = () => svg.transition().duration(300).call(zoom.scaleBy, 1.5);
document.getElementById('btn-zoom-out').onclick = () => svg.transition().duration(300).call(zoom.scaleBy, 0.67);

// Click outside to deselect
svg.on('click', (e) => {
    if (e.target === svg.node() || e.target.classList.contains('container')) {
        if (selectedNode) {
            selectNode(selectedNode, {});
        }
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && selectedNode) {
        selectNode(selectedNode, {});
    }
    if (e.key === 'f' || e.key === 'F') {
        fitToView();
    }
});

// Initial fit with delay for DOM render
setTimeout(fitToView, 150);
</script>
</body>
</html>"""


class HtmlRenderer:
    """Render PipelineGraph to interactive HTML with D3.js."""

    def __init__(self, graph: "PipelineGraph"):
        self.graph = graph

    def render(self, output_path: str | Path | None = None) -> str:
        """Render to HTML string or file."""
        graph_json = json.dumps(self.graph.to_cytoscape())
        html = HTML_TEMPLATE.replace("{graph_json}", graph_json)

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
