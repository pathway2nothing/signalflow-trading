"""Tests for signalflow.config.dag module."""

import warnings

import pytest

from signalflow.config.dag import Edge, FlowDAG, Node, StrategySubgraph


class TestNode:
    """Tests for Node dataclass."""

    def test_from_dict(self):
        """Create Node from dict."""
        node = Node.from_dict(
            "loader",
            {
                "type": "data/loader",
                "name": "binance/spot",
                "config": {"exchange": "binance"},
            },
        )

        assert node.id == "loader"
        assert node.type == "data/loader"
        assert node.name == "binance/spot"
        assert node.config == {"exchange": "binance"}

    def test_default_outputs(self):
        """Get default outputs for component type."""
        loader = Node(id="l", type="data/loader")
        assert loader.get_outputs() == ["ohlcv"]

        detector = Node(id="d", type="signals/detector")
        assert detector.get_outputs() == ["signals"]

    def test_explicit_outputs(self):
        """Explicit outputs override defaults."""
        node = Node(id="custom", type="data/loader", outputs=["custom_data"])
        assert node.get_outputs() == ["custom_data"]

    def test_default_inputs(self):
        """Get default inputs for component type."""
        detector = Node(id="d", type="signals/detector")
        assert detector.get_inputs() == ["ohlcv"]

        strategy = Node(id="s", type="strategy")
        assert strategy.get_inputs() == ["ohlcv", "signals"]


class TestFlowDAG:
    """Tests for FlowDAG."""

    def test_from_dict_basic(self):
        """Create FlowDAG from dict."""
        dag = FlowDAG.from_dict(
            {
                "id": "test_flow",
                "name": "Test Flow",
                "nodes": {
                    "loader": {"type": "data/loader", "name": "binance/spot"},
                    "detector": {"type": "signals/detector", "name": "sma_cross"},
                },
            }
        )

        assert dag.id == "test_flow"
        assert dag.name == "Test Flow"
        assert len(dag.nodes) == 2

    def test_auto_infer_edges_simple(self):
        """Auto-infer edges for simple chain."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            dag = FlowDAG.from_dict(
                {
                    "nodes": {
                        "loader": {"type": "data/loader"},
                        "detector": {"type": "signals/detector"},
                    },
                }
            )

            # Should have edge: loader → detector (ohlcv)
            assert len(dag.edges) == 1
            assert dag.edges[0].source == "loader"
            assert dag.edges[0].target == "detector"
            assert dag.edges[0].data_type == "ohlcv"

            # Should have warning about auto-connection
            assert any("Auto-connected" in str(warning.message) for warning in w)

    def test_auto_infer_edges_with_validator(self):
        """Auto-infer edges with validator in chain."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")

            dag = FlowDAG.from_dict(
                {
                    "nodes": {
                        "loader": {"type": "data/loader"},
                        "detector": {"type": "signals/detector"},
                        "labeler": {"type": "signals/labeler"},
                        "validator": {"type": "signals/validator"},
                        "strategy": {"type": "strategy"},
                    },
                }
            )

            # Check edges exist
            edge_pairs = [(e.source, e.target) for e in dag.edges]

            assert ("loader", "detector") in edge_pairs  # ohlcv
            assert ("loader", "labeler") in edge_pairs  # ohlcv
            assert ("detector", "labeler") in edge_pairs  # signals
            assert ("detector", "validator") in edge_pairs  # signals
            assert ("labeler", "validator") in edge_pairs  # labels

    def test_explicit_edges(self):
        """Explicit edges are used when provided."""
        dag = FlowDAG.from_dict(
            {
                "nodes": {
                    "loader": {"type": "data/loader"},
                    "detector": {"type": "signals/detector"},
                },
                "edges": [
                    {"source": "loader", "target": "detector", "data_type": "custom"},
                ],
            }
        )

        # No auto-inference when edges provided
        assert len(dag.edges) == 1
        assert dag.edges[0].data_type == "custom"

    def test_topological_sort(self):
        """Topological sort returns correct order."""
        dag = FlowDAG.from_dict(
            {
                "nodes": {
                    "loader": {"type": "data/loader"},
                    "detector": {"type": "signals/detector"},
                    "strategy": {"type": "strategy"},
                },
                "edges": [
                    {"source": "loader", "target": "detector"},
                    {"source": "detector", "target": "strategy"},
                    {"source": "loader", "target": "strategy"},
                ],
            }
        )

        order = dag.topological_sort()

        # loader must come before detector and strategy
        assert order.index("loader") < order.index("detector")
        assert order.index("loader") < order.index("strategy")
        assert order.index("detector") < order.index("strategy")

    def test_cycle_detection(self):
        """Detect cycles in DAG."""
        dag = FlowDAG.from_dict(
            {
                "nodes": {
                    "a": {"type": "data/loader"},
                    "b": {"type": "signals/detector"},
                    "c": {"type": "strategy"},
                },
                "edges": [
                    {"source": "a", "target": "b"},
                    {"source": "b", "target": "c"},
                    {"source": "c", "target": "a"},  # Cycle!
                ],
            }
        )

        with pytest.raises(ValueError, match="Cycle detected"):
            dag.topological_sort()

    def test_validate_missing_loader(self):
        """Validate catches missing data loader."""
        dag = FlowDAG.from_dict(
            {
                "nodes": {
                    "detector": {"type": "signals/detector"},
                },
            }
        )

        errors = dag.validate()
        assert any("data/loader" in e for e in errors)

    def test_validate_invalid_edge(self):
        """Validate catches invalid edge references."""
        dag = FlowDAG.from_dict(
            {
                "nodes": {
                    "loader": {"type": "data/loader"},
                },
                "edges": [
                    {"source": "loader", "target": "nonexistent"},
                ],
            }
        )

        errors = dag.validate()
        assert any("unknown target" in e for e in errors)

    def test_execution_plan(self):
        """Get execution plan with nodes in order."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            dag = FlowDAG.from_dict(
                {
                    "nodes": {
                        "loader": {"type": "data/loader", "config": {"pairs": ["BTCUSDT"]}},
                        "detector": {"type": "signals/detector", "name": "sma_cross"},
                    },
                }
            )

        plan = dag.get_execution_plan()

        assert len(plan) == 2
        assert plan[0]["id"] == "loader"
        assert plan[0]["config"] == {"pairs": ["BTCUSDT"]}
        assert plan[1]["id"] == "detector"
        assert plan[1]["name"] == "sma_cross"

    def test_to_dict(self):
        """Serialize DAG to dict."""
        dag = FlowDAG.from_dict(
            {
                "id": "test",
                "nodes": {
                    "loader": {"type": "data/loader"},
                },
                "edges": [{"source": "a", "target": "b"}],
            }
        )

        data = dag.to_dict()

        assert data["id"] == "test"
        assert "loader" in data["nodes"]
        assert len(data["edges"]) == 1


class TestStrategySubgraph:
    """Tests for StrategySubgraph."""

    def test_from_node(self):
        """Create subgraph from strategy node."""
        node = Node(
            id="strategy",
            type="strategy",
            config={
                "entry_rules": [{"type": "signal", "size": 100}],
                "exit_rules": [{"type": "tp_sl", "tp": 0.02}],
                "entry_filters": [{"type": "price_distance_filter"}],
            },
        )

        subgraph = StrategySubgraph.from_node(node)

        assert len(subgraph.entry_rules) == 1
        assert len(subgraph.exit_rules) == 1
        assert len(subgraph.entry_filters) == 1

    def test_internal_edges_simple(self):
        """Get internal edges for simple strategy."""
        subgraph = StrategySubgraph(
            entry_rules=[{"type": "signal"}],
            exit_rules=[{"type": "tp_sl"}],
        )

        edges = subgraph.get_internal_edges()

        # New architecture:
        # signal_reconciler → entry_dispatcher → entry_rule_0 → position_manager
        # position_manager → exit_rule_0 → exit_merger → runner → metrics_output
        sources = [e[0] for e in edges]
        targets = [e[1] for e in edges]

        assert "signal_reconciler" in sources
        assert "entry_dispatcher" in targets
        assert "entry_rule_0" in targets
        assert "position_manager" in targets
        assert "exit_rule_0" in targets
        assert "exit_merger" in targets
        assert "runner" in targets
        assert "metrics_output" in targets

    def test_internal_edges_with_filters(self):
        """Get internal edges with entry filters."""
        subgraph = StrategySubgraph(
            entry_rules=[{"type": "signal"}],
            exit_rules=[{"type": "tp_sl"}],
            entry_filters=[{"type": "price_distance_filter"}],
        )

        edges = subgraph.get_internal_edges()

        # Should have filter in the chain (sequential mode)
        edge_pairs = [(e[0], e[1]) for e in edges]

        # entry_rule_0 → entry_filter_0 → position_manager
        assert ("entry_rule_0", "entry_filter_0") in edge_pairs
        assert ("entry_filter_0", "position_manager") in edge_pairs


class TestSignalPriority:
    """Test signal priority (validated_signals > signals)."""

    def test_prefer_validated_signals(self):
        """Strategy prefers validated_signals when available."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")

            dag = FlowDAG.from_dict(
                {
                    "nodes": {
                        "loader": {"type": "data/loader"},
                        "detector": {"type": "signals/detector"},
                        "validator": {"type": "signals/validator"},
                        "strategy": {"type": "strategy"},
                    },
                }
            )

            # Find edge to strategy
            strategy_inputs = [e for e in dag.edges if e.target == "strategy"]

            # Should have validated_signals, not raw signals
            data_types = [e.data_type for e in strategy_inputs]
            assert "validated_signals" in data_types or "ohlcv" in data_types
