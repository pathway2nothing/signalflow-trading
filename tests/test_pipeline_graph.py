"""Feature configuration as a tree: nesting is the edge, and it scopes the encoder."""

import polars as pl
import pytest

import signalflow as sf
from signalflow.transform.graph import parse_nodes


def test_nesting_orders_producers_before_the_consumer():
    tree = {
        "woe_sma": {
            "registry": "woe",
            "inputs": [
                {"registry": "sma", "params": {"length": 10}},
                {"registry": "sma", "params": {"length": 50}},
            ],
        }
    }
    names = [n for n, _ in parse_nodes(tree)]
    kinds = [t.name for _, t in parse_nodes(tree)]
    assert kinds == ["sma", "sma", "woe"]
    assert names[-1] == "woe_sma" and names[0].startswith("woe_sma.")


def test_nesting_scopes_the_encoder_to_its_own_group():
    """One encoder per feature group: WoE sees exactly what its inputs produce."""
    tree = {
        "woe_fast": {"registry": "woe", "inputs": [{"registry": "sma", "params": {"length": 10}}]},
        "woe_slow": {"registry": "woe", "inputs": [{"registry": "sma", "params": {"length": 50}}]},
    }
    pipe = sf.FeaturePipeline.from_tree(tree)
    encoders = [t for t in pipe.transforms if t.name == "woe"]
    assert [t.columns for t in encoders] == [["sma_10"], ["sma_50"]]


def test_explicit_columns_win_over_the_nesting_scope():
    tree = {
        "woe": {
            "registry": "woe",
            "params": {"columns": ["sma_10"]},
            "inputs": [
                {"registry": "sma", "params": {"length": 10}},
                {"registry": "sma", "params": {"length": 50}},
            ],
        }
    }
    pipe = sf.FeaturePipeline.from_tree(tree)
    assert pipe.transforms[-1].columns == ["sma_10"]


def test_tree_round_trips_through_yaml(tmp_path):
    tree = {
        "woe_sma": {"registry": "woe", "inputs": [{"registry": "sma", "params": {"length": 20}}]},
        "keep": {"registry": "iv_selector", "params": {"min_iv": 0.1}},
    }
    pipe = sf.FeaturePipeline.from_tree(tree)
    path = tmp_path / "pipeline.yaml"
    pipe.save(str(path))
    back = sf.FeaturePipeline.from_yaml(str(path))
    assert [t.to_config() for t in back.transforms] == [t.to_config() for t in pipe.transforms]
    assert "pipeline" in path.read_text(encoding="utf-8")


def test_graph_shows_who_feeds_whom():
    pipe = sf.FeaturePipeline.from_tree(
        {"woe": {"registry": "woe", "inputs": [{"registry": "sma", "params": {"length": 20}}]}}
    )
    graph = pipe.graph
    assert [g["transform"] for g in graph] == ["sma", "woe"]
    assert graph[0]["reads"] == ["close"] and graph[0]["fed_by"] == []
    assert graph[1]["fit"] is True and graph[1]["fed_by"] == [graph[0]["node"]]


@pytest.mark.parametrize(
    "bad,match",
    [
        ({"x": {"registry": "no_such_thing"}}, "unknown transform"),
        ({"x": {"registry": "sma", "cls": "EMA"}}, "disagrees with the code"),
        ({"x": {"registry": "sma", "fit": True}}, "does not require fitting"),
        ({"x": {"registry": "sma", "oops": 1}}, "unknown keys"),
        ({"x": {"params": {}}}, "no 'registry'"),
        ({"x": {"registry": "sma", "params": {"nope": 1}}}, "rejects params"),
        ({}, "empty"),
    ],
)
def test_a_broken_tree_fails_at_parse_time(bad, match):
    with pytest.raises(sf.PipelineError, match=match):
        parse_nodes(bad)


def test_a_forest_of_groups_computes(ds):
    """Two independent groups, each with its own encoder, run end to end."""
    pipe = sf.FeaturePipeline.from_tree(
        {
            "fast": {"registry": "woe", "inputs": [{"registry": "sma", "params": {"length": 10}}]},
            "slow": {"registry": "woe", "inputs": [{"registry": "sma", "params": {"length": 50}}]},
        }
    )
    y = (ds.frame.get_column("close").shift(-12) > ds.frame.get_column("close")).cast(pl.Int8)
    out = pipe.fit(ds.frame, y).compute(ds.frame)
    assert {"sma_10__woe", "sma_50__woe"} <= set(out.columns)
    assert pipe.effective_warmups() == [10, 10, 50, 50]
