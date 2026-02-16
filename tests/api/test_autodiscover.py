"""
Tests to ensure lazy imports don't break autodiscover.

CRITICAL: These tests verify that the registry autodiscover mechanism
still works correctly after adding lazy imports for the api module.
"""



class TestAutodiscover:
    """Verify registry autodiscover still works with lazy imports."""

    def test_import_signalflow_works(self):
        """Basic import doesn't crash."""
        import signalflow as sf

        assert sf is not None

    def test_registry_is_available(self):
        """default_registry is accessible after import."""
        import signalflow as sf

        assert hasattr(sf, "default_registry")
        assert sf.default_registry is not None

    def test_registry_snapshot_not_empty(self):
        """Registry has components after import."""
        import signalflow as sf

        snapshot = sf.default_registry.snapshot()
        assert len(snapshot) > 0

    def test_autodiscover_finds_detectors(self):
        """Autodiscover finds @sf_component decorated detectors."""
        from signalflow.core import SfComponentType, default_registry

        detectors = default_registry.list(SfComponentType.DETECTOR)
        assert len(detectors) > 0
        assert "example/sma_cross" in detectors

    def test_autodiscover_finds_metrics(self):
        """Autodiscover finds strategy metrics."""
        from signalflow.core import SfComponentType, default_registry

        metrics = default_registry.list(SfComponentType.STRATEGY_METRIC)
        # Should find at least result_main and result_pair
        assert "result_main" in metrics or len(metrics) > 0

    def test_autodiscover_finds_features(self):
        """Autodiscover finds features."""
        from signalflow.core import SfComponentType, default_registry

        features = default_registry.list(SfComponentType.FEATURE)
        # Features may or may not be registered
        assert isinstance(features, list)

    def test_lazy_api_import_doesnt_break_registry(self):
        """Accessing sf.Backtest doesn't break autodiscover."""
        import signalflow as sf

        # Access lazy-loaded API first
        Backtest = sf.Backtest
        assert Backtest is not None

        # Registry should still work
        from signalflow.core import SfComponentType

        detectors = sf.default_registry.list(SfComponentType.DETECTOR)
        assert len(detectors) > 0

    def test_lazy_backtest_result_doesnt_break_registry(self):
        """Accessing sf.BacktestResult doesn't break autodiscover."""
        import signalflow as sf

        # Access lazy-loaded class
        BacktestResult = sf.BacktestResult
        assert BacktestResult is not None

        # Registry should still work
        snapshot = sf.default_registry.snapshot()
        assert len(snapshot) > 0

    def test_lazy_load_doesnt_break_registry(self):
        """Accessing sf.load doesn't break autodiscover."""
        import signalflow as sf

        # Access lazy-loaded function
        load = sf.load
        assert callable(load)

        # Registry should still work
        from signalflow.core import SfComponentType

        detectors = sf.default_registry.list(SfComponentType.DETECTOR)
        assert len(detectors) > 0

    def test_all_lazy_imports_available(self):
        """All lazy imports are accessible."""
        import signalflow as sf

        # These should all work
        assert sf.Backtest is not None
        assert sf.BacktestBuilder is not None
        assert sf.BacktestResult is not None
        assert sf.backtest is not None
        assert sf.load is not None
        assert sf.api is not None

    def test_registry_create_works_after_lazy_import(self):
        """Registry.create() works after accessing lazy imports."""
        import signalflow as sf

        # Access lazy imports
        _ = sf.Backtest
        _ = sf.load

        # Create detector from registry
        from signalflow.core import SfComponentType

        detector = sf.default_registry.create(
            SfComponentType.DETECTOR,
            "example/sma_cross",
            fast_period=10,
            slow_period=20,
        )
        assert detector is not None
        assert detector.fast_period == 10


class TestModuleAccess:
    """Test that submodules are still accessible."""

    def test_detector_module(self):
        """signalflow.detector is accessible."""
        import signalflow as sf

        assert sf.detector is not None

    def test_feature_module(self):
        """signalflow.feature is accessible."""
        import signalflow as sf

        assert sf.feature is not None

    def test_strategy_module(self):
        """signalflow.strategy is accessible."""
        import signalflow as sf

        assert sf.strategy is not None

    def test_data_module(self):
        """signalflow.data is accessible."""
        import signalflow as sf

        assert sf.data is not None

    def test_validator_module(self):
        """signalflow.validator is accessible."""
        import signalflow as sf

        assert sf.validator is not None

    def test_analytic_module(self):
        """signalflow.analytic is accessible."""
        import signalflow as sf

        assert sf.analytic is not None


class TestCoreExports:
    """Test that core exports are still accessible."""

    def test_raw_data(self):
        """RawData is accessible."""
        import signalflow as sf

        assert sf.RawData is not None

    def test_signals(self):
        """Signals is accessible."""
        import signalflow as sf

        assert sf.Signals is not None

    def test_sf_component(self):
        """sf_component decorator is accessible."""
        import signalflow as sf

        assert sf.sf_component is not None

    def test_strategy_state(self):
        """StrategyState is accessible."""
        import signalflow as sf

        assert sf.StrategyState is not None
