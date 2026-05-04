"""Plugin development helpers -- scaffolding, validation, testing.

Usage::

    import signalflow as sf

    # Generate a component file
    sf.contrib.scaffold("my_rsi_detector", component_type="detector")

    # Validate a component class
    sf.contrib.validate_component(MyDetector)

    # Quick sanity check
    sf.contrib.check_component(MyDetector)
"""

from signalflow.contrib.scaffold import check_component, scaffold, validate_component

__all__ = ["check_component", "scaffold", "validate_component"]
