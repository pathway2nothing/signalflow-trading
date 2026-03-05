"""Signal-level features (meta-features from signal history).

This module provides the :class:`SignalFeature` base class for features
that operate on the *signal stream* rather than raw market data.

Two categories:
    - **Unsupervised**: signal frequency, entropy, flip-rate, streak, ...
    - **Supervised**: rolling accuracy, expected value, Bayesian posterior, ...

Interface lives here (``signalflow.signal_feature``); concrete
implementations are in ``signalflow-ta`` (``signalflow.ta.signal_features``).
"""

from signalflow.signal_feature.base import SignalFeature

__all__ = ["SignalFeature"]
