"""
SignalFlow - real-time-first framework for trading signal research and execution.

import signalflow as sf

The public surface is the six-noun model: Dataset, Transform, Models, Flow,
Engine, Run - plus the WoE/IV feature policy, samplers, detectors, and strategy
models.
"""

from signalflow._version import __version__


from signalflow.enums import (
    FALL,
    NONE,
    RISE,
    ComponentType,
    IntentKind,
    OrderType,
    PositionSide,
    Provenance,
    RunMode,
    Side,
    Signal,
)
from signalflow.errors import (
    ArtifactError,
    DegenerateTargetError,
    FingerprintMismatch,
    FlowConfigError,
    KillSwitchTripped,
    LeakageError,
    PipeError,
    SchemaVersionError,
    SignalFlowError,
    UnfittedTransformError,
    UnknownComponentError,
    UntrainedModelError,
)
from signalflow.registry import registry


from signalflow.data import BinanceSource, Dataset, MemorySource, data


from signalflow.transform import SMA, Feature, FeaturePipe, Transform, build_pipe
from signalflow.transform.encode import Binning, IVSelector, WoE


from signalflow import target
from signalflow.target import FixedHorizon, Target, TripleBarrier
from signalflow.sampler import (
    CUSUMSampler,
    MetaLabelingSampler,
    SampleSet,
    Sampler,
    UniformSampler,
    UniquenessSampler,
)


from signalflow.model import (
    ForecastModel,
    MaxValidator,
    MeanValidator,
    VoteValidator,
    WalkForwardFold,
    WalkForwardResult,
    walk_forward,
)
from signalflow.detector import (
    MarketDropDetector,
    RevertDetector,
    SignalDetector,
    SmaCrossDetector,
    ThresholdDetector,
)


from signalflow.engine import (
    BinanceBroker,
    Broker,
    Engine,
    ExchangeBroker,
    Fill,
    Intent,
    Order,
    PortfolioSnapshot,
    Position,
    SimBroker,
)
from signalflow.strategy import (
    OBSERVATION_SCHEMA_VERSION,
    Entry,
    Exit,
    Observation,
    Risk,
    RulesStrategy,
    StrategyModel,
)
from signalflow.flow import Flow, LiveFeed, PollingFeed, ReplayFeed, Run, run_live_loop


from signalflow.experiment import ArtifactCache, Experiment, Scorecard, bootstrap_ci, monte_carlo_bounds


_OPT = []
try:
    from signalflow.strategy import LLMClient, LLMStrategy, OpenAICompatClient

    _OPT += ["LLMStrategy", "LLMClient", "OpenAICompatClient"]
except Exception:
    pass

__all__ = [
    "__version__",
    "RISE",
    "FALL",
    "NONE",
    "Signal",
    "Side",
    "OrderType",
    "PositionSide",
    "IntentKind",
    "RunMode",
    "Provenance",
    "ComponentType",
    "registry",
    "SignalFlowError",
    "UntrainedModelError",
    "FlowConfigError",
    "LeakageError",
    "PipeError",
    "KillSwitchTripped",
    "ArtifactError",
    "FingerprintMismatch",
    "SchemaVersionError",
    "UnknownComponentError",
    "UnfittedTransformError",
    "DegenerateTargetError",
    "data",
    "Dataset",
    "BinanceSource",
    "MemorySource",
    "Transform",
    "Feature",
    "FeaturePipe",
    "build_pipe",
    "SMA",
    "WoE",
    "Binning",
    "IVSelector",
    "target",
    "Target",
    "FixedHorizon",
    "TripleBarrier",
    "Sampler",
    "SampleSet",
    "UniformSampler",
    "MetaLabelingSampler",
    "CUSUMSampler",
    "UniquenessSampler",
    "ForecastModel",
    "MeanValidator",
    "MaxValidator",
    "VoteValidator",
    "walk_forward",
    "WalkForwardResult",
    "WalkForwardFold",
    "SignalDetector",
    "SmaCrossDetector",
    "ThresholdDetector",
    "RevertDetector",
    "MarketDropDetector",
    "Engine",
    "Broker",
    "SimBroker",
    "ExchangeBroker",
    "BinanceBroker",
    "Fill",
    "Order",
    "Intent",
    "Position",
    "PortfolioSnapshot",
    "RulesStrategy",
    "Entry",
    "Exit",
    "Risk",
    "Observation",
    "StrategyModel",
    "OBSERVATION_SCHEMA_VERSION",
    "Flow",
    "Run",
    "LiveFeed",
    "ReplayFeed",
    "PollingFeed",
    "run_live_loop",
    "Experiment",
    "Scorecard",
    "ArtifactCache",
    "bootstrap_ci",
    "monte_carlo_bounds",
] + _OPT
