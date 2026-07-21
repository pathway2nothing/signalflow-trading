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
    RegistryError,
    SchemaVersionError,
    SignalFlowError,
    UnfittedTransformError,
    UnknownComponentError,
    UntrainedModelError,
)
from signalflow.registry import registry
from signalflow.decorators import (
    broker as register_broker,
    detector as register_detector,
    feature as register_feature,
    metric as register_metric,
    model as register_model,
    sampler as register_sampler,
    source as register_source,
    strategy as register_strategy,
    transform as register_transform,
)


from signalflow.data import BinanceSource, Dataset, MemorySource, data


from signalflow.transform import SMA, Feature, FeaturePipe, Transform, build_pipe
from signalflow.transform.encode import Binning, IVSelector, WoE
from signalflow.transform.store import FeatureStore


from signalflow import target
from signalflow.target import FixedHorizon, ReversionBarrier, Target, TripleBarrier, VolHorizon, VolTripleBarrier
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
    classification_scorecard,
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


from signalflow.experiment import (
    ArtifactCache,
    Experiment,
    Scorecard,
    bootstrap_ci,
    experiment_run,
    monte_carlo_bounds,
    run_experiment,
    seed_everything,
)


_OPT = []
try:
    from signalflow.strategy import (
        LLMClient as LLMClient,
        LLMStrategy as LLMStrategy,
        OpenAICompatClient as OpenAICompatClient,
    )

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
    "register_transform",
    "register_feature",
    "register_detector",
    "register_model",
    "register_strategy",
    "register_sampler",
    "register_broker",
    "register_metric",
    "register_source",
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
    "RegistryError",
    "data",
    "Dataset",
    "BinanceSource",
    "MemorySource",
    "Transform",
    "Feature",
    "FeaturePipe",
    "FeatureStore",
    "build_pipe",
    "SMA",
    "WoE",
    "Binning",
    "IVSelector",
    "target",
    "Target",
    "FixedHorizon",
    "TripleBarrier",
    "VolTripleBarrier",
    "VolHorizon",
    "ReversionBarrier",
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
    "classification_scorecard",
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
    "experiment_run",
    "seed_everything",
    "run_experiment",
    *_OPT,
]
