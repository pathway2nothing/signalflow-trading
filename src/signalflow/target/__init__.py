"""Target specs consumed by ForecastModel.fit."""

from signalflow.target import adapter
from signalflow.target.anomaly_labeler import AnomalyLabeler
from signalflow.target.base import LABEL_COL, Target, make_target, register_target
from signalflow.target.directional_mean_reversion_labeler import DirectionalMeanReversionLabeler
from signalflow.target.drawdown_labeler import DrawdownLabeler
from signalflow.target.fixed_horizon import FixedHorizon
from signalflow.target.fixed_horizon_labeler import FixedHorizonLabeler
from signalflow.target.flash_move_labeler import FlashMoveLabeler
from signalflow.target.hmm_vol_regime_labeler import HMMVolRegime2StateLabeler
from signalflow.target.labeler import Labeler
from signalflow.target.market_wide_volatility_labeler import MarketWideVolatilityRegimeLabeler
from signalflow.target.mean_reversion_magnitude_labeler import MeanReversionMagnitudeLabeler
from signalflow.target.meta_label_labeler import MetaLabelLabeler
from signalflow.target.multi_horizon_mean_reversion_labeler import MultiHorizonMeanReversionLabeler
from signalflow.target.multi_target_generator import (
    DEFAULT_HORIZONS,
    DEFAULT_TARGET_TYPES,
    HorizonConfig,
    MultiTargetGenerator,
    TargetType,
)
from signalflow.target.path_labeler import (
    HurstRegimeLabeler,
    MeanReversionEventLabeler,
    TrendBreakLabeler,
)
from signalflow.target.sharpe_tercile_labeler import SharpeTercileLabeler
from signalflow.target.structure_labeler import StructureLabeler, ZigzagStructureLabeler
from signalflow.target.take_profit_labeler import TakeProfitLabeler
from signalflow.target.time_to_barrier_labeler import TimeToBarrierLabeler
from signalflow.target.trend_scanning import TrendScanningLabeler
from signalflow.target.triple_barrier import TripleBarrier
from signalflow.target.triple_barrier_labeler import TripleBarrierLabeler
from signalflow.target.utils import mask_targets_by_signals, mask_targets_by_timestamps
from signalflow.target.volatility_labeler import VolatilityRegimeLabeler
from signalflow.target.volatility_shock_labeler import VolatilityShockLabeler
from signalflow.target.volume_climax_labeler import VolumeClimaxLabeler
from signalflow.target.volume_labeler import VolumeRegimeLabeler

__all__ = [

    "Target",
    "FixedHorizon",
    "TripleBarrier",
    "make_target",
    "register_target",
    "LABEL_COL",

    "Labeler",
    "adapter",
    "mask_targets_by_signals",
    "mask_targets_by_timestamps",

    "DEFAULT_HORIZONS",
    "DEFAULT_TARGET_TYPES",
    "HorizonConfig",
    "MultiTargetGenerator",
    "TargetType",

    "AnomalyLabeler",
    "DirectionalMeanReversionLabeler",
    "DrawdownLabeler",
    "FixedHorizonLabeler",
    "FlashMoveLabeler",
    "HMMVolRegime2StateLabeler",
    "HurstRegimeLabeler",
    "MarketWideVolatilityRegimeLabeler",
    "MeanReversionEventLabeler",
    "MeanReversionMagnitudeLabeler",
    "MetaLabelLabeler",
    "MultiHorizonMeanReversionLabeler",
    "SharpeTercileLabeler",
    "StructureLabeler",
    "TakeProfitLabeler",
    "TimeToBarrierLabeler",
    "TrendBreakLabeler",
    "TrendScanningLabeler",
    "TripleBarrierLabeler",
    "VolatilityRegimeLabeler",
    "VolatilityShockLabeler",
    "VolumeClimaxLabeler",
    "VolumeRegimeLabeler",
    "ZigzagStructureLabeler",
]
