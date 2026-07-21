# Target

Targets define how forward returns are turned into labels, and samplers select
which observations to keep for training.

## Target inventory

`sf list target` separates the two native V5 targets from the legacy labelers
(both register under stable `flow.yaml` names; the split is presentation only).

### Native V5 targets

| name | class | summary |
| --- | --- | --- |
| `fixed_horizon` | `FixedHorizon` | Label 1 if close rises by more than `threshold` after `bars` bars. |
| `triple_barrier` | `TripleBarrier` | Meta-labeling target: success = tp hit before sl within the horizon. |
| `vol_triple_barrier` | `VolTripleBarrier` | Triple barrier with tp/sl as multiples of trailing EWMA return volatility (optional 3-class). |
| `vol_horizon` | `VolHorizon` | Fixed horizon with a volatility-scaled dead zone; null inside the zone. |
| `reversion_barrier` | `ReversionBarrier` | From a below-anchor entry, 1 if price reverts to the SMA anchor before the stop. |

`reversion_barrier` overlaps conceptually with the legacy `mean_reversion_event`
(both label reversion within a horizon), but uses a distinct SMA-anchor-plus-stop
barrier rather than an overstretch/z-score event. Trend scanning is already provided
by the legacy `trend_scanning` labeler, so no native duplicate was added.

### Legacy labelers

| name | class | summary |
| --- | --- | --- |
| `anomaly` | `AnomalyLabeler` | Labels black swan and flash crash events in historical data. |
| `directional_mean_reversion` | `DirectionalMeanReversionLabeler` | Three-class long/short/none mean-reversion label. |
| `drawdown` | `DrawdownLabeler` | Label bars by forward path-risk metric, terciled within a rolling base. |
| `fixed_horizon_labeler` | `FixedHorizonLabeler` | Fixed-horizon labeling (legacy pandas-derived variant). |
| `flash_move` | `FlashMoveLabeler` | Label bars by extreme forward short-horizon returns. |
| `hmm_vol_regime_2state` | `HMMVolRegime2StateLabeler` | 2-state Gaussian HMM (calm / turbulent) on rolling log-vol. |
| `hurst_regime` | `HurstRegimeLabeler` | Label bars by forward Hurst-exponent regime. |
| `market_wide_volatility_regime` | `MarketWideVolatilityRegimeLabeler` | Cross-sectional vol regime: mean forward vol across pairs, terciled. |
| `mean_reversion_event` | `MeanReversionEventLabeler` | Label bars by whether an overstretched price reverts within horizon. |
| `mean_reversion_magnitude` | `MeanReversionMagnitudeLabeler` | Continuous revert-strength target plus three soft buckets. |
| `meta_label` | `MetaLabelLabeler` | Binary meta-label conditional on a primary directional signal. |
| `multi_horizon_mean_reversion` | `MultiHorizonMeanReversionLabeler` | Average `MeanReversionEventLabeler` posteriors across horizons. |
| `sharpe_tercile` | `SharpeTercileLabeler` | Label bars by tercile of forward risk-adjusted log-return. |
| `structure` | `StructureLabeler` | Label local tops and bottoms using a symmetric window. |
| `take_profit` | `TakeProfitLabeler` | First-touch labeling with symmetric fixed-percentage barriers. |
| `time_to_barrier` | `TimeToBarrierLabeler` | Time-to-first-touch triple barrier with survival-style outputs. |
| `trend_break` | `TrendBreakLabeler` | Label bars by whether forward OLS slope flips sign vs past slope. |
| `trend_scanning` | `TrendScanningLabeler` | Label bars using De Prado's trend scanning method. |
| `triple_barrier_labeler` | `TripleBarrierLabeler` | Triple-barrier labeling (De Prado), Numba-accelerated. |
| `volatility_regime` | `VolatilityRegimeLabeler` | Label bars by forward realized volatility regime. |
| `volatility_shock` | `VolatilityShockLabeler` | Label bars by forward-vs-past volatility z-score shock. |
| `volume_climax` | `VolumeClimaxLabeler` | Label bars by forward max-volume vs trailing SMA ratio. |
| `volume_regime` | `VolumeRegimeLabeler` | Label bars by forward volume regime. |
| `zigzag_structure` | `ZigzagStructureLabeler` | Label local tops and bottoms using a full-series zigzag algorithm. |

The two confusable near-duplicates differ by parameters and generation:
`fixed_horizon` (native) takes `bars`/`threshold` and emits a 0/1 rise label,
while `fixed_horizon_labeler` (legacy) is the older pandas-derived variant;
`triple_barrier` (native) is the meta-labeling tp/sl race, while
`triple_barrier_labeler` (legacy) is the Numba-accelerated De Prado port with a
different parameter set.

## Labelers

::: signalflow.Target
    options:
      show_root_heading: true

::: signalflow.FixedHorizon
    options:
      show_root_heading: true

::: signalflow.TripleBarrier
    options:
      show_root_heading: true

## Samplers

::: signalflow.Sampler
    options:
      show_root_heading: true

::: signalflow.SampleSet
    options:
      show_root_heading: true

::: signalflow.UniformSampler
    options:
      show_root_heading: true

::: signalflow.MetaLabelingSampler
    options:
      show_root_heading: true

::: signalflow.CUSUMSampler
    options:
      show_root_heading: true

::: signalflow.UniquenessSampler
    options:
      show_root_heading: true
