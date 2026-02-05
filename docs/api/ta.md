# Technical Analysis Module (signalflow-ta)

API reference for `signalflow.ta` — 199+ technical indicators.

!!! info "Separate Package"
    Install with `pip install signalflow-ta`. See the [ecosystem page](../ecosystem/signalflow-ta.md) for usage guide, examples, and pipeline factories.

All classes extend `signalflow.feature.base.Feature` and are registered via `@sf_component`. Import directly from `signalflow.ta`:

```python
import signalflow.ta as ta
rsi = ta.RsiMom(period=14, normalized=True)
```

---

## Momentum

`signalflow.ta.momentum` — 18 indicators

### Core (`signalflow.ta.momentum.core`)

| Class | `@sf_component` | Parameters | Output |
|-------|----------------|------------|--------|
| `RsiMom` | `momentum/rsi` | `period=14`, `source_col="close"` | `rsi_{period}` |
| `RocMom` | `momentum/roc` | `period=12`, `source_col="close"` | `roc_{period}` |
| `MomMom` | `momentum/mom` | `period=10`, `source_col="close"` | `mom_{period}` |
| `CmoMom` | `momentum/cmo` | `period=14`, `source_col="close"` | `cmo_{period}` |

### Oscillators (`signalflow.ta.momentum.oscillators`)

| Class | `@sf_component` | Parameters | Output |
|-------|----------------|------------|--------|
| `StochMom` | `momentum/stoch` | `k_period=14`, `d_period=3` | `stoch_k_{k_period}`, `stoch_d_{k_period}` |
| `StochRsiMom` | `momentum/stoch_rsi` | `period=14`, `k_period=3`, `d_period=3` | `stoch_rsi_k_{period}`, `stoch_rsi_d_{period}` |
| `WillrMom` | `momentum/willr` | `period=14` | `willr_{period}` |
| `CciMom` | `momentum/cci` | `period=20` | `cci_{period}` |
| `UoMom` | `momentum/uo` | `s=7`, `m=14`, `l=28` | `uo` |
| `AoMom` | `momentum/ao` | `s=5`, `l=34` | `ao` |

### MACD Family (`signalflow.ta.momentum.macd`)

| Class | `@sf_component` | Parameters | Output |
|-------|----------------|------------|--------|
| `MacdMom` | `momentum/macd` | `fast=12`, `slow=26`, `signal=9` | `macd`, `macd_signal`, `macd_hist` |
| `PpoMom` | `momentum/ppo` | `fast=12`, `slow=26`, `signal=9` | `ppo`, `ppo_signal`, `ppo_hist` |
| `TsiMom` | `momentum/tsi` | `long=25`, `short=13`, `signal=13` | `tsi`, `tsi_signal` |
| `TrixMom` | `momentum/trix` | `period=15`, `signal=9` | `trix`, `trix_signal` |

### Kinematics (`signalflow.ta.momentum.kinematics`)

| Class | `@sf_component` | Parameters | Output |
|-------|----------------|------------|--------|
| `AccelerationMom` | `momentum/acceleration` | `period=14` | `acceleration_{period}` |
| `JerkMom` | `momentum/jerk` | `period=14` | `jerk_{period}` |
| `AngularMomentumMom` | `momentum/angular_momentum` | `period=14` | `angular_momentum_{period}` |
| `TorqueMom` | `momentum/torque` | `period=14` | `torque_{period}` |

---

## Overlap

`signalflow.ta.overlap` — 26 indicators

### Smoothers (`signalflow.ta.overlap.smoothers`)

| Class | `@sf_component` | Parameters | Output |
|-------|----------------|------------|--------|
| `SmaSmooth` | `overlap/sma` | `period=20`, `source_col="close"` | `sma_{period}` |
| `EmaSmooth` | `overlap/ema` | `period=20`, `source_col="close"` | `ema_{period}` |
| `WmaSmooth` | `overlap/wma` | `period=20`, `source_col="close"` | `wma_{period}` |
| `RmaSmooth` | `overlap/rma` | `period=20`, `source_col="close"` | `rma_{period}` |
| `DemaSmooth` | `overlap/dema` | `period=20`, `source_col="close"` | `dema_{period}` |
| `TemaSmooth` | `overlap/tema` | `period=20`, `source_col="close"` | `tema_{period}` |
| `HmaSmooth` | `overlap/hma` | `period=20`, `source_col="close"` | `hma_{period}` |
| `TrimaSmooth` | `overlap/trima` | `period=20`, `source_col="close"` | `trima_{period}` |
| `SwmaSmooth` | `overlap/swma` | `period=20`, `source_col="close"` | `swma_{period}` |
| `SsfSmooth` | `overlap/ssf` | `period=20`, `source_col="close"` | `ssf_{period}` |
| `FftSmooth` | `overlap/fft` | `period=20`, `source_col="close"` | `fft_{period}` |

### Adaptive (`signalflow.ta.overlap.adaptive`)

| Class | `@sf_component` | Parameters | Output |
|-------|----------------|------------|--------|
| `KamaSmooth` | `overlap/kama` | `period=10`, `fast=2`, `slow=30` | `kama_{period}` |
| `AlmaSmooth` | `overlap/alma` | `period=20`, `sigma=6.0`, `offset=0.85` | `alma_{period}` |
| `JmaSmooth` | `overlap/jma` | `period=7`, `phase=50`, `power=2` | `jma_{period}` |
| `VidyaSmooth` | `overlap/vidya` | `period=20`, `cmo_period=9` | `vidya_{period}` |
| `T3Smooth` | `overlap/t3` | `period=5`, `v_factor=0.7` | `t3_{period}` |
| `ZlmaSmooth` | `overlap/zlma` | `period=20` | `zlma_{period}` |
| `McGinleySmooth` | `overlap/mcginley` | `period=20` | `mcginley_{period}` |
| `FramaSmooth` | `overlap/frama` | `period=20` | `frama_{period}` |

### Price Transforms (`signalflow.ta.overlap.price`)

| Class | `@sf_component` | Output |
|-------|----------------|--------|
| `Hl2Price` | `overlap/hl2` | `hl2` |
| `Hlc3Price` | `overlap/hlc3` | `hlc3` |
| `Ohlc4Price` | `overlap/ohlc4` | `ohlc4` |
| `WcpPrice` | `overlap/wcp` | `wcp` |
| `MidpointPrice` | `overlap/midpoint` | `midpoint` |
| `MidpricePrice` | `overlap/midprice` | `midprice` |
| `TypicalPrice` | `overlap/typical` | `typical` |

---

## Volatility

`signalflow.ta.volatility` — 18+ indicators

### Range (`signalflow.ta.volatility.range`)

| Class | `@sf_component` | Parameters | Output |
|-------|----------------|------------|--------|
| `TrueRangeVol` | `volatility/true_range` | — | `true_range` |
| `AtrVol` | `volatility/atr` | `period=14` | `atr_{period}` |
| `NatrVol` | `volatility/natr` | `period=14` | `natr_{period}` |

### Bands (`signalflow.ta.volatility.bands`)

| Class | `@sf_component` | Parameters | Output |
|-------|----------------|------------|--------|
| `BollingerVol` | `volatility/bollinger` | `period=20`, `num_std=2.0` | `bb_upper`, `bb_middle`, `bb_lower`, `bb_width`, `bb_pct` |
| `KeltnerVol` | `volatility/keltner` | `period=20`, `atr_period=10`, `multiplier=1.5` | `kc_upper`, `kc_middle`, `kc_lower` |
| `DonchianVol` | `volatility/donchian` | `period=20` | `dc_upper`, `dc_middle`, `dc_lower` |
| `AccBandsVol` | `volatility/acc_bands` | `period=20` | `acc_upper`, `acc_middle`, `acc_lower` |

### Measures (`signalflow.ta.volatility.measures`)

| Class | `@sf_component` | Parameters | Output |
|-------|----------------|------------|--------|
| `MassIndexVol` | `volatility/mass_index` | `fast=9`, `slow=25` | `mass_index` |
| `UlcerIndexVol` | `volatility/ulcer_index` | `period=14` | `ulcer_index_{period}` |
| `RviVol` | `volatility/rvi` | `period=14` | `rvi_{period}` |
| `GapVol` | `volatility/gap` | `period=14` | `gap_vol_{period}` |

### Energy — Physics-Based (`signalflow.ta.volatility.energy`)

| Class | `@sf_component` | Parameters | Output |
|-------|----------------|------------|--------|
| `KineticEnergyVol` | `volatility/kinetic_energy` | `period=20` | `kinetic_energy_{period}` |
| `PotentialEnergyVol` | `volatility/potential_energy` | `period=20` | `potential_energy_{period}` |
| `TotalEnergyVol` | `volatility/total_energy` | `period=20` | `total_energy_{period}` |
| `EnergyFlowVol` | `volatility/energy_flow` | `period=20` | `energy_flow_{period}` |
| `ElasticStrainVol` | `volatility/elastic_strain` | `period=20` | `elastic_strain_{period}` |
| `TemperatureVol` | `volatility/temperature` | `period=20` | `temperature_{period}` |
| `HeatCapacityVol` | `volatility/heat_capacity` | `period=20` | `heat_capacity_{period}` |
| `FreeEnergyVol` | `volatility/free_energy` | `period=20` | `free_energy_{period}` |

---

## Volume

`signalflow.ta.volume` — 16 indicators

### Cumulative (`signalflow.ta.volume.cumulative`)

| Class | `@sf_component` | Output |
|-------|----------------|--------|
| `ObvVolume` | `volume/obv` | `obv` |
| `AdVolume` | `volume/ad` | `ad` |
| `PvtVolume` | `volume/pvt` | `pvt` |
| `NviVolume` | `volume/nvi` | `nvi` |
| `PviVolume` | `volume/pvi` | `pvi` |

### Oscillators (`signalflow.ta.volume.oscillators`)

| Class | `@sf_component` | Parameters | Output |
|-------|----------------|------------|--------|
| `MfiVolume` | `volume/mfi` | `period=14` | `mfi_{period}` |
| `CmfVolume` | `volume/cmf` | `period=20` | `cmf_{period}` |
| `EfiVolume` | `volume/efi` | `period=13` | `efi_{period}` |
| `EomVolume` | `volume/eom` | `period=14` | `eom_{period}` |
| `KvoVolume` | `volume/kvo` | `fast=34`, `slow=55`, `signal=13` | `kvo`, `kvo_signal` |

### Dynamics — Physics-Based (`signalflow.ta.volume.dynamics`)

| Class | `@sf_component` | Parameters | Output |
|-------|----------------|------------|--------|
| `MarketForceVolume` | `volume/market_force` | `period=14` | `market_force_{period}` |
| `ImpulseVolume` | `volume/impulse` | `period=14` | `impulse_{period}` |
| `MarketMomentumVolume` | `volume/market_momentum` | `period=14` | `market_momentum_{period}` |
| `MarketPowerVolume` | `volume/market_power` | `period=14` | `market_power_{period}` |
| `MarketCapacitanceVolume` | `volume/market_capacitance` | `period=14` | `market_capacitance_{period}` |
| `GravitationalPullVolume` | `volume/gravitational_pull` | `period=14` | `gravitational_pull_{period}` |

---

## Trend

`signalflow.ta.trend` — 22 indicators

### Strength (`signalflow.ta.trend.strength`)

| Class | `@sf_component` | Parameters | Output |
|-------|----------------|------------|--------|
| `AdxTrend` | `trend/adx` | `period=14` | `adx_{period}`, `di_plus`, `di_minus` |
| `AroonTrend` | `trend/aroon` | `period=25` | `aroon_up`, `aroon_down`, `aroon_osc` |
| `VortexTrend` | `trend/vortex` | `period=14` | `vi_plus`, `vi_minus` |
| `VhfTrend` | `trend/vhf` | `period=28` | `vhf_{period}` |
| `ChopTrend` | `trend/chop` | `period=14` | `chop_{period}` |
| `ViscosityTrend` | `trend/viscosity` | `period=20` | `viscosity_{period}` |
| `ReynoldsTrend` | `trend/reynolds` | `period=20` | `reynolds_{period}` |
| `RotationalInertiaTrend` | `trend/rotational_inertia` | `period=20` | `rotational_inertia_{period}` |
| `MarketImpedanceTrend` | `trend/market_impedance` | `period=20` | `market_impedance_{period}` |
| `RCTimeConstantTrend` | `trend/rc_time_constant` | `period=20` | `rc_time_constant_{period}` |
| `SNRTrend` | `trend/snr` | `period=20` | `snr_{period}` |
| `OrderParameterTrend` | `trend/order_parameter` | `period=20` | `order_parameter_{period}` |
| `SusceptibilityTrend` | `trend/susceptibility` | `period=20` | `susceptibility_{period}` |

### Stops (`signalflow.ta.trend.stops`)

| Class | `@sf_component` | Parameters | Output |
|-------|----------------|------------|--------|
| `PsarTrend` | `trend/psar` | `af=0.02`, `max_af=0.2` | `psar` |
| `SupertrendTrend` | `trend/supertrend` | `period=10`, `multiplier=3.0` | `supertrend`, `supertrend_dir` |
| `ChandelierTrend` | `trend/chandelier` | `period=22`, `multiplier=3.0` | `chandelier_long`, `chandelier_short` |
| `HiloTrend` | `trend/hilo` | `period=13` | `hilo` |
| `CkspTrend` | `trend/cksp` | `p=10`, `q=3`, `x=1` | `cksp_stop_long`, `cksp_stop_short` |

### Detection (`signalflow.ta.trend.detection`)

| Class | `@sf_component` | Parameters | Output |
|-------|----------------|------------|--------|
| `IchimokuTrend` | `trend/ichimoku` | `tenkan=9`, `kijun=26`, `senkou=52` | `tenkan_sen`, `kijun_sen`, `senkou_a`, `senkou_b`, `chikou` |
| `DpoTrend` | `trend/dpo` | `period=20` | `dpo_{period}` |
| `QstickTrend` | `trend/qstick` | `period=14` | `qstick_{period}` |
| `TtmTrend` | `trend/ttm` | `period=20` | `ttm_squeeze` |

---

## Statistics

`signalflow.ta.stat` — 73 indicators across 11 submodules

### Dispersion

| Class | `@sf_component` | Description |
|-------|----------------|-------------|
| `VarianceStat` | `stat/variance` | Rolling variance |
| `StdStat` | `stat/std` | Rolling standard deviation |
| `MadStat` | `stat/mad` | Median absolute deviation |
| `ZscoreStat` | `stat/zscore` | Rolling z-score |
| `CvStat` | `stat/cv` | Coefficient of variation |
| `RangeStat` | `stat/range` | Rolling range (max - min) |
| `IqrStat` | `stat/iqr` | Interquartile range |
| `AadStat` | `stat/aad` | Average absolute deviation |
| `RobustZscoreStat` | `stat/robust_zscore` | Median-based z-score |

### Distribution

| Class | `@sf_component` | Description |
|-------|----------------|-------------|
| `MedianStat` | `stat/median` | Rolling median |
| `QuantileStat` | `stat/quantile` | Rolling quantile |
| `PctRankStat` | `stat/pct_rank` | Percentile rank |
| `MinMaxStat` | `stat/minmax` | Min-max normalization |
| `SkewStat` | `stat/skew` | Rolling skewness |
| `KurtosisStat` | `stat/kurtosis` | Rolling kurtosis |
| `EntropyStat` | `stat/entropy` | Shannon entropy |
| `JarqueBeraStat` | `stat/jarque_bera` | Normality test statistic |
| `ModeDistanceStat` | `stat/mode_distance` | Distance from mode |
| `AboveMeanRatioStat` | `stat/above_mean_ratio` | Proportion above mean |
| `EntropyRateStat` | `stat/entropy_rate` | Entropy change rate |

### Memory & Diffusion

| Class | `@sf_component` | Description |
|-------|----------------|-------------|
| `HurstStat` | `stat/hurst` | Hurst exponent (mean-reversion vs trend) |
| `AutocorrStat` | `stat/autocorr` | Autocorrelation |
| `VarianceRatioStat` | `stat/variance_ratio` | Lo-MacKinlay variance ratio |
| `DiffusionCoeffStat` | `stat/diffusion_coeff` | Diffusion coefficient |
| `AnomalousDiffusionStat` | `stat/anomalous_diffusion` | Anomalous diffusion exponent |
| `MsdRatioStat` | `stat/msd_ratio` | Mean squared displacement ratio |
| `SpringConstantStat` | `stat/spring_constant` | Harmonic oscillator spring constant |
| `DampingRatioStat` | `stat/damping_ratio` | Oscillation damping |
| `NaturalFrequencyStat` | `stat/natural_frequency` | Natural oscillation frequency |
| `PlasticStrainStat` | `stat/plastic_strain` | Irreversible deformation |
| `EscapeVelocityStat` | `stat/escape_velocity` | Breakout threshold |
| `CorrelationLengthStat` | `stat/correlation_length` | Spatial correlation decay |

### Cycle Analysis

| Class | `@sf_component` | Description |
|-------|----------------|-------------|
| `InstAmplitudeStat` | `stat/inst_amplitude` | Instantaneous amplitude via Hilbert |
| `InstPhaseStat` | `stat/inst_phase` | Instantaneous phase |
| `InstFrequencyStat` | `stat/inst_frequency` | Instantaneous frequency |
| `PhaseAccelerationStat` | `stat/phase_acceleration` | Phase acceleration |
| `ConstructiveInterferenceStat` | `stat/constructive_interference` | Wave interference measure |
| `BeatFrequencyStat` | `stat/beat_frequency` | Beat frequency from phase |
| `StandingWaveRatioStat` | `stat/standing_wave_ratio` | Standing wave ratio |
| `SpectralCentroidStat` | `stat/spectral_centroid` | Frequency center of mass |
| `SpectralEntropyStat` | `stat/spectral_entropy` | Spectral entropy |

### Complexity

| Class | `@sf_component` | Description |
|-------|----------------|-------------|
| `PermutationEntropyStat` | `stat/permutation_entropy` | Bandt-Pompe permutation entropy |
| `SampleEntropyStat` | `stat/sample_entropy` | Sample entropy |
| `LempelZivStat` | `stat/lempel_ziv` | Lempel-Ziv complexity |
| `FisherInformationStat` | `stat/fisher_information` | Fisher information |
| `DfaExponentStat` | `stat/dfa_exponent` | Detrended fluctuation analysis |

### Information Theory

| Class | `@sf_component` | Description |
|-------|----------------|-------------|
| `KLDivergenceStat` | `stat/kl_divergence` | Kullback-Leibler divergence |
| `JSDivergenceStat` | `stat/js_divergence` | Jensen-Shannon divergence |
| `RenyiEntropyStat` | `stat/renyi_entropy` | Renyi entropy |
| `AutoMutualInfoStat` | `stat/auto_mutual_info` | Auto mutual information |
| `RelativeInfoGainStat` | `stat/relative_info_gain` | Relative information gain |

### Regression

| Class | `@sf_component` | Description |
|-------|----------------|-------------|
| `CorrelationStat` | `stat/correlation` | Rolling Pearson correlation |
| `BetaStat` | `stat/beta` | Rolling beta (vs benchmark) |
| `RSquaredStat` | `stat/r_squared` | R-squared |
| `LinRegSlopeStat` | `stat/linreg_slope` | Linear regression slope |
| `LinRegInterceptStat` | `stat/linreg_intercept` | Linear regression intercept |
| `LinRegResidualStat` | `stat/linreg_residual` | Linear regression residual |

### Realized Volatility

| Class | `@sf_component` | Description |
|-------|----------------|-------------|
| `RealizedVolStat` | `stat/realized_vol` | Realized (close-to-close) volatility |
| `ParkinsonVolStat` | `stat/parkinson_vol` | Parkinson range-based volatility |
| `GarmanKlassVolStat` | `stat/garman_klass_vol` | Garman-Klass volatility |
| `RogersSatchellVolStat` | `stat/rogers_satchell_vol` | Rogers-Satchell volatility |
| `YangZhangVolStat` | `stat/yang_zhang_vol` | Yang-Zhang volatility |

### DSP / Signal Processing

| Class | `@sf_component` | Description |
|-------|----------------|-------------|
| `SpectralFluxStat` | `stat/spectral_flux` | Spectral flux |
| `ZeroCrossingRateStat` | `stat/zero_crossing_rate` | Zero crossing rate |
| `SpectralRolloffStat` | `stat/spectral_rolloff` | Spectral rolloff frequency |
| `SpectralFlatnessStat` | `stat/spectral_flatness` | Wiener entropy |
| `PowerCepstrumStat` | `stat/power_cepstrum` | Power cepstrum |

### Control Theory

| Class | `@sf_component` | Description |
|-------|----------------|-------------|
| `KalmanInnovationStat` | `stat/kalman_innovation` | Kalman filter innovation |
| `ARCoefficientStat` | `stat/ar_coefficient` | Autoregressive coefficient |
| `LyapunovExponentStat` | `stat/lyapunov_exponent` | Largest Lyapunov exponent |
| `PIDErrorStat` | `stat/pid_error` | PID controller error |
| `PredictionErrorDecompositionStat` | `stat/prediction_error_decomposition` | Prediction error decomposition |

### Cross-Sectional

| Class | `@sf_component` | Description |
|-------|----------------|-------------|
| `CrossSectionalStat` | `stat/cross_sectional` | Cross-sectional aggregation |

---

## Performance

`signalflow.ta.performance` — 2 indicators

| Class | `@sf_component` | Parameters | Output |
|-------|----------------|------------|--------|
| `LogReturn` | `performance/log_return` | `period=1` | `log_return_{period}` |
| `PctReturn` | `performance/pct_return` | `period=1` | `pct_return_{period}` |

---

## Divergence

`signalflow.ta.divergence` — 2 detectors

| Class | `@sf_component` | Description |
|-------|----------------|-------------|
| `RsiDivergence` | `divergence/rsi` | RSI-based divergence detector (regular & hidden) |
| `MacdDivergence` | `divergence/macd` | MACD histogram divergence detector |

---

## Common Parameters

All indicators inherit these from `Feature`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `normalized` | `bool` | `False` | Enable normalization (z-score for unbounded, linear for bounded) |
| `norm_period` | `int \| None` | `None` | Custom normalization window (auto-computed if None) |
| `group_col` | `str` | `"pair"` | Column for per-asset grouping |
| `ts_col` | `str` | `"timestamp"` | Timestamp column |

Each indicator also has a `warmup` property indicating the minimum bars needed for stable output.
