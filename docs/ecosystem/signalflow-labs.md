---
title: signalflow-labs
description: Torch backend, RL strategy, and parked neural stack for SignalFlow
---

# signalflow-labs - Torch & RL

**signalflow-labs** extends SignalFlow with PyTorch. Two components plug into
the core `Flow` contract today; a larger neural time-series stack is kept as
importable building blocks until it is re-adapted to the V5 model contract.

---

## Installation

```bash
pip install signalflow-labs            # torch + lightning
pip install "signalflow-labs[rl]"      # + stable-baselines3, gymnasium
```

Requires `signalflow-trading >= 0.8.5`, `torch >= 2.2`, `lightning >= 2.5`.
Installing the package registers `strategy: rl` via the `signalflow.components`
entry point.

---

## TorchMLPBackend

A small MLP classifier exposing the sklearn estimator surface, usable wherever
`ForecastModel` takes a `backend`. Training stays inside the core's embargoed
walk-forward, so out-of-fold predictions, WoE encoding, and the flow round-trip
are unchanged.

```python
import signalflow as sf
import signalflow.labs as labs

model = sf.ForecastModel(
    backend=labs.TorchMLPBackend(hidden_sizes=(64, 32), epochs=50),
    target=sf.FixedHorizon(bars=12),
    features=sf.FeaturePipeline(sf.SMA(10), sf.SMA(20), sf.SMA(50)),
)
model.fit(ds)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `hidden_sizes` | `(64, 32)` | hidden layer widths |
| `epochs` | `50` | full passes over the WoE matrix |
| `lr` / `weight_decay` | `1e-3` / `0.0` | Adam settings |
| `batch_size` | `256` | minibatch size |
| `seed` | `0` | torch seed for reproducible fits |

---

## RLStrategy and make_env

`make_env(flow, ds)` returns a gymnasium `Env` over the Engine replay. Each step
is one bar; the observation is `Observation.to_vector()` - the same vector the
live loop hands to `RLStrategy.decide` - and the reward is the log-change in
equity. The action space is `Discrete(3)`: hold, open one position (`size_pct`
of equity) on the strongest RISE pair not yet held, close all.

```python
from stable_baselines3 import PPO

base = sf.Flow(name="rl", detectors=[sf.SmaCrossDetector()])
policy = PPO("MlpPolicy", labs.make_env(base, ds)).learn(10_000)

flow = base.replace(strategy=labs.RLStrategy(model=policy, size_pct=0.1))
run = flow.backtest(ds, capital=50_000)
```

### Deploy is data

`flow.save(path, model_dir=...)` persists the policy under
`<model_dir>/strategy/` (SB3 `policy.zip` when the object has `save`/`load`,
`policy.pkl` via cloudpickle otherwise) and writes the strategy config as:

```yaml
strategy:
  name: rl
  params:
    size_pct: 0.1
    policy_uri: file://flows/models/strategy/policy.zip
    policy_class: stable_baselines3.ppo.ppo:PPO
    schema_version: 1
```

`sf.Flow.load` rebuilds the strategy from that block and the backtest is
byte-identical. `schema_version` is the `OBSERVATION_SCHEMA_VERSION` the policy
was trained against; a mismatch at decision time raises `SchemaVersionError`.

---

## Parked neural stack

Importable, unit-tested where they are plain modules, but not registered and not
wired into `Flow`:

| Module | Contents |
|--------|----------|
| `signalflow.labs.encoder` | 16 encoders - LSTM, GRU, TCN, Transformer, PatchTST, TSMixer, InceptionTime, ResNet1d, XceptionTime, Conv1d, XCM, gMLP, OmniScaleCNN, ConvTran, iTransformer, Mamba |
| `signalflow.labs.head` | 7 heads - Linear, MLP, Residual, Attention, OrdinalRegression, Distribution, ClassificationWithConfidence |
| `signalflow.labs.loss` | FocalLoss, DiceLoss, LDAMLoss, SymmetricCrossEntropyLoss |
| `signalflow.labs.data`, `.model`, `.validator` | `TimeSeriesPreprocessor`, `SignalWindowDataset`, `SignalDataModule`, `TemporalClassificator`, `TemporalValidator` - pre-V5, integration tests skipped |

All encoders share the interface `forward(x: [batch, seq_len, features]) ->
[batch, embedding]`; heads take `[batch, embedding] -> [batch, num_classes]`.

---

## Links

- [:material-github: GitHub Repository](https://github.com/pathway2nothing/signalflow-labs)
- [:material-package: PyPI Package](https://pypi.org/project/signalflow-labs/)
