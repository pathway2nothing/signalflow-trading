# Roadmap: Розмітка фінансових часових рядів

## Поточний стан SignalFlow

SignalFlow вже має розвинену інфраструктуру розмітки фінансових рядів:

### Labelers (розмітники)

| Компонент | Файл | Опис |
|-----------|------|------|
| `Labeler` (ABC) | `target/base.py` | Базовий клас: `compute()`, `compute_group()`, валідація, signal masking |
| `FixedHorizonLabeler` | `target/fixed_horizon_labeler.py` | `label = sign(close[t+h] - close[t])` |
| `TripleBarrierLabeler` | `target/triple_barrier.py` | De Prado, волатильно-адаптивні бар'єри, Numba-accelerated |
| `StaticTripleBarrierLabeler` | `target/static_triple_barrier.py` | De Prado, фіксовані % бар'єри |

### Event Detectors (детектори подій)

| Компонент | Файл | Алгоритм |
|-----------|------|----------|
| `EventDetectorBase` (ABC) | `target/event_detector_base.py` | Базовий клас, маскування міток біля подій |
| `GlobalEventDetector` | `target/global_event_detector.py` | Agreement-based: ≥ 80% пар рухаються в одному напрямку |
| `ZScoreEventDetector` | `target/zscore_event_detector.py` | Z-score агрегованого доходу: \|z\| > threshold |
| `CusumEventDetector` | `target/cusum_event_detector.py` | CUSUM крос-пар з дрейфом та скиданням |

### Meta-Labeling (SignalValidator)

| Компонент | Файл | Опис |
|-----------|------|------|
| `SignalValidator` (ABC) | `validator/base.py` | "In De Prado's terminology - this is a meta-labeler" |
| `SklearnSignalValidator` | `validator/sklearn_validator.py` | LightGBM, XGBoost, RF, LogReg, SVM, Auto-select |
| `VotingMode.META_LABELING` | `aggregation.py` | combined = detector_prob x validator_prob |

Архітектура мета-лейблінгу в SF:
```
Detector -> Signal Side (RISE/FALL/NONE)     [primary model]
       |
Labeler(mask_to_signals=True) -> Ground truth
       |
SignalValidator.fit(features, labels)        [meta-labeler training]
       |
SignalValidator.predict_proba() -> P(correct) [meta-label prediction]
       |
VotingMode.META_LABELING -> weighted signal   [aggregation]
```

### Аналіз інформативності

| Компонент | Файл | Опис |
|-----------|------|------|
| `MultiTargetGenerator` | `target/multi_target_generator.py` | Мульти-горизонт/мульти-тип генерація міток |
| `FeatureInformativenessAnalyzer` | `feature/informativeness.py` | MI-based оцінка інформативності фічей |
| Mutual Information | `feature/mutual_information.py` | Ентропія (Shannon), MI, NMI |

---

## Таксономія методів розмітки De Prado

### 1. Triple Barrier Method ✅

**Джерело**: AFML, Chapter 3

Три бар'єри -- верхній (profit-taking), нижній (stop-loss), вертикальний (час).

```
PT = S_0 x exp(sigma x profit_multiplier)      -- верхній
SL = S_0 x exp(-sigma x stop_loss_multiplier)   -- нижній
t_1 = t_0 + lookforward_window                  -- вертикальний
```

Мітка визначається тим, який бар'єр торкнуто першим: RISE / FALL / NONE.

**Параметри**: `vol_window` (60), `lookforward_window` (1440), `profit_multiplier` (1.0), `stop_loss_multiplier` (1.0)

| Переваги | Недоліки |
|----------|----------|
| Адаптивність до волатильності | Label overlap/concurrency |
| Реалістичність (TP/SL) | Чутливість до параметрів |
| Path-aware | Потребує OHLC дані |

### 2. Meta-Labeling ✅

**Джерело**: AFML, Chapter 3.6

Двошаровий підхід: первинна модель дає напрямок, вторинна -- рішення торгувати чи ні.

Реалізовано через `SignalValidator` + `VotingMode.META_LABELING` (див. вище).

**Емпіричні результати (Hudson & Thames)**:
- Sharpe Ratio: 1.18 -> 3.08 (+161%)
- Max Drawdown: -6.2% -> -4.1%
- Cumulative Return: 5.0% -> 11.8%

### 3. CUSUM Filter (частково ✅)

**Джерело**: AFML, Chapter 2.5.2

Кумулятивний фільтр для виявлення структурних зрушень (regime shifts).

```
S+ = max(0, S+ + (x_t - mu - drift))
S- = max(0, S- + (-x_t + mu - drift))
Event if S+ > h OR S- > h -> Reset
```

В SignalFlow реалізовано як `CusumEventDetector` -- крос-пар агрегація для глобальних подій.
**НЕ** реалізовано як фільтр для per-pair семплінгу (De Prado's original use case).

### 4. Fixed-Time Horizon ✅

**Джерело**: AFML, Chapter 3.1

`label = sign(close[t+h] - close[t])` -- найпростіший підхід.

**Проблеми** (чому De Prado критикує):
1. Не враховує волатильність -- однакові пороги для різних режимів ринку
2. Ігнорує шлях ціни (path) -- дивиться лише на кінцеві точки
3. Label instability -- мітки "стрибають" при малих флуктуаціях
4. Дисбаланс класів -- клас "0" (нема руху) домінує
5. Не реалістичний -- трейдери використовують TP/SL, а не фіксований горизонт

### 5. Trend Scanning ❌

**Джерело**: ML for Asset Managers, Chapter 5

Мітка на основі статистичної значущості тренду (t-statistic лінійної регресії).

```
Для кожного t, для кожного L in {L_min...L_max}:
  fit: Price(t+i) = alpha + beta x i + epsilon
  t_stat(t,L) = beta / SE(beta)

L* = argmax_L |t_stat(t,L)|
Label = sign(t_stat(t, L*)) if |t_stat| > critical_value
```

| Переваги | Недоліки |
|----------|----------|
| Статистично обгрунтований | O(n x m) обчислення |
| Адаптивний горизонт (L*) | Не реалістичний (без TP/SL) |
| Balanced classes | Чутливий до outliers |
| t-stat має чіткий статистичний сенс | Assumes linear trends |

### 6. Fractional Differentiation (preprocessing, НЕ labeling)

**Джерело**: AFML, Chapter 5

Це **НЕ** метод розмітки, а feature engineering / preprocessing: трансформує ціновий ряд для стаціонарності зі збереженням пам'яті. Порядок d in (0, 1) контролює trade-off між стаціонарністю (d -> 1) і пам'яттю (d -> 0). Не входить в план імплементації labeling.

### 7. Alternative Bars ❌

**Джерело**: AFML, Chapter 2

Замість time bars -- bars на основі ринкової активності.

- **Volume Bars**: Кожні N одиниць об'єму
- **Dollar Bars**: Кожні N доларів обороту
- **Information Bars**: На основі ентропії / buy-sell imbalance

Позиція De Prado: "Time bars should be avoided" -- вони мають сезонність і нерівномірну інформативність.

---

## Методи поза De Prado (нові дослідження 2023-2025)

### 8. Adaptive Event-Driven Labeling (AEDL, 2025)

Мульти-масштабний каузальний фреймворк з мета-навчанням.
- Multi-scale temporal analysis (5 резолюцій одночасно)
- Granger causality + Transfer entropy для фільтрації spurious correlations
- Meta-learning (MAML) для адаптації параметрів під конкретний актив/режим

**Результати**: Sharpe 0.48 (vs ~0 для baseline), 16 активів, 25 років даних.

Paper: "Adaptive Event-Driven Labeling: Multi-Scale Causal Framework with Meta-Learning" (MDPI Applied Sciences, 2025)

### 9. Continuous Trend Labeling (CTL)

Безперервні мітки замість бінарних -- зберігають більше інформації про величину руху.
Зменшує необхідність агресивного denoising.

Paper: "A Labeling Method for Financial Time Series Prediction Based on Trends" (2020)

### 10. N-Period Volatility Labeling (2024)

Мітка на основі волатильності ціни за N періодів.
- Балансує пропорції класів (29-денні вікна, 9% бар'єри)
- Стабільні довгострокові торгові системи

Paper: "Improving ML Stock Trading: N-Period Volatility Labeling and Instance Selection" (2024)

### 11. Self-Supervised Denoised Labels (2021)

Denoising autoencoder для фільтрації шуму в мітках.
- Treats label generation як pretext task
- Покращує downstream classification для малих і великих датасетів
- Diffusion model denoisers (2024) -- ще кращі результати

Paper: ArXiv 2112.10139

### 12. Change-Point Detection (PELT, 2025)

Автоматичне виявлення множинних точок зміни (Pruned Exact Linear Time).
- Ефективніший за CUSUM для множинних changepoints
- Не потребує заданого threshold
- Автоматично визначає кількість і позиції зрушень

Paper: "Change-Point Detection in Financial Time Series Using the PELT Algorithm" (ACM/ICCSAI, 2025)

### 13. Entropy-Based Methods

- **Shannon entropy** для market uncertainty -- визначення інформативності окремих барів
- **Transfer entropy** для direction of information flow між активами
- **Mutual information** для feature-target relationship (вже є в SF ✅)
- **Block entropy** для ідентифікації торгових поведінок

Paper: "Financial Information Theory" (ArXiv:2511.16339, 2025)

---

## Gap Analysis

| Метод | Статус в SF | Пріоритет |
|-------|-------------|-----------|
| Triple Barrier (dynamic) | ✅ | -- |
| Triple Barrier (static) | ✅ | -- |
| Fixed Horizon | ✅ | -- |
| CUSUM (global events) | ✅ | -- |
| Z-Score detector | ✅ | -- |
| Agreement detector | ✅ | -- |
| Multi-target generator | ✅ | -- |
| MI-based informativeness | ✅ | -- |
| Meta-Labeling | ✅ SignalValidator | -- |
| **Trend Scanning** | ❌ | Високий |
| **CUSUM per-pair filter** | ❌ | Середній |
| **Volatility Regime Labeler** | ❌ | Середній |
| Continuous Trend Labels | ❌ | Низький |
| Alternative Bars | ❌ | Низький (інша абстракція) |
| AEDL (мета-навчання) | ❌ | Дослідницький |

---

## План імплементації

### Phase 1: Trend Scanning Labeler

**Файл**: `src/signalflow/target/trend_scanning.py`
**Клас**: `TrendScanningLabeler(Labeler)` з `@sf_component(name="trend_scanning")`

Алгоритм (Numba-accelerated):
- Для кожного бару: OLS лінійна регресія за вікнами L in [min_L, max_L, step]
- `Price(t+i) = alpha + beta x i + epsilon`
- `t_stat = beta / SE(beta)`, де `SE(beta) = sqrt(MSE / Sxx)`
- Оптимальне `L* = argmax_L |t_stat(t,L)|`
- Label: RISE if t_stat > cv, FALL if t_stat < -cv, NONE otherwise

Детальний OLS:
```
x = [0, 1, ..., L-1]
Sxx = L(L-1)(2L-1)/6 - L * x_mean^2
Sxy = sum(x*y) - L * x_mean * y_mean
b = Sxy / Sxx
a = y_mean - b * x_mean
RSS = sum((y_i - a - b*x_i)^2)
MSE = RSS / (L - 2)
SE(b) = sqrt(MSE / Sxx)
t = b / SE(b)
```

**Параметри**:
| Параметр | Default | Опис |
|----------|---------|------|
| `price_col` | `"close"` | Колонка ціни |
| `min_lookforward` | `5` | Мін. вікно (>= 3 для df) |
| `max_lookforward` | `60` | Макс. вікно |
| `step` | `5` | Крок між вікнами |
| `critical_value` | `1.96` | Поріг t-statistic (95% CI) |

**Meta-columns** (при `include_meta=True`): `t_stat`, `best_window`, `slope`
**Performance**: O(n x W x L_max) з Numba `@njit(parallel=True)` -- prange по барах
**Тести**: `tests/test_trend_scanning.py` (~12 тестів: validation, correctness, multi-pair, meta)

---

### Phase 2: Per-Pair CUSUM Filter

**Файл**: `src/signalflow/target/cusum_filter.py`
**Клас**: `CusumFilter` (standalone `@dataclass`, як `MultiTargetGenerator`)

Відмінність від існуючого `CusumEventDetector`:
- `CusumEventDetector`: агрегує returns крос-пар, потім CUSUM на агрегаті (глобальні події)
- `CusumFilter`: CUSUM per-pair на індивідуальних return series (event-driven sampling)

Алгоритм (Numba):
```
returns = log(price[t] / price[t-1])

S_pos = 0, S_neg = 0
For each bar t:
    S_pos = max(0, S_pos + return[t])
    S_neg = min(0, S_neg + return[t])

    if S_pos > threshold[t]:
        Mark event, reset S_pos=0, S_neg=0
    elif S_neg < -threshold[t]:
        Mark event, reset S_pos=0, S_neg=0
```

Dynamic threshold: `threshold[t] = rolling_std(returns, vol_window) x threshold_multiplier`

**Параметри**:
| Параметр | Default | Опис |
|----------|---------|------|
| `threshold` | `0.02` | Статичний поріг |
| `dynamic_threshold` | `False` | Використовувати rolling vol |
| `vol_window` | `100` | Вікно rolling std |
| `threshold_multiplier` | `2.0` | Множник для динамічного порогу |
| `out_col` | `"cusum_event"` | Ім'я boolean колонки |

**Методи**:
- `filter(df) -> DataFrame` -- додає boolean column `cusum_event`
- `filter_timestamps(df) -> DataFrame` -- повертає тільки (pair, timestamp) подій

**Performance**: Sequential per-pair (CUSUM stateful), Polars group_by для паралелізму між парами
**Тести**: `tests/test_cusum_filter.py` (~12 тестів: static/dynamic, reset, multi-pair, monotonicity)

---

### Phase 3: Volatility Regime Labeler

**Файл**: `src/signalflow/target/volatility_labeler.py`
**Клас**: `VolatilityRegimeLabeler(Labeler)` з `@sf_component(name="volatility_regime")`

Мітка на основі реалізованої волатильності за forward window -- визначає режим ринку (HIGH/MED/LOW).

Алгоритм:
```
Для кожного бару t:
  1. forward_returns = log(close[t+1:t+horizon+1] / close[t:t+horizon])
  2. realized_vol[t] = std(forward_returns)
  3. Дискретизація через адаптивні пороги:
     - rolling percentiles за lookback window
     - або фіксовані квантилі (tertiles/quartiles)

Label:
  HIGH  якщо realized_vol > upper_quantile (e.g. 67th percentile)
  LOW   якщо realized_vol < lower_quantile (e.g. 33rd percentile)
  NONE  інакше (нормальна волатильність)
```

Два режими дискретизації:
- **Adaptive** (default): rolling percentiles за `lookback_window` -- адаптується до зміни волатильності з часом
- **Static**: фіксовані пороги в абсолютних одиницях волатильності

**Параметри**:
| Параметр | Default | Опис |
|----------|---------|------|
| `price_col` | `"close"` | Колонка ціни |
| `horizon` | `60` | Forward window для розрахунку vol |
| `upper_quantile` | `0.67` | Верхній поріг (HIGH) |
| `lower_quantile` | `0.33` | Нижній поріг (LOW) |
| `lookback_window` | `1440` | Вікно для adaptive percentiles |
| `static_mode` | `False` | Фіксовані пороги замість adaptive |
| `static_high` | `None` | Фіксований поріг HIGH (static mode) |
| `static_low` | `None` | Фіксований поріг LOW (static mode) |

**Meta-columns** (при `include_meta=True`): `realized_vol`, `vol_percentile`

Використання:
- Фільтрація сигналів: торгувати тільки при HIGH vol (momentum) або LOW vol (mean reversion)
- Position sizing: зменшити розмір позиції при HIGH vol
- Комбінування з Triple Barrier: різні параметри бар'єрів для різних режимів
- Target для MultiTargetGenerator: вже є `volume_regime` target type -- цей labeler дає точнішу альтернативу

Paper: "Improving ML Stock Trading: N-Period Volatility Labeling and Instance Selection" (2024)

**Performance**: O(n x horizon) для rolling std, векторизовано через Polars
**Тести**: `tests/test_volatility_labeler.py` (~12 тестів: adaptive/static, quantile correctness, multi-pair, meta)

---

### Файли для зміни

**Нові файли (6)**:
1. `src/signalflow/target/trend_scanning.py` -- TrendScanningLabeler
2. `src/signalflow/target/cusum_filter.py` -- CusumFilter
3. `src/signalflow/target/volatility_labeler.py` -- VolatilityRegimeLabeler
4. `tests/test_trend_scanning.py` -- тести trend scanning
5. `tests/test_cusum_filter.py` -- тести CUSUM filter
6. `tests/test_volatility_labeler.py` -- тести volatility labeler

**Модифікація існуючих (2)**:
7. `src/signalflow/target/__init__.py` -- export TrendScanningLabeler, CusumFilter, VolatilityRegimeLabeler
8. `docs/api/labeler.md` -- документація всіх трьох компонентів

### Верифікація

```bash
pytest tests/test_trend_scanning.py tests/test_cusum_filter.py tests/test_volatility_labeler.py -v
```

---

## Джерела

### Книги
- Lopez de Prado M. "Advances in Financial Machine Learning" (Wiley, 2018)
- Lopez de Prado M. "Machine Learning for Asset Managers" (Cambridge, 2020)

### Бібліотеки
- [mlfinlab](https://www.mlfinlab.com/) -- Hudson & Thames, повна імплементація AFML
- [mlfinpy](https://mlfinpy.readthedocs.io/) -- альтернативна pure-Python AFML імплементація

### Papers (2024-2025)
- "Adaptive Event-Driven Labeling: Multi-Scale Causal Framework with Meta-Learning" (MDPI, 2025)
- "Enhanced GA-Driven Triple Barrier Labeling" (MDPI, 2024)
- "Change-Point Detection in Financial Time Series Using the PELT Algorithm" (ACM, 2025)
- "Denoised Labels for Financial Time-Series via Self-Supervised Learning" (ArXiv 2112.10139, 2021)
- "TLOB: Dual Attention for Stock Price Trend Prediction" (2025)
- "Financial Information Theory" (ArXiv:2511.16339, 2025)
- "Contrastive Learning of Asset Embeddings from Financial Time Series" (2024)
- "Improving ML Stock Trading: N-Period Volatility Labeling" (2024)
