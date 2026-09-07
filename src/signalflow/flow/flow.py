"""Flow - the declarative, deployable, tradeable unit."""

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from signalflow.enums import ComponentType, RunMode
from signalflow.errors import FlowConfigError, UnknownComponentError, UntrainedModelError
from signalflow.flow.decide import Buffer, Decision, decide
from signalflow.flow.live import ReplayFeed, run_live_loop
from signalflow.flow.loop import run_event_loop, run_quicktest
from signalflow.strategy.risk import Risk
from signalflow.strategy.rules import RulesStrategy

if TYPE_CHECKING:
    from signalflow.detector.base import SignalDetector
    from signalflow.model.forecast import ForecastModel
    from signalflow.strategy.protocol import StrategyModel


@dataclass
class Flow:
    """The framework's central noun: what you research, promote, and trade."""

    name: str
    forecasts: "dict[str, ForecastModel]" = field(default_factory=dict)
    detectors: "list[SignalDetector]" = field(default_factory=list)
    strategy: "StrategyModel" = field(default_factory=RulesStrategy)
    risk: Risk = field(default_factory=Risk)
    validator: "ForecastModel | None" = None
    quote: str = "USDT"

    def __post_init__(self) -> None:
        self._check_fitted()
        self._check_wiring()

    def _check_fitted(self) -> None:
        for slot, model in self.forecasts.items():
            if not getattr(model, "is_fitted", False):
                raise UntrainedModelError(
                    f"forecast slot {slot!r} holds an unfitted model; fit or load it before assembly"
                )
        if self.validator is not None and not getattr(self.validator, "is_fitted", False):
            raise UntrainedModelError("validator slot holds an unfitted model")

    def _available_slots(self) -> "list[str]":
        slots = list(self.forecasts)
        if self.validator is not None:
            slots.append("validator")
        return slots

    def _check_wiring(self) -> None:
        from loguru import logger

        available = set(self._available_slots())
        for det in self.detectors:
            if getattr(det, "learned", False):
                logger.warning(
                    f"detector {det.name!r} is learned: it fits on the frame it detects on, so its signals "
                    f"are in-sample; treat backtests of this flow as not promotable"
                )
            for slot in getattr(det, "required_slots", lambda: ())():
                if slot not in available:
                    raise FlowConfigError(
                        f"detector {det.name!r} references forecast slot {slot!r} which is not wired "
                        f"into the flow; available slots: {sorted(available)}"
                    )
            self._check_targets(det)

    def _check_targets(self, det: Any) -> None:
        constraints = getattr(det, "required_targets", {}) or {}
        for slot, accepted in constraints.items():
            model = self.forecasts.get(slot)
            target = getattr(model, "target", None) if model is not None else None
            if target is None:
                continue
            classes = self._resolve_targets(accepted)
            if classes and not isinstance(target, classes):
                raise FlowConfigError(
                    f"detector {det.name!r} slot {slot!r} requires a target in {sorted(accepted)}, "
                    f"but the wired model targets {getattr(target, 'name', type(target).__name__)!r}"
                )

    @staticmethod
    def _resolve_targets(names: Any) -> tuple:
        from signalflow.registry import registry

        resolved = []
        for name in names:
            try:
                resolved.append(registry.get(ComponentType.TARGET, name))
            except UnknownComponentError:
                continue
        return tuple(resolved)

    @property
    def required_warmup(self) -> int:
        """Bars of history the flow needs before its outputs are valid.

        Max over each detector's warmup, each forecast model's feature-pipe warmup,
        and the validator's feature warmup. Zero when the flow has none of these.
        """
        candidates = [0]
        candidates += [int(getattr(det, "warmup", 0)) for det in self.detectors]
        candidates += [self._model_warmup(model) for model in self.forecasts.values()]
        if self.validator is not None:
            candidates.append(self._model_warmup(self.validator))
        return max(candidates)

    def buffer(self, window: int | None = None) -> "Buffer":
        """A trailing-window buffer sized for this flow (``required_warmup + 1`` bars by default)."""
        return Buffer(self.required_warmup + 1 if window is None else window)

    def decide(
        self,
        history: Any,
        snapshot: Any,
        ts: Any = None,
        *,
        peak: float | None = None,
        mandate: dict | None = None,
        raise_on_trip: bool = False,
        prices: dict | None = None,
    ) -> "Decision":
        """What this flow would do on one bar, with no side effects.

        The live loop's body as a call: forecasts and detectors over ``history``
        (a :meth:`buffer`, a Dataset or a frame), the strategy on the signals at
        ``ts`` and the portfolio ``snapshot``, the risk layer, and the orders to
        send. See :func:`signalflow.flow.decide.decide` for the parameters.
        """
        return decide(
            self, history, snapshot, ts, peak=peak, mandate=mandate, raise_on_trip=raise_on_trip, prices=prices
        )

    def check_warmup(self, interval: str = "1h", raise_on_fail: bool = True, **kw: Any) -> list:
        """Measure every detector and model pipeline against its declared warmup (the canary).

        Runs each component on a deterministic synthetic series and on a trailing
        window of exactly the declared bars; they must agree on the last row(s).
        Returns the list of :class:`~signalflow.transform.warmup.WarmupCheck`; with
        ``raise_on_fail`` a failing component raises :class:`WarmupError`. Called by
        default from :meth:`simulate` and :meth:`live`.
        """
        from signalflow.transform.warmup import check_flow, raise_if_failed

        checks = check_flow(self, interval=interval, **kw)
        if raise_on_fail:
            raise_if_failed(checks)
        return checks

    @staticmethod
    def _model_warmup(model: Any) -> int:
        """Feature-pipe warmup of a forecast model or a validator combinator."""
        features = getattr(model, "features", None)
        if features is not None and hasattr(features, "warmup"):
            return int(features.warmup)
        children = getattr(model, "children", None)
        if children:
            return max((Flow._model_warmup(child) for child in children), default=0)
        return 0

    def quicktest(
        self, data: Any, capital: Any, target: str | None = None, horizon: int = 24, fee: float = 0.001
    ) -> Any:
        return run_quicktest(self, data, capital, target, horizon=horizon, fee=fee)

    def backtest(
        self, data: Any, capital: Any, target: str | None = None, broker: Any = None, oos: bool = False
    ) -> Any:
        """Backtest the flow.

        ``oos=True`` scores leak-free out-of-fold predictions and stamps the Run
        promotable (with enough OOS coverage). Without it the run is in-sample and not
        promotable - for a rule-only flow too, since its parameters were tuned somewhere.
        """
        broker = broker or self._sim_broker()
        return run_event_loop(self, data, capital, target, broker, RunMode.BACKTEST, oos=oos)

    def paper(self, data: Any, capital: Any, target: str | None = None, broker: Any = None) -> Any:
        """Replay a Dataset with simulated fills - the same loop as backtest, paper mode."""
        broker = broker or self._sim_broker()
        return run_event_loop(self, data, capital, target, broker, RunMode.PAPER)

    def live(
        self,
        feed: Any,
        capital: Any,
        target: str | None = None,
        broker: Any = None,
        armed: bool = False,
        maxlen: int = 5000,
        max_bars: int | None = None,
        state_path: str | None = None,
        compute_window: "int | None" = None,
        check_warmup: bool = True,
        mandate: dict | None = None,
        on_bar: Any = None,
        max_latency_s: float = 10.0,
        late_bar_policy: str = "warn",
    ) -> Any:
        """Trade a live (or replayed) feed via the real-time loop.

        ``feed`` may be a LiveFeed or a Dataset (wrapped in a ReplayFeed). Armed
        trading requires an explicit ExchangeBroker; SimBroker is paper-only.
        ``check_warmup`` runs :meth:`check_warmup` first and refuses to start on a
        component that needs more bars than it declares. ``mandate`` reaches the
        strategy on every bar; ``on_bar(engine, bar, fills, latency)`` is called
        after each bar; ``max_latency_s``/``late_bar_policy`` (``"warn"`` or
        ``"skip"``) govern bars that arrive late.
        """
        if armed and broker is None:
            raise FlowConfigError(
                "armed live requires an explicit ExchangeBroker; refusing to send real orders via SimBroker"
            )
        broker = broker or self._sim_broker()
        if not hasattr(feed, "stream"):
            feed = ReplayFeed(feed)
        if check_warmup:
            self.check_warmup(interval=getattr(feed, "interval", None) or "1h")
        return run_live_loop(
            self,
            feed,
            capital,
            broker,
            target=target,
            maxlen=maxlen,
            max_bars=max_bars,
            state_path=state_path,
            armed=armed,
            compute_window=compute_window,
            mandate=mandate,
            on_bar=on_bar,
            max_latency_s=max_latency_s,
            late_bar_policy=late_bar_policy,
        )

    def simulate(
        self,
        data: Any,
        capital: Any,
        target: str | None = None,
        broker: Any = None,
        warmup: int | None = None,
        maxlen: int = 5000,
        state_path: str | None = None,
        compute_window: "int | None" = None,
        check_warmup: bool = True,
        mandate: dict | None = None,
        on_bar: Any = None,
    ) -> Any:
        """Full-speed incremental live simulation (walk-forward).

        Replays a Dataset through the live decision loop with no real-time wait:
        the flow sees only data up to each bar, recomputed step by step, exactly
        as in live. ``warmup`` reserves a leading lookback window that fills the
        buffer without trading; ``None`` resolves to :attr:`required_warmup` while
        an explicit ``0`` is honored. Use it to confirm the live path before arming.
        ``check_warmup`` runs the warmup canary first (see :meth:`check_warmup`);
        ``mandate`` and ``on_bar`` are passed to the loop as in :meth:`live`.
        """
        broker = broker or self._sim_broker()
        if check_warmup:
            self.check_warmup(interval=getattr(data, "source_params", {}).get("interval") or "1h")
        warmup = self.required_warmup if warmup is None else warmup
        feed = ReplayFeed(data, warmup_bars=warmup)
        return run_live_loop(
            self,
            feed,
            capital,
            broker,
            target=target,
            maxlen=maxlen,
            state_path=state_path,
            compute_window=compute_window,
            mandate=mandate,
            on_bar=on_bar,
        )

    def _sim_broker(self) -> Any:
        from signalflow.engine.broker import SimBroker

        return SimBroker(quote=self.quote)

    def replace(self, **changes: Any) -> "Flow":
        return replace(self, **changes)

    def save(self, path: str, model_dir: str | None = None, run: Any = None) -> str:
        """Serialize the flow to YAML at ``path`` and return it.

        Each forecast/validator must already have a pinned URI, or pass ``model_dir`` to
        save the trained artifacts there. ``load`` restores a byte-identical backtest.
        Passing ``run`` also writes its ``scorecard()`` to ``scorecard.json`` beside the
        yaml as promotion evidence.
        """
        from signalflow.flow.yaml import save_flow

        return save_flow(self, path, model_dir=model_dir, run=run)

    def save_bundle(self, dir_path: str, run: Any) -> str:
        """Write a promotable bundle (flow.yaml + models + scorecard.json + manifest.json)."""
        from signalflow.flow.bundle import write_bundle

        return write_bundle(self, run, dir_path)

    @classmethod
    def load(cls, path: str, trust_remote: bool = False) -> "Flow":
        """Load a saved flow; ``trust_remote=True`` is required for ``hf://`` model artifacts."""
        from signalflow.flow.yaml import load_flow

        return load_flow(path, trust_remote=trust_remote)

    def __repr__(self) -> str:
        return (
            f"Flow({self.name!r}, forecasts={list(self.forecasts)}, "
            f"detectors={[d.name for d in self.detectors]}, "
            f"validator={'yes' if self.validator else 'none'})"
        )
