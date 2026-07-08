"""Flow - the declarative, deployable, tradeable unit."""

from dataclasses import dataclass, field, replace

from signalflow.enums import ComponentType, RunMode
from signalflow.errors import FlowConfigError, UnknownComponentError, UntrainedModelError
from signalflow.flow.live import ReplayFeed, run_live_loop
from signalflow.flow.loop import run_event_loop, run_quicktest
from signalflow.strategy.risk import Risk
from signalflow.strategy.rules import RulesStrategy


@dataclass
class Flow:
    """The framework's central noun: what you research, promote, and trade."""

    name: str
    forecasts: dict = field(default_factory=dict)
    detectors: list = field(default_factory=list)
    strategy: object = field(default_factory=RulesStrategy)
    risk: Risk = field(default_factory=Risk)
    validator: object | None = None
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
        available = set(self._available_slots())
        for det in self.detectors:
            for slot in getattr(det, "required_slots", lambda: ())():
                if slot not in available:
                    raise FlowConfigError(
                        f"detector {det.name!r} references forecast slot {slot!r} which is not wired "
                        f"into the flow; available slots: {sorted(available)}"
                    )
            self._check_targets(det)

    def _check_targets(self, det) -> None:
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
    def _resolve_targets(names) -> tuple:
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

    @staticmethod
    def _model_warmup(model) -> int:
        """Feature-pipe warmup of a forecast model or a validator combinator."""
        features = getattr(model, "features", None)
        if features is not None and hasattr(features, "warmup"):
            return int(features.warmup)
        children = getattr(model, "children", None)
        if children:
            return max((Flow._model_warmup(child) for child in children), default=0)
        return 0

    def quicktest(self, data, capital, target: str | None = None, horizon: int = 24, fee: float = 0.001):
        return run_quicktest(self, data, capital, target, horizon=horizon, fee=fee)

    def backtest(self, data, capital, target: str | None = None, broker=None, oos: bool = False):
        """Backtest the flow. ``oos=True`` scores leak-free out-of-fold predictions and
        stamps the Run promotable; the default in-sample run of a model flow is not promotable.
        """
        broker = broker or self._sim_broker()
        return run_event_loop(self, data, capital, target, broker, RunMode.BACKTEST, oos=oos)

    def paper(self, data, capital, target: str | None = None, broker=None):
        """Replay a Dataset with simulated fills - the same loop as backtest, paper mode."""
        broker = broker or self._sim_broker()
        return run_event_loop(self, data, capital, target, broker, RunMode.PAPER)

    def live(
        self,
        feed,
        capital,
        target: str | None = None,
        broker=None,
        armed: bool = False,
        max_bars: int | None = None,
        state_path: str | None = None,
    ):
        """Trade a live (or replayed) feed via the real-time loop.

        ``feed`` may be a LiveFeed or a Dataset (wrapped in a ReplayFeed). Armed
        trading requires an explicit ExchangeBroker; SimBroker is paper-only.
        """
        if armed and broker is None:
            raise FlowConfigError(
                "armed live requires an explicit ExchangeBroker; refusing to send real orders via SimBroker"
            )
        broker = broker or self._sim_broker()
        if not hasattr(feed, "stream"):
            feed = ReplayFeed(feed)
        return run_live_loop(self, feed, capital, broker, target=target, max_bars=max_bars, state_path=state_path)

    def simulate(
        self,
        data,
        capital,
        target: str | None = None,
        broker=None,
        warmup: int | None = None,
        maxlen: int = 5000,
        state_path: str | None = None,
    ):
        """Full-speed incremental live simulation (walk-forward).

        Replays a Dataset through the live decision loop with no real-time wait:
        the flow sees only data up to each bar, recomputed step by step, exactly
        as in live. ``warmup`` reserves a leading lookback window that fills the
        buffer without trading; ``None`` resolves to :attr:`required_warmup` while
        an explicit ``0`` is honored. Use it to confirm the live path before arming.
        """
        broker = broker or self._sim_broker()
        warmup = self.required_warmup if warmup is None else warmup
        feed = ReplayFeed(data, warmup_bars=warmup)
        return run_live_loop(self, feed, capital, broker, target=target, maxlen=maxlen, state_path=state_path)

    def _sim_broker(self):
        from signalflow.engine.broker import SimBroker

        return SimBroker(quote=self.quote)

    def replace(self, **changes) -> "Flow":
        return replace(self, **changes)

    def save(self, path: str, model_dir: str | None = None) -> str:
        """Serialize the flow to YAML at ``path`` and return it.

        Each forecast/validator must already have a pinned URI, or pass ``model_dir`` to
        save the trained artifacts there. ``load`` restores a byte-identical backtest.
        """
        from signalflow.flow.yaml import save_flow

        return save_flow(self, path, model_dir=model_dir)

    @classmethod
    def load(cls, path: str) -> "Flow":
        from signalflow.flow.yaml import load_flow

        return load_flow(path)

    def __repr__(self) -> str:
        return (
            f"Flow({self.name!r}, forecasts={list(self.forecasts)}, "
            f"detectors={[d.name for d in self.detectors]}, "
            f"validator={'yes' if self.validator else 'none'})"
        )
