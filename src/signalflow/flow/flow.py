"""Flow - the declarative, deployable, tradeable unit."""


from dataclasses import dataclass, field, replace

from signalflow.enums import RunMode
from signalflow.errors import UntrainedModelError
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

    def _check_fitted(self) -> None:
        for slot, model in self.forecasts.items():
            if not getattr(model, "is_fitted", False):
                raise UntrainedModelError(
                    f"forecast slot {slot!r} holds an unfitted model; fit or load it before assembly"
                )
        if self.validator is not None and not getattr(self.validator, "is_fitted", False):
            raise UntrainedModelError("validator slot holds an unfitted model")


    def quicktest(self, data, capital, target: str | None = None, horizon: int = 24):
        return run_quicktest(self, data, capital, target, horizon=horizon)

    def backtest(self, data, capital, target: str | None = None, broker=None):
        broker = broker or self._sim_broker()
        return run_event_loop(self, data, capital, target, broker, RunMode.BACKTEST)

    def paper(self, data, capital, target: str | None = None, broker=None, load_data: bool = True):
        broker = broker or self._sim_broker()
        return run_event_loop(self, data, capital, target, broker, RunMode.PAPER)

    def live(self, data, capital, target: str | None = None, broker=None, load_data: bool = True, armed: bool = False):
        if armed and broker is None:
            raise UntrainedModelError(
                "armed live requires an explicit ExchangeBroker; refusing to send real orders via SimBroker"
            )
        broker = broker or self._sim_broker()
        return run_event_loop(self, data, capital, target, broker, RunMode.LIVE)

    def _sim_broker(self):
        from signalflow.engine.broker import SimBroker

        return SimBroker(quote=self.quote)


    def replace(self, **changes) -> "Flow":
        return replace(self, **changes)

    def save(self, path: str, model_dir: str | None = None) -> str:
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
