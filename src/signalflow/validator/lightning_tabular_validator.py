from signalflow.core import sf_component
from signalflow.validator.base_signal_validator import SignalValidator
from dataclasses import dataclass

@dataclass
@sf_component(name="lightning/tabular")
class LightningTabularValidator(SignalValidator):
    pass