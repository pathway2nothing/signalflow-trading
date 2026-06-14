"""SignalFlow exception hierarchy."""


class SignalFlowError(Exception):
    """Base class for every SignalFlow error."""


class UntrainedModelError(SignalFlowError):
    """A Flow slot holds a model that has not been fitted."""


class LeakageError(SignalFlowError):
    """Invariant L violated."""


class PipeError(SignalFlowError):
    """A FeaturePipe was built with an incompatible transform."""


class RiskViolation(SignalFlowError):
    """A proposed intent breached a hard risk constraint."""


class KillSwitchTripped(RiskViolation):
    """The persistent kill switch is engaged; no orders may be sent."""


class ArtifactError(SignalFlowError):
    """A model/OOS artifact could not be saved, loaded, or resolved."""


class FingerprintMismatch(ArtifactError):
    """Cached OOS does not cover the requested span and cannot be regenerated."""


class SchemaVersionError(SignalFlowError):
    """A frozen strategy model was loaded against a different Observation schema than it was trained on."""


class UnknownComponentError(SignalFlowError, KeyError):
    """A name was not found in the registry."""


class UnfittedTransformError(SignalFlowError):
    """A stateful transform (``requires_fit``) was used before ``fit``."""
