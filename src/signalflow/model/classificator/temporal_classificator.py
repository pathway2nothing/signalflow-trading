from signalflow.core import SfTorchModuleMixin
import lightning as L

class TemporalClassificator(L.LightningModule, SfTorchModuleMixin):
    def __init__(self):
        super().__init__()
        self.model = L.LightningModule()