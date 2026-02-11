"""VL-JEPA: Simplified Video-Language Joint Embedding Predictive Architecture."""

from vljepa.config import Config
from vljepa.models import VLJepa
from vljepa.losses import vl_jepa_loss

__all__ = ["Config", "VLJepa", "vl_jepa_loss"]
