"""World model engine adapters."""

from .base import BaseWorldEngine, EngineState, EngineStatus, Frame
from .mineworld_engine import MineWorldEngine
from .oasis_engine import OasisEngine
from .world_engine_adapter import WorldEngineAdapter

__all__ = [
    "BaseWorldEngine",
    "EngineState",
    "EngineStatus",
    "Frame",
    "MineWorldEngine",
    "OasisEngine",
    "WorldEngineAdapter",
]

ENGINE_CLASSES = {
    "mineworld": MineWorldEngine,
    "open_oasis": OasisEngine,
    "world_engine": WorldEngineAdapter,
}
