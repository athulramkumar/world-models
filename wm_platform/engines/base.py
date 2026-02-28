"""Abstract base class for world-model engine adapters."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np


class EngineState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    GENERATING = "generating"
    ERROR = "error"


@dataclass
class Frame:
    rgb: np.ndarray  # uint8 (H, W, 3)
    latent: Optional[np.ndarray] = None  # float32 latent tensor, flattened or shaped
    frame_idx: int = 0
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineStatus:
    name: str
    state: EngineState = EngineState.UNLOADED
    model_name: str = ""
    frames_generated: int = 0
    vram_used_mb: float = 0.0
    last_gen_time_ms: float = 0.0
    error: Optional[str] = None


class BaseWorldEngine(ABC):
    """Unified interface that every world-model adapter must implement."""

    def __init__(self, name: str, repo_path: Path, checkpoint_dir: Path):
        self.name = name
        self.repo_path = repo_path
        self.checkpoint_dir = checkpoint_dir
        self._status = EngineStatus(name=name)
        self._frame_counter = 0
        self._context_frames: list[Frame] = []

    @property
    def status(self) -> EngineStatus:
        return self._status

    @abstractmethod
    def load(self, model_variant: Optional[str] = None, **kwargs) -> None:
        """Load model weights onto GPU."""
        ...

    @abstractmethod
    def generate_frame(self, actions: dict[str, Any]) -> Frame:
        """Generate the next frame given an action dict."""
        ...

    @abstractmethod
    def get_latents(self) -> Optional[np.ndarray]:
        """Return the most recent latent representation (for MemFlow)."""
        ...

    def inject_conditioning(self, memory_state: Any) -> None:
        """Reinject MemFlow corrections. Override per engine for actual logic."""
        pass

    def append_context_frame(self, frame: Frame) -> None:
        """Add an externally supplied frame to the context window."""
        self._context_frames.append(frame)

    @abstractmethod
    def reset(self) -> None:
        """Clear context and reset generation state."""
        ...

    def unload(self) -> None:
        """Release GPU resources."""
        self._status.state = EngineState.UNLOADED
        self._context_frames.clear()
        self._frame_counter = 0

    def _tick_frame(self) -> int:
        idx = self._frame_counter
        self._frame_counter += 1
        self._status.frames_generated = self._frame_counter
        return idx

    def _make_frame(
        self,
        rgb: np.ndarray,
        latent: Optional[np.ndarray] = None,
        **meta,
    ) -> Frame:
        idx = self._tick_frame()
        return Frame(
            rgb=rgb,
            latent=latent,
            frame_idx=idx,
            timestamp=time.time(),
            metadata=meta,
        )
