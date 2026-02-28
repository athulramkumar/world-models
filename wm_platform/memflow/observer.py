"""MemFlow Observer -- captures frames, latents, and actions from engine adapters."""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

import numpy as np

from .types import Observation
from ..engines.base import BaseWorldEngine, Frame


class Observer:
    """Wraps an engine adapter and records every generated frame as an Observation."""

    def __init__(self, engine: BaseWorldEngine, window_size: int = 128):
        self.engine = engine
        self.window_size = window_size
        self._history: deque[Observation] = deque(maxlen=window_size)
        self._frame_idx = 0
        self._callbacks: list = []

    @property
    def history(self) -> list[Observation]:
        return list(self._history)

    @property
    def latest(self) -> Optional[Observation]:
        return self._history[-1] if self._history else None

    def on_observation(self, callback) -> None:
        """Register a callback invoked after each observation."""
        self._callbacks.append(callback)

    def observe_frame(self, frame: Frame, action: Optional[dict] = None) -> Observation:
        """Manually record a frame (e.g. from external source)."""
        latent = self.engine.get_latents()
        obs = Observation(
            frame_idx=self._frame_idx,
            timestamp=time.time(),
            rgb=frame.rgb,
            latent=latent,
            action=action,
            engine_id=self.engine.name,
        )
        self._history.append(obs)
        self._frame_idx += 1
        for cb in self._callbacks:
            cb(obs)
        return obs

    def generate_and_observe(self, actions: dict) -> tuple[Frame, Observation]:
        """Generate a frame via the engine and simultaneously observe it."""
        frame = self.engine.generate_frame(actions)
        obs = self.observe_frame(frame, action=actions)
        return frame, obs

    def get_observations_since(self, since_idx: int) -> list[Observation]:
        return [o for o in self._history if o.frame_idx >= since_idx]

    def get_observations_in_window(self, last_n: int) -> list[Observation]:
        return list(self._history)[-last_n:]

    def reset(self) -> None:
        self._history.clear()
        self._frame_idx = 0
        self.engine.reset()

    def frame_count(self) -> int:
        return self._frame_idx
