"""MemFlow Corrector -- reinjects memory state into generation to maintain consistency."""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from .types import CorrectionStrategy, MemoryState, Observation
from ..engines.base import BaseWorldEngine, Frame


class Corrector:
    """Reinjects structured memory into the world-model generation loop.

    Three pluggable strategies, each suited to different engine capabilities:

    1. Latent Nudge   -- modify latent tokens toward stored reference patterns
    2. Frame Injection -- periodically inject a reference frame via append_frame()
    3. Prompt Conditioning -- update text prompt with memory description (WorldEngine)
    """

    def __init__(
        self,
        strategy: CorrectionStrategy = CorrectionStrategy.FRAME_INJECTION,
        nudge_strength: float = 0.15,
        injection_interval: int = 30,
    ):
        self.strategy = strategy
        self.nudge_strength = nudge_strength
        self.injection_interval = injection_interval
        self._reference_latents: dict[str, np.ndarray] = {}
        self._reference_frames: dict[str, np.ndarray] = {}
        self._last_injection_idx = 0
        self._correction_log: list[dict] = []

    # ------------------------------------------------------------------ #
    #  Reference storage
    # ------------------------------------------------------------------ #

    def store_reference_latent(self, key: str, latent: np.ndarray) -> None:
        """Store a latent pattern associated with a memory key (e.g. 'kitchen_chest_diamond')."""
        self._reference_latents[key] = latent.copy()

    def store_reference_frame(self, key: str, frame: np.ndarray) -> None:
        """Store a reference RGB frame for frame-injection strategy."""
        self._reference_frames[key] = frame.copy()

    # ------------------------------------------------------------------ #
    #  Correction application
    # ------------------------------------------------------------------ #

    def should_correct(self, frame_idx: int) -> bool:
        """Check if a correction should be applied at this frame index."""
        return (frame_idx - self._last_injection_idx) >= self.injection_interval

    def apply(
        self,
        engine: BaseWorldEngine,
        memory: MemoryState,
        current_obs: Optional[Observation] = None,
        frame_idx: int = 0,
    ) -> Optional[dict]:
        """Apply the configured correction strategy. Returns a log dict or None."""

        if self.strategy == CorrectionStrategy.LATENT_NUDGE:
            return self._apply_latent_nudge(engine, memory, current_obs)

        if self.strategy == CorrectionStrategy.FRAME_INJECTION:
            if not self.should_correct(frame_idx):
                return None
            return self._apply_frame_injection(engine, memory, frame_idx)

        if self.strategy == CorrectionStrategy.PROMPT_CONDITIONING:
            return self._apply_prompt_conditioning(engine, memory)

        return None

    def _apply_latent_nudge(
        self,
        engine: BaseWorldEngine,
        memory: MemoryState,
        current_obs: Optional[Observation],
    ) -> Optional[dict]:
        """Nudge current latents toward stored reference patterns."""
        if current_obs is None or current_obs.latent is None:
            return None

        nudges_applied = []
        current_latent = current_obs.latent.copy()

        for key, ref_latent in self._reference_latents.items():
            if ref_latent.shape != current_latent.shape:
                continue
            delta = ref_latent - current_latent
            current_latent += self.nudge_strength * delta
            nudges_applied.append(key)

        if nudges_applied:
            log_entry = {
                "strategy": "latent_nudge",
                "keys": nudges_applied,
                "strength": self.nudge_strength,
                "timestamp": time.time(),
            }
            self._correction_log.append(log_entry)
            return log_entry
        return None

    def _apply_frame_injection(
        self,
        engine: BaseWorldEngine,
        memory: MemoryState,
        frame_idx: int,
    ) -> Optional[dict]:
        """Inject a stored reference frame to anchor the model's memory."""
        scene_id = memory.current_scene
        if scene_id and scene_id in self._reference_frames:
            ref_frame = self._reference_frames[scene_id]
            frame = Frame(rgb=ref_frame, frame_idx=frame_idx)
            engine.append_context_frame(frame)
            self._last_injection_idx = frame_idx
            log_entry = {
                "strategy": "frame_injection",
                "scene_id": scene_id,
                "frame_idx": frame_idx,
                "timestamp": time.time(),
            }
            self._correction_log.append(log_entry)
            return log_entry

        for key, ref_frame in self._reference_frames.items():
            frame = Frame(rgb=ref_frame, frame_idx=frame_idx)
            engine.append_context_frame(frame)
            self._last_injection_idx = frame_idx
            log_entry = {
                "strategy": "frame_injection",
                "key": key,
                "frame_idx": frame_idx,
                "timestamp": time.time(),
            }
            self._correction_log.append(log_entry)
            return log_entry

        return None

    def _apply_prompt_conditioning(
        self,
        engine: BaseWorldEngine,
        memory: MemoryState,
    ) -> Optional[dict]:
        """Update text prompt with memory description (WorldEngine only)."""
        prompt = memory.to_prompt()
        engine.inject_conditioning(memory)
        log_entry = {
            "strategy": "prompt_conditioning",
            "prompt": prompt,
            "timestamp": time.time(),
        }
        self._correction_log.append(log_entry)
        return log_entry

    # ------------------------------------------------------------------ #
    #  Logging
    # ------------------------------------------------------------------ #

    def get_correction_log(self) -> list[dict]:
        return list(self._correction_log)

    def reset(self) -> None:
        self._reference_latents.clear()
        self._reference_frames.clear()
        self._last_injection_idx = 0
        self._correction_log.clear()
