"""MemFlow Pipeline -- orchestrates Observer, Extractor, Memory, and Corrector."""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from .corrector import Corrector
from .extractor import StateExtractor
from .memory import StructuredMemory
from .observer import Observer
from .types import CorrectionStrategy, MemoryState, Observation, SceneState
from ..engines.base import BaseWorldEngine, Frame


class MemFlowPipeline:
    """Full MemFlow pipeline wrapping a world-model engine.

    Usage::

        pipeline = MemFlowPipeline(engine, strategy=CorrectionStrategy.FRAME_INJECTION)
        pipeline.start()
        for action in action_stream:
            frame, obs, scene = pipeline.step(action)
        snapshot = pipeline.memory.snapshot()
    """

    def __init__(
        self,
        engine: BaseWorldEngine,
        strategy: CorrectionStrategy = CorrectionStrategy.FRAME_INJECTION,
        observer_window: int = 128,
        nudge_strength: float = 0.15,
        injection_interval: int = 30,
        enable_correction: bool = True,
    ):
        self.engine = engine
        self.observer = Observer(engine, window_size=observer_window)
        self.extractor = StateExtractor()
        self.memory = StructuredMemory()
        self.corrector = Corrector(
            strategy=strategy,
            nudge_strength=nudge_strength,
            injection_interval=injection_interval,
        )
        self.enable_correction = enable_correction
        self._step_count = 0
        self._event_log: list[dict] = []

    def start(self) -> None:
        """Reset all components for a fresh session."""
        self.observer.reset()
        self.memory.clear()
        self.corrector.reset()
        self._step_count = 0
        self._event_log.clear()

    def step(self, actions: dict[str, Any]) -> tuple[Frame, Observation, SceneState]:
        """Run one generate-observe-extract-correct cycle."""
        # 1. Generate and observe
        frame, obs = self.observer.generate_and_observe(actions)

        # 2. Extract scene state
        scene = self.extractor.classify_scene(obs)

        # 3. Ingest into structured memory
        self.memory.ingest_scene(scene)

        # 4. Detect scene changes
        scene_changed = self.extractor.detect_scene_change(obs)
        if scene_changed:
            self._event_log.append({
                "event": "scene_change",
                "from": self.memory.current_scene,
                "to": scene.scene_id,
                "frame_idx": obs.frame_idx,
                "timestamp": time.time(),
            })

        # 5. Apply memory decay
        self.memory.decay(current_time=obs.timestamp)

        # 6. Apply correction if enabled and due
        correction = None
        if self.enable_correction:
            mem_state = self.memory.snapshot()
            correction = self.corrector.apply(
                self.engine, mem_state, current_obs=obs, frame_idx=obs.frame_idx
            )
            if correction:
                self._event_log.append({
                    "event": "correction",
                    "details": correction,
                    "frame_idx": obs.frame_idx,
                })

        self._step_count += 1
        return frame, obs, scene

    def store_reference(self, key: str, obs: Observation) -> None:
        """Store a reference observation for later correction."""
        self.corrector.store_reference_frame(key, obs.rgb)
        if obs.latent is not None:
            self.corrector.store_reference_latent(key, obs.latent)

    def run_sequence(
        self,
        action_sequence: list[dict[str, Any]],
        store_ref_at: Optional[dict[int, str]] = None,
    ) -> list[tuple[Frame, Observation, SceneState]]:
        """Run a full action sequence, optionally storing references at specific steps.

        Args:
            action_sequence: List of action dicts.
            store_ref_at: Dict mapping step index -> reference key to store.
        """
        store_ref_at = store_ref_at or {}
        results = []
        for i, action in enumerate(action_sequence):
            frame, obs, scene = self.step(action)
            if i in store_ref_at:
                self.store_reference(store_ref_at[i], obs)
            results.append((frame, obs, scene))
        return results

    def get_event_log(self) -> list[dict]:
        return list(self._event_log)

    def get_stats(self) -> dict:
        return {
            "steps": self._step_count,
            "observations": self.observer.frame_count(),
            "memory": self.memory.stats(),
            "corrections": len(self.corrector.get_correction_log()),
            "events": len(self._event_log),
        }
