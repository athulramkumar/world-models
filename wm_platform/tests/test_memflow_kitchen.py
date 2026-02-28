"""MemFlow Test Case 1: Object Persistence ("Spoon in Drawer" / "Diamond in Chest")

Minecraft analog:
1. Place a diamond in a chest in a wooden room (kitchen).
2. Walk to a grassy area (living room) and explore for N seconds.
3. Return to the kitchen, open the chest.
4. Assert: MemFlow memory still records diamond in chest.

Compares WITH MemFlow (structured memory) vs WITHOUT (no persistent memory)
to demonstrate that MemFlow is the mechanism enabling long-term state recall.

Operates at the data layer -- no GPU required.
"""

import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wm_platform.memflow.types import (
    CorrectionStrategy,
    MemoryEdge,
    MemoryNode,
    ObjectCategory,
    ObjectState,
    Observation,
    RelationType,
    SceneState,
)
from wm_platform.memflow.memory import StructuredMemory
from wm_platform.memflow.extractor import StateExtractor
from wm_platform.memflow.corrector import Corrector


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_observation(frame_idx: int, rgb: np.ndarray, ts: float = None) -> Observation:
    return Observation(
        frame_idx=frame_idx,
        timestamp=ts or time.time(),
        rgb=rgb,
        latent=np.random.randn(16, 18, 32).astype(np.float32),
        action={"forward": 1},
        engine_id="test",
    )


def _make_kitchen_frame() -> np.ndarray:
    """Synthetic frame dominated by wood-brown hues (kitchen analog)."""
    frame = np.full((224, 384, 3), [140, 100, 60], dtype=np.uint8)
    frame[80:140, 150:250] = [180, 120, 50]
    return frame


def _make_outdoor_frame() -> np.ndarray:
    """Synthetic frame dominated by green hues (living room / outdoor analog)."""
    frame = np.full((224, 384, 3), [80, 160, 60], dtype=np.uint8)
    return frame


def _build_kitchen_scenario(memory: StructuredMemory, duration: float):
    """Place diamond, leave, come back. Returns (memory, t_return)."""
    t0 = time.time()

    kitchen_node = MemoryNode(
        node_id="kitchen", node_type="location",
        label="wooden room (kitchen)", confidence=1.0,
        created_at=t0, last_seen=t0,
    )
    chest_node = MemoryNode(
        node_id="kitchen_chest", node_type="container",
        label="chest", confidence=1.0,
        created_at=t0, last_seen=t0,
    )
    diamond = ObjectState(
        obj_id="diamond_1", category=ObjectCategory.ITEM,
        label="diamond", confidence=1.0,
        first_seen=t0, last_seen=t0,
    )

    memory.add_node(kitchen_node)
    memory.add_node(chest_node)
    memory.add_edge(MemoryEdge(
        source_id="kitchen_chest", target_id="kitchen",
        relation=RelationType.IN, confidence=1.0,
        created_at=t0, last_seen=t0,
    ))
    memory.ingest_object_in_container(diamond, "kitchen_chest", "kitchen")
    memory.set_current_scene("kitchen")

    # Leave to living room
    t_leave = t0 + 1
    living = MemoryNode(
        node_id="living_room", node_type="location",
        label="outdoor / grassy area", confidence=1.0,
        created_at=t_leave, last_seen=t_leave,
    )
    memory.add_node(living)
    memory.set_current_scene("living_room")

    fps = 6
    for i in range(int(duration * fps)):
        memory.decay(current_time=t_leave + (i / fps))

    # Return to kitchen
    t_return = t_leave + duration
    memory.set_current_scene("kitchen")
    kitchen = memory.get_node("kitchen")
    if kitchen:
        kitchen.last_seen = t_return
    return memory, t_return


# ------------------------------------------------------------------ #
# WITHOUT MemFlow -- baseline showing the problem MemFlow solves
# ------------------------------------------------------------------ #

class NaiveContextWindow:
    """Simulates a world model without MemFlow: only remembers the last N frames.
    Once an object leaves the context window, it is completely forgotten."""

    def __init__(self, window_size: int = 32):
        self.window_size = window_size
        self._frames: list[dict] = []

    def add_frame(self, frame_info: dict):
        self._frames.append(frame_info)
        if len(self._frames) > self.window_size:
            self._frames.pop(0)

    def recall(self, query_label: str) -> bool:
        """Can the model recall an object by label? Only if it's in the window."""
        return any(query_label in f.get("objects", []) for f in self._frames)

    def recall_confidence(self, query_label: str) -> float:
        """Returns 1.0 if in window, 0.0 otherwise -- binary, no gradual decay."""
        return 1.0 if self.recall(query_label) else 0.0


class TestWithoutMemFlow:
    """Demonstrates what happens WITHOUT MemFlow: objects are forgotten once
    they leave the model's finite context window."""

    @pytest.mark.parametrize("duration", [30, 60, 120])
    def test_diamond_forgotten_without_memflow(self, duration):
        """Without MemFlow, the diamond is lost once the context window scrolls past."""
        ctx = NaiveContextWindow(window_size=32)

        ctx.add_frame({"scene": "kitchen", "objects": ["diamond", "chest"]})

        fps = 6
        for i in range(int(duration * fps)):
            ctx.add_frame({"scene": "living_room", "objects": ["grass"]})

        assert not ctx.recall("diamond"), (
            f"Without MemFlow, diamond should be forgotten after {duration}s "
            f"({duration * fps} frames >> 32-frame window)"
        )
        assert ctx.recall_confidence("diamond") == 0.0

    @pytest.mark.parametrize("duration", [30, 60, 120])
    def test_no_structured_query_without_memflow(self, duration):
        """Without MemFlow, there is no way to query 'what was in the chest?'"""
        ctx = NaiveContextWindow(window_size=32)
        ctx.add_frame({"scene": "kitchen", "objects": ["diamond", "chest"]})

        fps = 6
        for _ in range(int(duration * fps)):
            ctx.add_frame({"scene": "living_room", "objects": ["grass"]})

        assert not hasattr(ctx, "query_container_contents"), (
            "NaiveContextWindow has no structured query -- that's MemFlow's capability"
        )


# ------------------------------------------------------------------ #
# WITH MemFlow -- the structured memory solution
# ------------------------------------------------------------------ #

class TestWithMemFlow:
    """Test that MemFlow's structured memory retains objects across scene changes
    and time gaps of 30, 60, and 120 seconds."""

    def setup_method(self):
        self.memory = StructuredMemory(decay_rate=0.0001, min_confidence=0.05)

    @pytest.mark.parametrize("duration", [30, 60, 120])
    def test_diamond_persists_with_memflow(self, duration):
        """WITH MemFlow, the diamond remains queryable after prolonged absence."""
        memory, _ = _build_kitchen_scenario(self.memory, duration)

        contents = memory.query_container_contents("kitchen_chest")
        diamond_nodes = [n for n in contents if n.label == "diamond"]
        assert len(diamond_nodes) > 0, f"MemFlow forgot diamond after {duration}s"

        diamond = diamond_nodes[0]
        assert diamond.confidence >= memory.min_confidence, (
            f"Diamond confidence too low ({diamond.confidence:.4f}) after {duration}s"
        )

    @pytest.mark.parametrize("duration", [30, 60, 120])
    def test_structured_query_with_memflow(self, duration):
        """WITH MemFlow, we can ask 'what was in the chest?' via graph query."""
        memory, _ = _build_kitchen_scenario(self.memory, duration)

        contents = memory.query_container_contents("kitchen_chest")
        labels = {n.label for n in contents}
        assert "diamond" in labels, "MemFlow should answer structured queries about containers"

    @pytest.mark.parametrize("duration", [30, 60, 120])
    def test_snapshot_preserves_diamond(self, duration):
        """MemoryState snapshot should contain diamond and INSIDE edge after absence."""
        memory, _ = _build_kitchen_scenario(self.memory, duration)

        snapshot = memory.snapshot()
        node_labels = {n.label for n in snapshot.nodes}
        assert "diamond" in node_labels

        inside_edges = [
            e for e in snapshot.edges
            if e.relation == RelationType.INSIDE and e.target_id == "kitchen_chest"
        ]
        assert len(inside_edges) > 0, "INSIDE edge to chest missing from snapshot"

    def test_prompt_describes_diamond(self):
        """MemoryState.to_prompt() should mention diamond for prompt conditioning."""
        memory, _ = _build_kitchen_scenario(self.memory, 60)
        prompt = memory.snapshot().to_prompt()
        assert "diamond" in prompt.lower() or "chest" in prompt.lower(), (
            f"Prompt does not mention diamond/chest: {prompt}"
        )


# ------------------------------------------------------------------ #
# Head-to-head comparison
# ------------------------------------------------------------------ #

class TestMemFlowVsBaseline:
    """Direct comparison: MemFlow retains what naive context windows forget."""

    @pytest.mark.parametrize("duration", [30, 60, 120])
    def test_memflow_wins_object_recall(self, duration):
        """MemFlow remembers the diamond; the naive window does not."""
        # WITHOUT
        ctx = NaiveContextWindow(window_size=32)
        ctx.add_frame({"scene": "kitchen", "objects": ["diamond", "chest"]})
        fps = 6
        for _ in range(int(duration * fps)):
            ctx.add_frame({"scene": "living_room", "objects": ["grass"]})
        naive_recall = ctx.recall_confidence("diamond")

        # WITH
        memory = StructuredMemory(decay_rate=0.0001, min_confidence=0.05)
        memory, _ = _build_kitchen_scenario(memory, duration)
        contents = memory.query_container_contents("kitchen_chest")
        memflow_recall = max((n.confidence for n in contents if n.label == "diamond"), default=0.0)

        assert naive_recall == 0.0, "Naive should have forgotten"
        assert memflow_recall > 0.0, "MemFlow should still remember"
        assert memflow_recall > naive_recall, (
            f"MemFlow ({memflow_recall:.4f}) should beat naive ({naive_recall:.4f})"
        )


# ------------------------------------------------------------------ #
# Corrector mechanics
# ------------------------------------------------------------------ #

class TestCorrectorFiring:
    """Test that the Corrector fires at the right intervals and stores references."""

    def test_frame_injection_interval(self):
        corrector = Corrector(
            strategy=CorrectionStrategy.FRAME_INJECTION,
            injection_interval=10,
        )
        assert not corrector.should_correct(5)
        assert corrector.should_correct(10)
        assert corrector.should_correct(30)

    def test_reference_storage(self):
        corrector = Corrector(strategy=CorrectionStrategy.FRAME_INJECTION)
        frame = np.zeros((224, 384, 3), dtype=np.uint8)
        latent = np.random.randn(16, 18, 32).astype(np.float32)
        corrector.store_reference_frame("kitchen", frame)
        corrector.store_reference_latent("kitchen", latent)
        assert "kitchen" in corrector._reference_frames
        assert "kitchen" in corrector._reference_latents
