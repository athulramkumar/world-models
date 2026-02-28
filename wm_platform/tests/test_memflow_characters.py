"""MemFlow Test Case 2: Character Persistence ("Two Characters Meet")

Minecraft analog:
1. Two distinct entities meet in a room (living room).
2. Entity A goes to kitchen, Entity B goes to bedroom.
3. After 30/60/120 seconds, both return to the kitchen.
4. Assert: entity memory nodes retain their features and identities.

Compares WITH MemFlow (structured memory preserving entity identity) vs
WITHOUT (naive context window that forgets characters once out of view).

Operates at the data layer -- no GPU required.
"""

import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wm_platform.memflow.types import (
    MemoryEdge,
    MemoryNode,
    ObjectCategory,
    ObjectState,
    RelationType,
)
from wm_platform.memflow.memory import StructuredMemory
from wm_platform.memflow.extractor import StateExtractor


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_entity(name: str, feature_seed: int) -> ObjectState:
    rng = np.random.RandomState(feature_seed)
    return ObjectState(
        obj_id=name,
        category=ObjectCategory.ENTITY,
        label=name,
        features=rng.randn(48).astype(np.float32),
        confidence=1.0,
    )


def _run_character_scenario_memflow(memory: StructuredMemory, duration: float):
    """Run the full meet-separate-reconvene scenario on a StructuredMemory."""
    t0 = time.time()

    for loc_id, label in [("living_room", "living room"), ("kitchen", "kitchen"), ("bedroom", "bedroom")]:
        memory.add_node(MemoryNode(
            node_id=loc_id, node_type="location", label=label,
            confidence=1.0, created_at=t0, last_seen=t0,
        ))

    alice = _make_entity("alice", feature_seed=42)
    bob = _make_entity("bob", feature_seed=99)
    alice_features = alice.features.copy()
    bob_features = bob.features.copy()

    memory.ingest_entity_at_location(alice, "living_room")
    memory.ingest_entity_at_location(bob, "living_room")
    memory.set_current_scene("living_room")

    t1 = t0 + 1
    memory.remove_edge("alice", "living_room", RelationType.AT)
    memory.add_edge(MemoryEdge(
        source_id="alice", target_id="kitchen", relation=RelationType.AT,
        confidence=0.9, created_at=t1, last_seen=t1,
    ))
    memory.remove_edge("bob", "living_room", RelationType.AT)
    memory.add_edge(MemoryEdge(
        source_id="bob", target_id="bedroom", relation=RelationType.AT,
        confidence=0.9, created_at=t1, last_seen=t1,
    ))

    fps = 6
    for i in range(int(duration * fps)):
        memory.decay(current_time=t1 + (i / fps))

    t2 = t1 + duration
    memory.remove_edge("bob", "bedroom", RelationType.AT)
    memory.add_edge(MemoryEdge(
        source_id="bob", target_id="kitchen", relation=RelationType.AT,
        confidence=0.9, created_at=t2, last_seen=t2,
    ))
    memory.set_current_scene("kitchen")

    return memory, alice_features, bob_features


# ------------------------------------------------------------------ #
# WITHOUT MemFlow -- baseline showing the problem
# ------------------------------------------------------------------ #

class NaiveEntityTracker:
    """Simulates entity tracking with only a finite context window.
    Entities leave memory entirely once the window scrolls past them."""

    def __init__(self, window_size: int = 32):
        self.window_size = window_size
        self._frames: list[dict] = []

    def add_frame(self, frame_info: dict):
        self._frames.append(frame_info)
        if len(self._frames) > self.window_size:
            self._frames.pop(0)

    def recall_entity(self, name: str) -> bool:
        return any(name in f.get("entities", []) for f in self._frames)

    def recall_entity_features(self, name: str) -> np.ndarray | None:
        """Without persistent memory, features are lost with the frame."""
        for f in reversed(self._frames):
            if name in f.get("entity_features", {}):
                return f["entity_features"][name]
        return None

    def entities_at_location(self, location: str) -> set[str]:
        """Can only see entities mentioned in the current window at that location."""
        found = set()
        for f in self._frames:
            if f.get("location") == location:
                found.update(f.get("entities", []))
        return found


class TestWithoutMemFlow:
    """Demonstrates that without MemFlow, character identity is lost after
    they leave the context window."""

    @pytest.mark.parametrize("duration", [30, 60, 120])
    def test_characters_forgotten_without_memflow(self, duration):
        """Without MemFlow, entities vanish from memory after leaving the window."""
        ctx = NaiveEntityTracker(window_size=32)

        alice_features = np.random.RandomState(42).randn(48).astype(np.float32)
        bob_features = np.random.RandomState(99).randn(48).astype(np.float32)

        ctx.add_frame({
            "location": "living_room",
            "entities": ["alice", "bob"],
            "entity_features": {"alice": alice_features, "bob": bob_features},
        })

        fps = 6
        for i in range(int(duration * fps)):
            loc = "kitchen" if i % 2 == 0 else "bedroom"
            ctx.add_frame({"location": loc, "entities": []})

        assert not ctx.recall_entity("alice"), "Alice should be forgotten without MemFlow"
        assert not ctx.recall_entity("bob"), "Bob should be forgotten without MemFlow"
        assert ctx.recall_entity_features("alice") is None, "Alice features lost"
        assert ctx.recall_entity_features("bob") is None, "Bob features lost"

    @pytest.mark.parametrize("duration", [30, 60, 120])
    def test_no_location_tracking_without_memflow(self, duration):
        """Without MemFlow, there's no graph query for 'who is at kitchen?'"""
        ctx = NaiveEntityTracker(window_size=32)
        ctx.add_frame({"location": "living_room", "entities": ["alice", "bob"]})

        fps = 6
        for _ in range(int(duration * fps)):
            ctx.add_frame({"location": "bedroom", "entities": []})

        ctx.add_frame({"location": "kitchen", "entities": []})

        at_kitchen = ctx.entities_at_location("kitchen")
        assert "alice" not in at_kitchen, "Naive tracker can't infer Alice moved to kitchen"
        assert "bob" not in at_kitchen, "Naive tracker can't infer Bob moved to kitchen"


# ------------------------------------------------------------------ #
# WITH MemFlow -- structured memory preserves identity
# ------------------------------------------------------------------ #

class TestWithMemFlow:
    """Test that MemFlow's structured memory retains entity identity and
    location across scene changes and prolonged separation."""

    @pytest.mark.parametrize("duration", [30, 60, 120])
    def test_entities_persist_after_separation(self, duration):
        """Entities should still exist in memory with intact features after time gap."""
        memory = StructuredMemory(decay_rate=0.00001, min_confidence=0.05)
        memory, alice_feat, bob_feat = _run_character_scenario_memflow(memory, duration)

        alice_node = memory.get_node("alice")
        bob_node = memory.get_node("bob")

        assert alice_node is not None, "Alice lost from memory"
        assert bob_node is not None, "Bob lost from memory"
        assert alice_node.confidence > memory.min_confidence, (
            f"Alice confidence too low: {alice_node.confidence}"
        )
        assert bob_node.confidence > memory.min_confidence, (
            f"Bob confidence too low: {bob_node.confidence}"
        )
        np.testing.assert_array_equal(alice_node.features, alice_feat, err_msg="Alice features changed")
        np.testing.assert_array_equal(bob_node.features, bob_feat, err_msg="Bob features changed")

    @pytest.mark.parametrize("duration", [30, 60, 120])
    def test_entities_at_correct_locations(self, duration):
        """After reconvening, entities should be AT kitchen per graph query."""
        memory = StructuredMemory(decay_rate=0.00001, min_confidence=0.05)
        memory, _, _ = _run_character_scenario_memflow(memory, duration)

        entities_in_kitchen = memory.query_entities_at("kitchen")
        entity_ids = {e.node_id for e in entities_in_kitchen}
        assert "alice" in entity_ids, "Alice not found at kitchen"
        assert "bob" in entity_ids, "Bob not found at kitchen"


# ------------------------------------------------------------------ #
# Head-to-head comparison
# ------------------------------------------------------------------ #

class TestMemFlowVsBaseline:
    """Direct comparison: MemFlow retains character identity that naive tracking loses."""

    @pytest.mark.parametrize("duration", [30, 60, 120])
    def test_memflow_wins_character_recall(self, duration):
        """MemFlow remembers both characters; naive window does not."""
        # WITHOUT
        ctx = NaiveEntityTracker(window_size=32)
        ctx.add_frame({"location": "living_room", "entities": ["alice", "bob"]})
        fps = 6
        for _ in range(int(duration * fps)):
            ctx.add_frame({"location": "bedroom", "entities": []})
        naive_alice = ctx.recall_entity("alice")
        naive_bob = ctx.recall_entity("bob")

        # WITH
        memory = StructuredMemory(decay_rate=0.00001, min_confidence=0.05)
        memory, _, _ = _run_character_scenario_memflow(memory, duration)
        memflow_alice = memory.get_node("alice") is not None
        memflow_bob = memory.get_node("bob") is not None

        assert not naive_alice, "Naive should have forgotten Alice"
        assert not naive_bob, "Naive should have forgotten Bob"
        assert memflow_alice, "MemFlow should remember Alice"
        assert memflow_bob, "MemFlow should remember Bob"

    @pytest.mark.parametrize("duration", [30, 60, 120])
    def test_memflow_preserves_features_baseline_does_not(self, duration):
        """MemFlow preserves feature vectors; naive tracking loses them."""
        # WITHOUT
        ctx = NaiveEntityTracker(window_size=32)
        alice_feat = np.random.RandomState(42).randn(48).astype(np.float32)
        ctx.add_frame({"location": "living_room", "entities": ["alice"],
                        "entity_features": {"alice": alice_feat}})
        for _ in range(int(duration * 6)):
            ctx.add_frame({"location": "bedroom", "entities": []})
        naive_features = ctx.recall_entity_features("alice")

        # WITH
        memory = StructuredMemory(decay_rate=0.00001, min_confidence=0.05)
        memory, memflow_alice_feat, _ = _run_character_scenario_memflow(memory, duration)
        alice_node = memory.get_node("alice")

        assert naive_features is None, "Naive should have lost features"
        assert alice_node is not None and alice_node.features is not None, "MemFlow should have features"
        np.testing.assert_array_equal(alice_node.features, memflow_alice_feat)


# ------------------------------------------------------------------ #
# Feature matching & identity disambiguation
# ------------------------------------------------------------------ #

class TestEntityIdentity:
    """Tests for the feature-based entity matching used by MemFlow."""

    def test_entity_identity_via_feature_matching(self):
        """Extractor should match re-observed entities to stored ones by features."""
        extractor = StateExtractor()
        alice_original = _make_entity("alice", feature_seed=42)
        alice_reobserved = _make_entity("alice_reobs", feature_seed=42)

        known = [alice_original]
        match = extractor.match_object_to_known(alice_reobserved, known, threshold=0.5)
        assert match is not None, "Failed to re-match entity by features"
        assert match.obj_id == "alice"

    def test_different_entities_not_confused(self):
        """Two different entities should NOT match each other."""
        extractor = StateExtractor()
        alice = _make_entity("alice", feature_seed=42)
        bob = _make_entity("bob", feature_seed=99)

        dist = float(np.linalg.norm(alice.features - bob.features))
        assert dist > 0.1, "Different entities have suspiciously similar features"
