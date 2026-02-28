"""MemFlow Panel -- memory visualization, test scenario runner, and comparison view."""

from __future__ import annotations

import json
import time
from typing import Optional

import gradio as gr
import numpy as np

from ..memflow.types import (
    CorrectionStrategy,
    MemoryNode,
    ObjectCategory,
    ObjectState,
    RelationType,
    MemoryEdge,
)
from ..memflow.memory import StructuredMemory
from ..memflow.corrector import Corrector


_demo_memory = StructuredMemory(decay_rate=0.0001, min_confidence=0.05)


def _render_memory_graph(memory: StructuredMemory) -> str:
    """Render the memory graph as a Markdown table."""
    snapshot = memory.snapshot()
    if not snapshot.nodes:
        return "Memory is empty."

    lines = [
        "### Nodes",
        "| ID | Type | Label | Confidence | Last Seen |",
        "| --- | --- | --- | --- | --- |",
    ]
    for n in snapshot.nodes:
        ago = f"{time.time() - n.last_seen:.1f}s ago"
        lines.append(f"| {n.node_id} | {n.node_type} | {n.label} | {n.confidence:.3f} | {ago} |")

    lines.append("\n### Edges")
    lines.append("| Source | Relation | Target | Confidence |")
    lines.append("| --- | --- | --- | --- |")
    for e in snapshot.edges:
        lines.append(f"| {e.source_id} | {e.relation.value} | {e.target_id} | {e.confidence:.3f} |")

    stats = memory.stats()
    lines.append(f"\n**Stats**: {stats['nodes']} nodes, {stats['edges']} edges, "
                 f"{stats['locations']} locations, {stats['objects']} objects, {stats['entities']} entities")
    if snapshot.current_scene:
        lines.append(f"**Current scene**: {snapshot.current_scene}")

    return "\n".join(lines)


def run_kitchen_test(duration_seconds: int) -> str:
    """Run the object persistence test (diamond in chest)."""
    memory = StructuredMemory(decay_rate=0.0001, min_confidence=0.05)
    t0 = time.time()

    kitchen = MemoryNode(node_id="kitchen", node_type="location", label="kitchen",
                         confidence=1.0, created_at=t0, last_seen=t0)
    chest = MemoryNode(node_id="chest_1", node_type="container", label="chest",
                       confidence=1.0, created_at=t0, last_seen=t0)
    memory.add_node(kitchen)
    memory.add_node(chest)
    memory.add_edge(MemoryEdge(source_id="chest_1", target_id="kitchen",
                               relation=RelationType.IN, confidence=1.0,
                               created_at=t0, last_seen=t0))
    diamond = ObjectState(obj_id="diamond_1", category=ObjectCategory.ITEM,
                          label="diamond", confidence=1.0, first_seen=t0, last_seen=t0)
    memory.ingest_object_in_container(diamond, "chest_1", "kitchen")

    living = MemoryNode(node_id="living_room", node_type="location", label="living room",
                        confidence=1.0, created_at=t0+1, last_seen=t0+1)
    memory.add_node(living)
    memory.set_current_scene("living_room")
    fps = 6
    for i in range(int(duration_seconds * fps)):
        memory.decay(current_time=t0 + 1 + i / fps)

    memory.set_current_scene("kitchen")
    contents = memory.query_container_contents("chest_1")
    diamond_found = any(n.label == "diamond" for n in contents)
    diamond_conf = next((n.confidence for n in contents if n.label == "diamond"), 0)

    result = [
        f"## Kitchen Test Results (duration={duration_seconds}s)",
        f"- Diamond in chest after {duration_seconds}s: **{'PASS' if diamond_found else 'FAIL'}**",
        f"- Diamond confidence: **{diamond_conf:.4f}**",
        f"- Min threshold: {memory.min_confidence}",
        "",
        _render_memory_graph(memory),
    ]
    return "\n".join(result)


def run_character_test(duration_seconds: int) -> str:
    """Run the character persistence test (two entities reconvene)."""
    memory = StructuredMemory(decay_rate=0.00001, min_confidence=0.05)
    t0 = time.time()

    for loc_id, label in [("living_room", "living room"), ("kitchen", "kitchen"), ("bedroom", "bedroom")]:
        memory.add_node(MemoryNode(node_id=loc_id, node_type="location", label=label,
                                   confidence=1.0, created_at=t0, last_seen=t0))

    rng_a = np.random.RandomState(42)
    rng_b = np.random.RandomState(99)
    alice = ObjectState(obj_id="alice", category=ObjectCategory.ENTITY, label="Alice",
                        features=rng_a.randn(48).astype(np.float32), confidence=1.0)
    bob = ObjectState(obj_id="bob", category=ObjectCategory.ENTITY, label="Bob",
                      features=rng_b.randn(48).astype(np.float32), confidence=1.0)

    memory.ingest_entity_at_location(alice, "living_room")
    memory.ingest_entity_at_location(bob, "living_room")

    t1 = t0 + 1
    memory.remove_edge("alice", "living_room", RelationType.AT)
    memory.add_edge(MemoryEdge(source_id="alice", target_id="kitchen", relation=RelationType.AT,
                               confidence=0.9, created_at=t1, last_seen=t1))
    memory.remove_edge("bob", "living_room", RelationType.AT)
    memory.add_edge(MemoryEdge(source_id="bob", target_id="bedroom", relation=RelationType.AT,
                               confidence=0.9, created_at=t1, last_seen=t1))

    fps = 6
    for i in range(int(duration_seconds * fps)):
        memory.decay(current_time=t1 + i / fps)

    t2 = t1 + duration_seconds
    memory.remove_edge("bob", "bedroom", RelationType.AT)
    memory.add_edge(MemoryEdge(source_id="bob", target_id="kitchen", relation=RelationType.AT,
                               confidence=0.9, created_at=t2, last_seen=t2))
    memory.set_current_scene("kitchen")

    alice_node = memory.get_node("alice")
    bob_node = memory.get_node("bob")
    alice_ok = alice_node is not None and alice_node.confidence >= memory.min_confidence
    bob_ok = bob_node is not None and bob_node.confidence >= memory.min_confidence

    features_match_a = alice_node is not None and np.array_equal(alice_node.features, alice.features)
    features_match_b = bob_node is not None and np.array_equal(bob_node.features, bob.features)

    entities_at_kitchen = memory.query_entities_at("kitchen")
    entity_ids = {e.node_id for e in entities_at_kitchen}

    result = [
        f"## Character Test Results (duration={duration_seconds}s)",
        f"- Alice persists: **{'PASS' if alice_ok else 'FAIL'}** (conf={alice_node.confidence:.4f})" if alice_node else "- Alice: **FAIL** (missing)",
        f"- Bob persists: **{'PASS' if bob_ok else 'FAIL'}** (conf={bob_node.confidence:.4f})" if bob_node else "- Bob: **FAIL** (missing)",
        f"- Alice features intact: **{'PASS' if features_match_a else 'FAIL'}**",
        f"- Bob features intact: **{'PASS' if features_match_b else 'FAIL'}**",
        f"- Both at kitchen: **{'PASS' if 'alice' in entity_ids and 'bob' in entity_ids else 'FAIL'}**",
        "",
        _render_memory_graph(memory),
    ]
    return "\n".join(result)


def build_memflow_tab() -> gr.Blocks:
    with gr.Blocks() as tab:
        gr.Markdown("## MemFlow -- Structured State Management")
        gr.Markdown(
            "Test long-term memory and state persistence across scene changes. "
            "MemFlow maintains a structured memory graph that persists beyond the "
            "world model's context window."
        )

        with gr.Tabs():
            with gr.Tab("Memory Visualization"):
                gr.Markdown("### Current Memory State")
                memory_display = gr.Markdown(_render_memory_graph(_demo_memory))
                refresh_btn = gr.Button("Refresh")
                refresh_btn.click(lambda: _render_memory_graph(_demo_memory), outputs=memory_display)

            with gr.Tab("Object Persistence Test"):
                gr.Markdown(
                    "**Scenario**: Place a diamond in a chest in the kitchen. "
                    "Explore another room for N seconds. Return. Does the memory "
                    "still record the diamond?"
                )
                duration_slider = gr.Slider(10, 300, 60, step=10, label="Absence Duration (seconds)")
                run_btn = gr.Button("Run Test", variant="primary")
                results = gr.Markdown()
                run_btn.click(run_kitchen_test, inputs=duration_slider, outputs=results)

            with gr.Tab("Character Persistence Test"):
                gr.Markdown(
                    "**Scenario**: Two characters (Alice and Bob) meet in the living room. "
                    "They separate to different rooms. After N seconds they reconvene in "
                    "the kitchen. Are their identities preserved?"
                )
                char_duration = gr.Slider(10, 300, 60, step=10, label="Separation Duration (seconds)")
                char_run_btn = gr.Button("Run Test", variant="primary")
                char_results = gr.Markdown()
                char_run_btn.click(run_character_test, inputs=char_duration, outputs=char_results)

    return tab
