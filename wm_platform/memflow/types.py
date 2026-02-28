"""Shared data types for the MemFlow pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

import numpy as np


@dataclass
class Observation:
    """A single observation from the world-model generation loop."""
    frame_idx: int
    timestamp: float
    rgb: np.ndarray                        # uint8 (H, W, 3)
    latent: Optional[np.ndarray] = None    # float32 latent tensor
    action: Optional[dict[str, Any]] = None
    engine_id: str = ""


class ObjectCategory(str, Enum):
    BLOCK = "block"
    ITEM = "item"
    ENTITY = "entity"
    CONTAINER = "container"
    STRUCTURE = "structure"
    UNKNOWN = "unknown"


@dataclass
class ObjectState:
    """Semantic state of a detected object."""
    obj_id: str
    category: ObjectCategory = ObjectCategory.UNKNOWN
    label: str = ""
    bbox: Optional[tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    features: Optional[np.ndarray] = None              # embedding vector
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


@dataclass
class SceneState:
    """High-level scene / location classification."""
    scene_id: str
    label: str = ""                    # e.g. "kitchen", "forest biome"
    features: Optional[np.ndarray] = None
    objects: list[ObjectState] = field(default_factory=list)
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)


class RelationType(str, Enum):
    IN = "in"              # object IN location
    AT = "at"              # entity AT location
    INSIDE = "inside"      # object INSIDE container
    NEAR = "near"          # proximity
    OWNS = "owns"          # entity owns object
    SAME_AS = "same_as"    # identity link across time


@dataclass
class MemoryNode:
    """A node in the structured memory graph."""
    node_id: str
    node_type: str                    # "object", "location", "entity"
    label: str = ""
    features: Optional[np.ndarray] = None
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    observation_count: int = 0


@dataclass
class MemoryEdge:
    """An edge (relationship) in the structured memory graph."""
    source_id: str
    target_id: str
    relation: RelationType
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


class CorrectionStrategy(str, Enum):
    LATENT_NUDGE = "latent_nudge"
    FRAME_INJECTION = "frame_injection"
    PROMPT_CONDITIONING = "prompt_conditioning"


@dataclass
class MemoryState:
    """Snapshot of the full structured memory for handoff to the Corrector."""
    nodes: list[MemoryNode] = field(default_factory=list)
    edges: list[MemoryEdge] = field(default_factory=list)
    current_scene: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_prompt(self) -> str:
        """Render memory state as a natural-language prompt for prompt conditioning."""
        parts = []
        location_objects: dict[str, list[str]] = {}

        for edge in self.edges:
            if edge.relation in (RelationType.IN, RelationType.INSIDE):
                location_objects.setdefault(edge.target_id, []).append(edge.source_id)

        node_map = {n.node_id: n for n in self.nodes}
        for loc_id, obj_ids in location_objects.items():
            loc = node_map.get(loc_id)
            loc_label = loc.label if loc else loc_id
            obj_labels = [node_map[o].label if o in node_map else o for o in obj_ids]
            parts.append(f"In the {loc_label}: {', '.join(obj_labels)}")

        entities = [n for n in self.nodes if n.node_type == "entity"]
        for e in entities:
            loc_edges = [ed for ed in self.edges if ed.source_id == e.node_id and ed.relation == RelationType.AT]
            if loc_edges:
                loc = node_map.get(loc_edges[0].target_id)
                parts.append(f"{e.label} is at the {loc.label if loc else loc_edges[0].target_id}")

        if self.current_scene:
            scene = node_map.get(self.current_scene)
            if scene:
                parts.insert(0, f"Current location: {scene.label}")

        return ". ".join(parts) if parts else "An explorable world"
