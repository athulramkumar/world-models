"""MemFlow Structured Memory -- graph-based world state with temporal tracking."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .types import (
    MemoryEdge,
    MemoryNode,
    MemoryState,
    ObjectState,
    RelationType,
    SceneState,
)


class StructuredMemory:
    """In-memory graph database tracking objects, locations, entities, and their
    relationships across time.  Persists beyond the world model's context window."""

    def __init__(self, decay_rate: float = 0.001, min_confidence: float = 0.05):
        self._nodes: dict[str, MemoryNode] = {}
        self._edges: list[MemoryEdge] = []
        self._current_scene: Optional[str] = None
        self.decay_rate = decay_rate
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------ #
    #  Node operations
    # ------------------------------------------------------------------ #

    def add_node(self, node: MemoryNode) -> MemoryNode:
        if node.node_id in self._nodes:
            existing = self._nodes[node.node_id]
            existing.last_seen = node.last_seen
            existing.observation_count += 1
            existing.confidence = min(1.0, existing.confidence + 0.1)
            if node.features is not None:
                existing.features = node.features
            existing.properties.update(node.properties)
            return existing
        self._nodes[node.node_id] = node
        node.observation_count = 1
        return node

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        return self._nodes.get(node_id)

    def find_nodes(self, node_type: Optional[str] = None, label: Optional[str] = None) -> list[MemoryNode]:
        results = []
        for n in self._nodes.values():
            if node_type and n.node_type != node_type:
                continue
            if label and label.lower() not in n.label.lower():
                continue
            results.append(n)
        return results

    # ------------------------------------------------------------------ #
    #  Edge operations
    # ------------------------------------------------------------------ #

    def add_edge(self, edge: MemoryEdge) -> MemoryEdge:
        for existing in self._edges:
            if (
                existing.source_id == edge.source_id
                and existing.target_id == edge.target_id
                and existing.relation == edge.relation
            ):
                existing.last_seen = edge.last_seen
                existing.confidence = min(1.0, existing.confidence + 0.1)
                return existing
        self._edges.append(edge)
        return edge

    def remove_edge(self, source_id: str, target_id: str, relation: RelationType) -> bool:
        before = len(self._edges)
        self._edges = [
            e for e in self._edges
            if not (e.source_id == source_id and e.target_id == target_id and e.relation == relation)
        ]
        return len(self._edges) < before

    def get_edges_for(self, node_id: str, relation: Optional[RelationType] = None) -> list[MemoryEdge]:
        return [
            e for e in self._edges
            if (e.source_id == node_id or e.target_id == node_id)
            and (relation is None or e.relation == relation)
        ]

    # ------------------------------------------------------------------ #
    #  Scene tracking
    # ------------------------------------------------------------------ #

    @property
    def current_scene(self) -> Optional[str]:
        return self._current_scene

    def set_current_scene(self, scene_id: str) -> None:
        self._current_scene = scene_id

    # ------------------------------------------------------------------ #
    #  Ingest from extractor
    # ------------------------------------------------------------------ #

    def ingest_scene(self, scene: SceneState) -> None:
        """Add a scene observation to memory, including all detected objects."""
        now = scene.timestamp

        scene_node = MemoryNode(
            node_id=scene.scene_id,
            node_type="location",
            label=scene.label,
            features=scene.features,
            confidence=scene.confidence,
            created_at=now,
            last_seen=now,
        )
        self.add_node(scene_node)
        self.set_current_scene(scene.scene_id)

        for obj in scene.objects:
            obj_node = MemoryNode(
                node_id=obj.obj_id,
                node_type=obj.category.value,
                label=obj.label,
                features=obj.features,
                properties=obj.properties,
                confidence=obj.confidence,
                created_at=now,
                last_seen=now,
            )
            self.add_node(obj_node)

            rel = RelationType.INSIDE if obj.category.value == "container" else RelationType.IN
            self.add_edge(
                MemoryEdge(
                    source_id=obj.obj_id,
                    target_id=scene.scene_id,
                    relation=rel,
                    confidence=obj.confidence,
                    created_at=now,
                    last_seen=now,
                )
            )

    def ingest_object_in_container(
        self,
        obj: ObjectState,
        container_id: str,
        location_id: Optional[str] = None,
    ) -> None:
        """Record that an object was placed inside a container."""
        now = time.time()
        obj_node = MemoryNode(
            node_id=obj.obj_id,
            node_type=obj.category.value,
            label=obj.label,
            features=obj.features,
            confidence=obj.confidence,
            created_at=now,
            last_seen=now,
        )
        self.add_node(obj_node)
        self.add_edge(
            MemoryEdge(
                source_id=obj.obj_id,
                target_id=container_id,
                relation=RelationType.INSIDE,
                confidence=obj.confidence,
                created_at=now,
                last_seen=now,
            )
        )
        if location_id:
            self.add_edge(
                MemoryEdge(
                    source_id=container_id,
                    target_id=location_id,
                    relation=RelationType.IN,
                    confidence=obj.confidence,
                    created_at=now,
                    last_seen=now,
                )
            )

    def ingest_entity_at_location(self, entity: ObjectState, location_id: str) -> None:
        """Record that an entity is at a specific location."""
        now = time.time()
        ent_node = MemoryNode(
            node_id=entity.obj_id,
            node_type="entity",
            label=entity.label,
            features=entity.features,
            confidence=entity.confidence,
            created_at=now,
            last_seen=now,
        )
        self.add_node(ent_node)
        self.add_edge(
            MemoryEdge(
                source_id=entity.obj_id,
                target_id=location_id,
                relation=RelationType.AT,
                confidence=entity.confidence,
                created_at=now,
                last_seen=now,
            )
        )

    # ------------------------------------------------------------------ #
    #  Confidence decay
    # ------------------------------------------------------------------ #

    def decay(self, current_time: Optional[float] = None) -> None:
        """Apply time-based confidence decay to all nodes and edges."""
        now = current_time or time.time()

        for node in list(self._nodes.values()):
            elapsed = now - node.last_seen
            node.confidence = max(self.min_confidence, node.confidence - self.decay_rate * elapsed)

        surviving = []
        for edge in self._edges:
            elapsed = now - edge.last_seen
            edge.confidence = max(self.min_confidence, edge.confidence - self.decay_rate * elapsed)
            surviving.append(edge)
        self._edges = surviving

    # ------------------------------------------------------------------ #
    #  Query
    # ------------------------------------------------------------------ #

    def query_objects_at(self, location_id: str) -> list[MemoryNode]:
        """Get all objects at a given location (direct or inside containers there)."""
        direct = [
            self._nodes[e.source_id]
            for e in self._edges
            if e.target_id == location_id
            and e.relation in (RelationType.IN, RelationType.INSIDE)
            and e.source_id in self._nodes
        ]
        containers_here = [n for n in direct if n.node_type == "container"]
        nested = []
        for c in containers_here:
            nested.extend(
                self._nodes[e.source_id]
                for e in self._edges
                if e.target_id == c.node_id
                and e.relation == RelationType.INSIDE
                and e.source_id in self._nodes
            )
        return direct + nested

    def query_entities_at(self, location_id: str) -> list[MemoryNode]:
        return [
            self._nodes[e.source_id]
            for e in self._edges
            if e.target_id == location_id
            and e.relation == RelationType.AT
            and e.source_id in self._nodes
        ]

    def query_container_contents(self, container_id: str) -> list[MemoryNode]:
        return [
            self._nodes[e.source_id]
            for e in self._edges
            if e.target_id == container_id
            and e.relation == RelationType.INSIDE
            and e.source_id in self._nodes
        ]

    # ------------------------------------------------------------------ #
    #  Snapshot
    # ------------------------------------------------------------------ #

    def snapshot(self) -> MemoryState:
        """Return a serializable snapshot of the current memory."""
        return MemoryState(
            nodes=list(self._nodes.values()),
            edges=list(self._edges),
            current_scene=self._current_scene,
            timestamp=time.time(),
        )

    def clear(self) -> None:
        self._nodes.clear()
        self._edges.clear()
        self._current_scene = None

    def stats(self) -> dict:
        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "current_scene": self._current_scene,
            "locations": len(self.find_nodes(node_type="location")),
            "objects": len([n for n in self._nodes.values() if n.node_type not in ("location", "entity")]),
            "entities": len(self.find_nodes(node_type="entity")),
        }
