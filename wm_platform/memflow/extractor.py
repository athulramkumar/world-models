"""MemFlow State Extractor -- derives semantic state from observed frames/latents."""

from __future__ import annotations

import hashlib
import time
from typing import Optional

import cv2
import numpy as np

from .types import ObjectCategory, ObjectState, Observation, SceneState


class StateExtractor:
    """Extracts objects, scenes, and entities from world-model observations.

    Uses lightweight computer-vision heuristics designed for Minecraft-style
    block-world imagery.  Can be upgraded to CLIP-based extraction later.
    """

    # Minecraft-approximate colour ranges in HSV for common block/item types
    COLOR_PROFILES: dict[str, dict] = {
        "chest": {
            "hsv_low": (10, 80, 80),
            "hsv_high": (25, 255, 200),
            "category": ObjectCategory.CONTAINER,
        },
        "diamond": {
            "hsv_low": (85, 100, 150),
            "hsv_high": (100, 255, 255),
            "category": ObjectCategory.ITEM,
        },
        "grass_block": {
            "hsv_low": (35, 50, 50),
            "hsv_high": (85, 255, 200),
            "category": ObjectCategory.BLOCK,
        },
        "wood_plank": {
            "hsv_low": (15, 40, 80),
            "hsv_high": (30, 180, 200),
            "category": ObjectCategory.BLOCK,
        },
        "stone": {
            "hsv_low": (0, 0, 80),
            "hsv_high": (180, 30, 180),
            "category": ObjectCategory.BLOCK,
        },
        "entity_skin": {
            "hsv_low": (0, 30, 100),
            "hsv_high": (20, 200, 255),
            "category": ObjectCategory.ENTITY,
        },
    }

    SCENE_PROFILES: dict[str, dict] = {
        "indoor_wood": {
            "dominant_hue_range": (15, 35),
            "label": "wooden room (kitchen/house)",
        },
        "outdoor_grass": {
            "dominant_hue_range": (35, 85),
            "label": "outdoor / grassy area",
        },
        "underground": {
            "dominant_hue_range": (0, 15),
            "label": "underground / cave",
        },
    }

    def __init__(self, min_object_area: int = 100, scene_history_len: int = 8):
        self.min_object_area = min_object_area
        self._scene_history: list[SceneState] = []
        self._scene_history_len = scene_history_len
        self._obj_counter = 0

    def _next_obj_id(self, prefix: str = "obj") -> str:
        self._obj_counter += 1
        return f"{prefix}_{self._obj_counter}"

    def extract_objects(self, obs: Observation) -> list[ObjectState]:
        """Detect objects in a single observation frame using colour segmentation."""
        hsv = cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2HSV)
        objects: list[ObjectState] = []

        for label, profile in self.COLOR_PROFILES.items():
            low = np.array(profile["hsv_low"], dtype=np.uint8)
            high = np.array(profile["hsv_high"], dtype=np.uint8)
            mask = cv2.inRange(hsv, low, high)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_object_area:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                region = obs.rgb[y : y + h, x : x + w]
                feature_vec = self._compute_feature(region)
                obj = ObjectState(
                    obj_id=self._next_obj_id(label),
                    category=profile["category"],
                    label=label,
                    bbox=(x, y, x + w, y + h),
                    features=feature_vec,
                    properties={"area": int(area)},
                    confidence=min(1.0, area / 500.0),
                    first_seen=obs.timestamp,
                    last_seen=obs.timestamp,
                )
                objects.append(obj)

        return objects

    def classify_scene(self, obs: Observation) -> SceneState:
        """Classify the overall scene/location from a frame."""
        hsv = cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2HSV)
        hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
        dominant_hue = int(np.argmax(hue_hist))

        scene_label = "unknown"
        scene_id = "scene_unknown"
        for sid, profile in self.SCENE_PROFILES.items():
            lo, hi = profile["dominant_hue_range"]
            if lo <= dominant_hue <= hi:
                scene_label = profile["label"]
                scene_id = sid
                break

        feature_vec = self._compute_scene_feature(obs.rgb)
        objects = self.extract_objects(obs)

        scene = SceneState(
            scene_id=scene_id,
            label=scene_label,
            features=feature_vec,
            objects=objects,
            confidence=0.8,
            timestamp=obs.timestamp,
        )
        self._scene_history.append(scene)
        if len(self._scene_history) > self._scene_history_len:
            self._scene_history.pop(0)
        return scene

    def detect_scene_change(self, obs: Observation, threshold: float = 0.4) -> bool:
        """Return True if the current frame is in a different scene than the previous."""
        if len(self._scene_history) < 2:
            return False
        prev = self._scene_history[-2]
        curr = self._scene_history[-1]
        if prev.features is not None and curr.features is not None:
            dist = np.linalg.norm(prev.features - curr.features)
            return dist > threshold
        return prev.scene_id != curr.scene_id

    def match_object_to_known(
        self,
        obj: ObjectState,
        known: list[ObjectState],
        threshold: float = 0.3,
    ) -> Optional[ObjectState]:
        """Find the closest match among known objects by feature distance."""
        if obj.features is None or not known:
            return None
        best, best_dist = None, float("inf")
        for k in known:
            if k.features is None or k.category != obj.category:
                continue
            dist = float(np.linalg.norm(obj.features - k.features))
            if dist < best_dist:
                best, best_dist = k, dist
        return best if best_dist < threshold else None

    @staticmethod
    def _compute_feature(region: np.ndarray, bins: int = 16) -> np.ndarray:
        """Colour-histogram feature vector for an image region."""
        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256]).flatten()
        feat = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat

    @staticmethod
    def _compute_scene_feature(rgb: np.ndarray, grid: int = 4) -> np.ndarray:
        """Spatial colour feature: divide frame into grid cells, compute histograms."""
        h, w = rgb.shape[:2]
        cell_h, cell_w = h // grid, w // grid
        features = []
        for r in range(grid):
            for c in range(grid):
                cell = rgb[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
                feat = StateExtractor._compute_feature(cell, bins=8)
                features.append(feat)
        return np.concatenate(features).astype(np.float32)
