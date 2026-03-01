"""Interactive Oasis demo with WebSocket streaming and MemFlow comparison."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import WebSocket, WebSocketDisconnect

ROOT = Path(__file__).resolve().parent.parent

OASIS_ACTION_KEYS = [
    "inventory", "ESC", "hotbar.1", "hotbar.2", "hotbar.3", "hotbar.4",
    "hotbar.5", "hotbar.6", "hotbar.7", "hotbar.8", "hotbar.9",
    "forward", "back", "left", "right", "cameraX", "cameraY",
    "jump", "sneak", "sprint", "swapHands", "attack", "use", "pickItem", "drop",
]

CAMERA_TURN_STRENGTH = 0.4

ACTION_MAP = {
    # Movement (binary: 0 or 1)
    "w": (11, 1.0),       # forward
    "s": (12, 1.0),       # back
    "a": (13, 1.0),       # left
    "d": (14, 1.0),       # right
    # Camera (continuous: [-1, 1])
    "ArrowLeft": (15, -CAMERA_TURN_STRENGTH),   # look left
    "ArrowRight": (15, CAMERA_TURN_STRENGTH),    # look right
    "ArrowUp": (16, -CAMERA_TURN_STRENGTH),      # look up
    "ArrowDown": (16, CAMERA_TURN_STRENGTH),     # look down
    "q": (15, -CAMERA_TURN_STRENGTH),            # look left (alt)
    "e": (15, CAMERA_TURN_STRENGTH),             # look right (alt)
    # Other actions (binary)
    "j": (17, 1.0),       # jump
    "Shift": (18, 1.0),   # sneak
    "Control": (19, 1.0), # sprint
    "f": (21, 1.0),       # attack
    "r": (22, 1.0),       # use/place
}

SAMPLE_DIR = ROOT / "repos" / "open-oasis" / "sample_data"

PROMPTS = {
    "default":        {"mp4": SAMPLE_DIR / "Player729-f153ac423f61-20210806-224813.chunk_000.mp4", "offset": 0},
    "default_mid":    {"mp4": SAMPLE_DIR / "Player729-f153ac423f61-20210806-224813.chunk_000.mp4", "offset": 400},
    "default_late":   {"mp4": SAMPLE_DIR / "Player729-f153ac423f61-20210806-224813.chunk_000.mp4", "offset": 800},
    "treechop":       {"mp4": SAMPLE_DIR / "treechop-f153ac423f61-20210916-183423.chunk_000.mp4", "offset": 0},
    "treechop_mid":   {"mp4": SAMPLE_DIR / "treechop-f153ac423f61-20210916-183423.chunk_000.mp4", "offset": 400},
    "treechop_late":  {"mp4": SAMPLE_DIR / "treechop-f153ac423f61-20210916-183423.chunk_000.mp4", "offset": 800},
    "snippy":         {"mp4": SAMPLE_DIR / "snippy-chartreuse-mastiff-f79998db196d-20220401-224517.chunk_001.mp4", "offset": 0},
    "snippy_mid":     {"mp4": SAMPLE_DIR / "snippy-chartreuse-mastiff-f79998db196d-20220401-224517.chunk_001.mp4", "offset": 400},
    "snippy_late":    {"mp4": SAMPLE_DIR / "snippy-chartreuse-mastiff-f79998db196d-20220401-224517.chunk_001.mp4", "offset": 800},
}


def key_to_action_vector(key: str) -> list[float]:
    """Convert a key name to a 25-dim action vector."""
    vec = [0.0] * 25
    mapping = ACTION_MAP.get(key)
    if mapping is not None:
        idx, val = mapping
        vec[idx] = val
    return vec


def frame_to_jpeg_b64(rgb: np.ndarray, quality: int = 85) -> str:
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def save_frame_as_prompt(rgb: np.ndarray, path: str):
    Image.fromarray(rgb).save(path)


class InteractiveSession:
    """Manages state for one interactive browser session."""

    def __init__(self):
        self.engine = None
        self.prompt_path: Optional[str] = None
        self.action_queue: list[list[float]] = []
        self.frame_count = 0
        self.total_frames_generated = 0
        self._memflow_memory = None
        self._memflow_extractor = None
        self._memflow_corrector = None
        self._bl_prompt_path: Optional[str] = None
        self._mf_prompt_path: Optional[str] = None
        self._tmp_dir = tempfile.mkdtemp(prefix="oasis_interactive_")

    def _load_engine(self):
        from wm_platform.engines.oasis_engine import OasisEngine
        self.engine = OasisEngine()
        self.engine.load()

    def _init_memflow(self):
        from wm_platform.memflow.extractor import StateExtractor
        from wm_platform.memflow.memory import StructuredMemory
        from wm_platform.memflow.corrector import Corrector
        from wm_platform.memflow.types import CorrectionStrategy

        self._memflow_extractor = StateExtractor()
        self._memflow_memory = StructuredMemory(decay_rate=0.0001, min_confidence=0.05)
        self._memflow_corrector = Corrector(
            strategy=CorrectionStrategy.FRAME_INJECTION,
            injection_interval=1,
        )

    def _extract_first_frame(self, prompt_name: str) -> str:
        """Extract a frame from sample MP4 (at the configured offset) as PNG prompt."""
        import cv2
        entry = PROMPTS.get(prompt_name)
        if entry:
            mp4_path = entry["mp4"]
            offset = entry.get("offset", 0)
            if mp4_path.exists():
                cap = cv2.VideoCapture(str(mp4_path))
                if offset > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, offset)
                ret, bgr = cap.read()
                cap.release()
                if ret:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    path = os.path.join(self._tmp_dir, f"prompt_{prompt_name}.png")
                    Image.fromarray(rgb).save(path)
                    return path
        fallback = str(SAMPLE_DIR / "sample_image_0.png")
        return fallback

    async def initialize(self, ws: WebSocket, prompt_name: str = "default"):
        await ws.send_json({"type": "status", "message": "Loading Oasis model..."})
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_engine)
        self._init_memflow()

        prompt_path = self._extract_first_frame(prompt_name)
        self._bl_prompt_path = prompt_path
        self._mf_prompt_path = prompt_path
        self.prompt_path = prompt_path

        img = np.array(Image.open(prompt_path))
        b64 = frame_to_jpeg_b64(img)

        await ws.send_json({
            "type": "init_done",
            "prompt_name": prompt_name,
            "prompt_image": b64,
        })

    async def generate_batch(self, ws: WebSocket, actions: list[list[float]]):
        """Generate frames for both baseline and MemFlow, streaming to client."""
        if not self.engine or not actions:
            return

        n = len(actions)
        from wm_platform.engines.worker_protocol import decode_frame

        # --- BASELINE (without MemFlow) ---
        await ws.send_json({"type": "status", "message": f"Generating {n} frames (without MemFlow)..."})

        loop = asyncio.get_event_loop()

        def _gen_baseline():
            resp = self.engine._client.send_command(
                "generate_interactive",
                prompt_path=self._bl_prompt_path,
                actions=actions,
            )
            return resp

        resp_bl = await loop.run_in_executor(None, _gen_baseline)

        if resp_bl["status"] != "ok":
            await ws.send_json({"type": "error", "message": resp_bl.get("error", "Generation failed")})
            return

        bl_frames = [decode_frame(b64) for b64 in resp_bl["frames"]]

        for i, rgb in enumerate(bl_frames):
            b64 = frame_to_jpeg_b64(rgb)
            await ws.send_json({
                "type": "frame",
                "side": "baseline",
                "idx": self.total_frames_generated + i,
                "image": b64,
            })
            await asyncio.sleep(0.05)

        bl_last = bl_frames[-1]
        bl_prompt = os.path.join(self._tmp_dir, f"bl_prompt_{self.frame_count}.png")
        save_frame_as_prompt(bl_last, bl_prompt)
        self._bl_prompt_path = bl_prompt

        # --- WITH MEMFLOW ---
        await ws.send_json({"type": "status", "message": f"Generating {n} frames (with MemFlow)..."})

        def _gen_memflow():
            self.engine.reset()
            resp = self.engine._client.send_command(
                "generate_interactive",
                prompt_path=self._mf_prompt_path,
                actions=actions,
            )
            return resp

        resp_mf = await loop.run_in_executor(None, _gen_memflow)

        if resp_mf["status"] != "ok":
            await ws.send_json({"type": "error", "message": resp_mf.get("error", "Generation failed")})
            return

        mf_frames = [decode_frame(b64) for b64 in resp_mf["frames"]]

        from wm_platform.memflow.types import Observation

        for i, rgb in enumerate(mf_frames):
            obs = Observation(
                frame_idx=self.total_frames_generated + i,
                timestamp=time.time(),
                rgb=rgb,
                engine_id="oasis",
            )
            scene = self._memflow_extractor.classify_scene(obs)
            self._memflow_memory.ingest_scene(scene)

            if scene.scene_id not in self._memflow_corrector._reference_frames:
                self._memflow_corrector.store_reference_frame(scene.scene_id, rgb)

            b64 = frame_to_jpeg_b64(rgb)
            await ws.send_json({
                "type": "frame",
                "side": "memflow",
                "idx": self.total_frames_generated + i,
                "image": b64,
            })
            await asyncio.sleep(0.05)

        mf_last = mf_frames[-1]
        mf_prompt = os.path.join(self._tmp_dir, f"mf_prompt_{self.frame_count}.png")
        save_frame_as_prompt(mf_last, mf_prompt)
        self._mf_prompt_path = mf_prompt

        self.total_frames_generated += n
        self.frame_count += 1

        stats = self._memflow_memory.stats()
        snapshot = self._memflow_memory.snapshot()
        await ws.send_json({
            "type": "memory_update",
            "stats": stats,
            "prompt_text": snapshot.to_prompt()[:300],
            "total_frames": self.total_frames_generated,
        })

        await ws.send_json({"type": "status", "message": "Ready - press WASD to explore"})

    def cleanup(self):
        if self.engine:
            try:
                self.engine.unload()
            except Exception:
                pass


async def interactive_ws_handler(ws: WebSocket):
    """WebSocket handler for the interactive demo."""
    await ws.accept()
    session = InteractiveSession()

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")

            if msg_type == "init":
                prompt = data.get("prompt", "default")
                await session.initialize(ws, prompt)

            elif msg_type == "action":
                key = data.get("key", "")
                vec = key_to_action_vector(key)
                session.action_queue.append(vec)
                await ws.send_json({
                    "type": "action_queued",
                    "key": key,
                    "queue_size": len(session.action_queue),
                })

                batch_size = data.get("batch_size", 8)
                if len(session.action_queue) >= batch_size:
                    actions = session.action_queue[:batch_size]
                    session.action_queue = session.action_queue[batch_size:]
                    await session.generate_batch(ws, actions)

            elif msg_type == "generate":
                if session.action_queue:
                    actions = session.action_queue[:]
                    session.action_queue.clear()
                    await session.generate_batch(ws, actions)
                else:
                    await ws.send_json({"type": "status", "message": "No actions queued. Press WASD first."})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        session.cleanup()
