"""World Engine adapter -- Overworld inference engine for general world models."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import BaseWorldEngine, EngineState, Frame
from .worker_protocol import EngineWorkerClient, decode_frame, encode_frame
from ..config import CHECKPOINTS_DIR, ENV_REGISTRY, REPOS_DIR


class WorldEngineAdapter(BaseWorldEngine):
    """Wraps the Overworld WorldEngine (DiT + autoencoder + text encoder)."""

    def __init__(self):
        repo = REPOS_DIR / "world_engine"
        ckpt_dir = CHECKPOINTS_DIR / "world_engine"
        super().__init__("world_engine", repo, ckpt_dir)
        env = ENV_REGISTRY["world_engine"]
        worker_script = Path(__file__).parent / "world_engine_worker.py"
        worker_env = {**os.environ, "WORLD_ENGINE_REPO": str(repo)}
        self._client = EngineWorkerClient(env.python_bin, worker_script, env=worker_env)
        self._prompt: str = "An explorable world"

    def load(self, model_variant: Optional[str] = None, **kwargs) -> None:
        self._status.state = EngineState.LOADING

        model_uri = model_variant or kwargs.get("model_uri", "Overworld/Waypoint-1-Small")
        quant = kwargs.get("quantization", None)

        self._client.start()
        resp = self._client.send_command(
            "load", model_uri=model_uri, quantization=quant
        )
        if resp["status"] == "ok":
            self._status.state = EngineState.READY
            self._status.model_name = f"WorldEngine ({resp.get('model_uri', model_uri)})"
        else:
            self._status.state = EngineState.ERROR
            self._status.error = resp.get("error", "Load failed")

    def set_prompt(self, prompt: str) -> None:
        self._prompt = prompt
        if self._client.alive:
            self._client.send_command("set_prompt", prompt=prompt)

    def generate_frame(self, actions: dict[str, Any]) -> Frame:
        self._status.state = EngineState.GENERATING
        t0 = time.time()

        ctrl = {
            "button": list(actions.get("button", [])),
            "mouse": list(actions.get("mouse", [0.0, 0.0])),
            "scroll_wheel": actions.get("scroll_wheel", 0),
        }

        resp = self._client.send_command("generate_frame", ctrl=ctrl)
        elapsed = (time.time() - t0) * 1000
        self._status.last_gen_time_ms = elapsed
        self._status.state = EngineState.READY

        if resp["status"] != "ok":
            raise RuntimeError(resp.get("error", "Generation failed"))

        rgb = decode_frame(resp["frame"])
        return self._make_frame(rgb, engine="world_engine")

    def append_frame_from_image(self, img: np.ndarray, ctrl: Optional[dict] = None) -> None:
        encoded = encode_frame(img)
        self._client.send_command("append_frame", frame=encoded, ctrl=ctrl or {})

    def get_latents(self) -> Optional[np.ndarray]:
        resp = self._client.send_command("get_latents")
        if resp.get("latent"):
            return decode_frame(resp["latent"])
        return None

    def inject_conditioning(self, memory_state: Any) -> None:
        """Use prompt conditioning to reinject memory state."""
        if hasattr(memory_state, "to_prompt"):
            self.set_prompt(memory_state.to_prompt())

    def reset(self) -> None:
        if self._client.alive:
            self._client.send_command("reset")
        self._frame_counter = 0
        self._context_frames.clear()

    def unload(self) -> None:
        self._client.stop()
        super().unload()
