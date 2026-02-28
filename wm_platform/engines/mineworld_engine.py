"""MineWorld engine adapter -- Llama-based autoregressive world model for Minecraft."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import BaseWorldEngine, EngineState, EngineStatus, Frame
from .worker_protocol import EngineWorkerClient, decode_frame, encode_frame
from ..config import CHECKPOINTS_DIR, ENV_REGISTRY, REPOS_DIR


MINEWORLD_ACTIONS = [
    "forward", "back", "left", "right", "jump", "sprint", "sneak",
    "attack", "use", "drop", "swapHands", "pickItem",
] + [f"hotbar.{i}" for i in range(1, 10)]


class MineWorldEngine(BaseWorldEngine):
    """Wraps the MineWorld Llama transformer + VQGAN tokenizer."""

    def __init__(self):
        repo = REPOS_DIR / "mineworld"
        ckpt_dir = CHECKPOINTS_DIR / "mineworld"
        super().__init__("mineworld", repo, ckpt_dir)
        env = ENV_REGISTRY["mineworld"]
        worker_script = Path(__file__).parent / "mineworld_worker.py"
        worker_env = {
            **os.environ,
            "MINEWORLD_REPO": str(repo),
        }
        self._client = EngineWorkerClient(env.python_bin, worker_script, env=worker_env)
        self._context_len = 0

    def load(self, model_variant: Optional[str] = "700M_16f", **kwargs) -> None:
        self._status.state = EngineState.LOADING
        variant = model_variant or "700M_16f"
        config_path = self.repo_path / "configs" / f"{variant}.yaml"
        ckpt_path = self.checkpoint_dir / f"{variant}.ckpt"

        if not ckpt_path.exists():
            self._status.state = EngineState.ERROR
            self._status.error = (
                f"Checkpoint {ckpt_path} not found. MineWorld checkpoints are "
                "temporarily unavailable on HuggingFace. Place them in "
                f"{self.checkpoint_dir}/ when available."
            )
            return

        self._client.start()
        resp = self._client.send_command(
            "load", config=str(config_path), checkpoint=str(ckpt_path)
        )
        if resp["status"] == "ok":
            self._status.state = EngineState.READY
            self._status.model_name = f"MineWorld {variant}"
            self._context_len = resp.get("context_len", 16)
        else:
            self._status.state = EngineState.ERROR
            self._status.error = resp.get("error", "Unknown load error")

    def generate_frame(self, actions: dict[str, Any]) -> Frame:
        self._status.state = EngineState.GENERATING
        t0 = time.time()

        action_dict = {k: actions.get(k, 0) for k in MINEWORLD_ACTIONS}
        action_dict["camera"] = actions.get("camera", [0, 0])

        resp = self._client.send_command("generate_frame", action=action_dict)
        elapsed = (time.time() - t0) * 1000
        self._status.last_gen_time_ms = elapsed
        self._status.state = EngineState.READY

        if resp["status"] != "ok":
            raise RuntimeError(resp.get("error", "Generation failed"))

        rgb = decode_frame(resp["frame"])
        return self._make_frame(rgb, engine="mineworld")

    def get_latents(self) -> Optional[np.ndarray]:
        resp = self._client.send_command("get_latents")
        if resp.get("latent"):
            return decode_frame(resp["latent"])
        return None

    def reset(self) -> None:
        if self._client.alive:
            self._client.send_command("reset")
        self._frame_counter = 0
        self._context_frames.clear()

    def unload(self) -> None:
        self._client.stop()
        super().unload()

    @staticmethod
    def available_actions() -> list[str]:
        return MINEWORLD_ACTIONS + ["camera"]
