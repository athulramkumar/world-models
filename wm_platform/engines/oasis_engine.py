"""Open-Oasis engine adapter -- Diffusion Transformer world model for Minecraft."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import BaseWorldEngine, EngineState, Frame
from .worker_protocol import EngineWorkerClient, decode_frame
from ..config import CHECKPOINTS_DIR, ENV_REGISTRY, REPOS_DIR

OASIS_ACTION_KEYS = [
    "inventory", "ESC", "hotbar.1", "hotbar.2", "hotbar.3", "hotbar.4",
    "hotbar.5", "hotbar.6", "hotbar.7", "hotbar.8", "hotbar.9",
    "forward", "back", "left", "right", "cameraX", "cameraY",
    "jump", "sneak", "sprint", "swapHands", "attack", "use", "pickItem", "drop",
]


class OasisEngine(BaseWorldEngine):
    """Wraps the Open-Oasis DiT + ViT-VAE for Minecraft generation."""

    def __init__(self):
        repo = REPOS_DIR / "open-oasis"
        ckpt_dir = CHECKPOINTS_DIR / "oasis"
        super().__init__("open_oasis", repo, ckpt_dir)
        env = ENV_REGISTRY["open_oasis"]
        worker_script = Path(__file__).parent / "oasis_worker.py"
        worker_env = {**os.environ, "OASIS_REPO": str(repo)}
        self._client = EngineWorkerClient(env.python_bin, worker_script, env=worker_env)
        self._generated_frames: list[np.ndarray] = []

    def load(self, model_variant: Optional[str] = None, **kwargs) -> None:
        self._status.state = EngineState.LOADING

        oasis_ckpt = self.checkpoint_dir / "oasis500m.safetensors"
        vae_ckpt = self.checkpoint_dir / "vit-l-20.safetensors"

        if not oasis_ckpt.exists() or not vae_ckpt.exists():
            self._status.state = EngineState.ERROR
            self._status.error = (
                f"Oasis checkpoints not found in {self.checkpoint_dir}. "
                "Run: huggingface-cli download Etched/oasis-500m oasis500m.safetensors "
                "&& huggingface-cli download Etched/oasis-500m vit-l-20.safetensors"
            )
            return

        ddim = kwargs.get("ddim_steps", 10)
        self._client.start()
        resp = self._client.send_command(
            "load",
            oasis_ckpt=str(oasis_ckpt),
            vae_ckpt=str(vae_ckpt),
            ddim_steps=ddim,
        )
        if resp["status"] == "ok":
            self._status.state = EngineState.READY
            self._status.model_name = "Oasis 500M (DiT-S/2)"
        else:
            self._status.state = EngineState.ERROR
            self._status.error = resp.get("error", "Load failed")

    def generate_video(
        self,
        prompt_path: str,
        actions_path: str,
        total_frames: int = 32,
        n_prompt_frames: int = 1,
    ) -> list[Frame]:
        """Generate a sequence of frames (batch mode, matching Oasis's design)."""
        self._status.state = EngineState.GENERATING
        t0 = time.time()

        resp = self._client.send_command(
            "generate_video",
            prompt_path=prompt_path,
            actions_path=actions_path,
            total_frames=total_frames,
            n_prompt_frames=n_prompt_frames,
        )
        elapsed = (time.time() - t0) * 1000
        self._status.last_gen_time_ms = elapsed
        self._status.state = EngineState.READY

        if resp["status"] != "ok":
            raise RuntimeError(resp.get("error", "Generation failed"))

        frames = []
        self._generated_frames.clear()
        for b64 in resp["frames"]:
            rgb = decode_frame(b64)
            self._generated_frames.append(rgb)
            frames.append(self._make_frame(rgb, engine="open_oasis"))
        return frames

    def generate_frame(self, actions: dict[str, Any]) -> Frame:
        """Single-frame stub -- Oasis is batch-oriented, so this generates 2 frames
        and returns the last one."""
        raise NotImplementedError(
            "Oasis generates videos in batch. Use generate_video() instead."
        )

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
        self._generated_frames.clear()

    def unload(self) -> None:
        self._client.stop()
        super().unload()

    @staticmethod
    def available_actions() -> list[str]:
        return OASIS_ACTION_KEYS
