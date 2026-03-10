#!/usr/bin/env python3
"""Worker process for LingBot-World -- runs inside the lingbot-world venv.

Generates video from image + text prompt + optional WASD actions/camera poses.
Unlike Oasis (frame-by-frame), LingBot-World produces all frames at once via
diffusion over the full latent sequence.
"""

import json
import os
import sys
import tempfile

import numpy as np

REPO_DIR = os.environ.get(
    "LINGBOT_REPO",
    os.path.join(os.path.dirname(__file__), "..", "..", "repos", "lingbot-world"),
)
sys.path.insert(0, REPO_DIR)

_pipeline = None
_loaded = False
_device = "cuda:0"
_config = None
_ckpt_dir = None


def handle(req: dict) -> dict:
    global _pipeline, _loaded, _config, _ckpt_dir

    cmd = req["cmd"]

    if cmd == "load":
        import torch
        from wan.configs import WAN_CONFIGS
        from wan.image2video import WanI2V

        ckpt_dir = req["ckpt_dir"]
        t5_cpu = req.get("t5_cpu", True)

        cfg = WAN_CONFIGS["i2v-A14B"]
        _config = cfg
        _ckpt_dir = ckpt_dir

        _pipeline = WanI2V(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=0,
            rank=0,
            t5_cpu=t5_cpu,
            init_on_cpu=True,
        )
        _loaded = True
        return {"status": "ok", "fps": cfg.sample_fps}

    if cmd == "generate_video":
        if not _loaded:
            return {"status": "error", "error": "Model not loaded"}

        import torch
        from PIL import Image

        image_path = req["image_path"]
        prompt = req.get("prompt", "")
        frame_num = req.get("frame_num", 81)
        action_path = req.get("action_path", None)
        seed = req.get("seed", 42)
        shift = req.get("shift", 3.0)
        sampling_steps = req.get("sampling_steps", 40)
        guide_scale = req.get("guide_scale", 5.0)
        max_area = req.get("max_area", 480 * 832)

        # If inline actions/poses provided, write to temp dir
        if action_path is None and ("actions" in req or "poses" in req):
            action_path = _write_temp_actions(req, frame_num)

        img = Image.open(image_path).convert("RGB")

        video = _pipeline.generate(
            input_prompt=prompt,
            img=img,
            action_path=action_path,
            max_area=max_area,
            frame_num=frame_num,
            shift=shift,
            sample_solver="unipc",
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            seed=seed,
            offload_model=True,
        )

        # video shape: (C, F, H, W) in [-1, 1]
        video = video.float().cpu()
        video = (video + 1) / 2  # → [0, 1]
        video = video.clamp(0, 1)

        # Rearrange to (F, H, W, C)
        frames_np = video.permute(1, 2, 3, 0).numpy()
        frames_np = (frames_np * 255).astype(np.uint8)

        from wm_platform.engines.worker_protocol import encode_frame

        encoded = [encode_frame(f) for f in frames_np]

        torch.cuda.empty_cache()

        return {
            "status": "ok",
            "frames": encoded,
            "n_frames": len(encoded),
            "shape": list(frames_np[0].shape),
            "fps": _config.sample_fps,
        }

    if cmd == "reset":
        import torch
        torch.cuda.empty_cache()
        return {"status": "ok"}

    if cmd == "status":
        return {"status": "ok", "loaded": _loaded}

    return {"status": "error", "error": f"Unknown command: {cmd}"}


def _write_temp_actions(req: dict, frame_num: int = 81) -> str:
    """Write inline action/pose arrays to a temp directory for WanI2V.

    If poses/intrinsics are not provided, generates static (identity)
    versions so that WASD actions still work.
    """
    tmpdir = tempfile.mkdtemp(prefix="lingbot_actions_")

    if "poses" in req:
        poses = np.array(req["poses"], dtype=np.float32)
    else:
        poses = np.zeros((frame_num, 4, 4), dtype=np.float32)
        for i in range(frame_num):
            poses[i] = np.eye(4, dtype=np.float32)
    np.save(os.path.join(tmpdir, "poses.npy"), poses)

    if "intrinsics" in req:
        intrinsics = np.array(req["intrinsics"], dtype=np.float32)
    else:
        intrinsics = np.tile(
            np.array([502.9, 503.1, 415.8, 239.8], dtype=np.float32),
            (frame_num, 1),
        )
    np.save(os.path.join(tmpdir, "intrinsics.npy"), intrinsics)

    if "actions" in req:
        actions = np.array(req["actions"], dtype=np.int32)
    else:
        actions = np.zeros((frame_num, 4), dtype=np.int32)
    np.save(os.path.join(tmpdir, "action.npy"), actions)

    return tmpdir


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, root)
    from wm_platform.engines.worker_protocol import worker_main_loop

    worker_main_loop(handle)
