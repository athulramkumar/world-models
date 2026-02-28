#!/usr/bin/env python3
"""Worker process for World Engine -- runs inside the world_engine venv."""

import json
import sys
import os

import numpy as np


def _setup_paths():
    repo = os.environ.get("WORLD_ENGINE_REPO", "")
    if repo and os.path.isdir(repo):
        sys.path.insert(0, repo)


_setup_paths()

engine = None
_loaded = False
_last_latent = None


def handle(req: dict) -> dict:
    global engine, _loaded, _last_latent

    cmd = req["cmd"]

    if cmd == "load":
        import torch

        model_uri = req.get("model_uri", "Overworld/Waypoint-1-Small")
        device = req.get("device", "cuda")
        quant = req.get("quantization", None)

        try:
            from src import WorldEngine

            engine = WorldEngine(model_uri, quant=quant, device=device)
            _loaded = True
            return {"status": "ok", "model_uri": model_uri}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    if cmd == "set_prompt":
        if not _loaded:
            return {"status": "error", "error": "Engine not loaded"}
        try:
            engine.set_prompt(req["prompt"])
            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    if cmd == "generate_frame":
        if not _loaded:
            return {"status": "error", "error": "Engine not loaded"}

        import torch
        from src import CtrlInput

        ctrl_data = req.get("ctrl", {})
        raw_buttons = ctrl_data.get("button", [])
        int_buttons = set()
        for b in raw_buttons:
            if isinstance(b, int):
                int_buttons.add(b)
            elif isinstance(b, str) and b.isdigit():
                int_buttons.add(int(b))
        ctrl = CtrlInput(
            button=int_buttons,
            mouse=tuple(ctrl_data.get("mouse", [0.0, 0.0])),
            scroll_wheel=ctrl_data.get("scroll_wheel", 0),
        )

        img = engine.gen_frame(ctrl=ctrl, return_img=True)

        if hasattr(img, "cpu"):
            img_np = img.cpu().numpy()
        else:
            img_np = np.array(img)

        if img_np.dtype != np.uint8:
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            else:
                img_np = img_np.clip(0, 255).astype(np.uint8)

        from wm_platform.engines.worker_protocol import encode_frame

        return {
            "status": "ok",
            "frame": encode_frame(img_np),
            "shape": list(img_np.shape),
        }

    if cmd == "append_frame":
        if not _loaded:
            return {"status": "error", "error": "Engine not loaded"}

        import torch
        from wm_platform.engines.worker_protocol import decode_frame
        from src import CtrlInput

        img_np = decode_frame(req["frame"])
        img_t = torch.from_numpy(img_np).to(engine.device)
        ctrl_data = req.get("ctrl", {})
        raw_buttons = ctrl_data.get("button", [])
        int_buttons = set()
        for b in raw_buttons:
            if isinstance(b, int):
                int_buttons.add(b)
            elif isinstance(b, str) and b.isdigit():
                int_buttons.add(int(b))
        ctrl = CtrlInput(
            button=int_buttons,
            mouse=tuple(ctrl_data.get("mouse", [0.0, 0.0])),
        )
        engine.append_frame(img_t, ctrl=ctrl)
        return {"status": "ok"}

    if cmd == "get_latents":
        return {"status": "ok", "latent": None}

    if cmd == "reset":
        if _loaded:
            engine.reset()
        return {"status": "ok"}

    if cmd == "status":
        return {"status": "ok", "loaded": _loaded}

    return {"status": "error", "error": f"Unknown command: {cmd}"}


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, root)
    from wm_platform.engines.worker_protocol import worker_main_loop

    worker_main_loop(handle)
