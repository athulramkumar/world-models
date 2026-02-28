#!/usr/bin/env python3
"""Worker process for MineWorld -- runs inside the mineworld venv."""

import json
import sys
import os

import numpy as np


def _setup_paths():
    repo = os.environ.get("MINEWORLD_REPO", "")
    if repo:
        sys.path.insert(0, repo)


_setup_paths()

model = None
tokenizer = None
mc_action_map = None
frame_cache = []
action_cache = []
last_pos = 0
_loaded = False
TOKEN_PER_IMAGE = 336
TOKEN_PER_ACTION = 11


def handle(req: dict) -> dict:
    global model, tokenizer, mc_action_map, frame_cache, action_cache, last_pos, _loaded

    cmd = req["cmd"]

    if cmd == "load":
        import torch
        from omegaconf import OmegaConf
        from utils import load_model
        from mcdataset import MCDataset

        config_path = req["config"]
        ckpt_path = req["checkpoint"]

        if not os.path.isfile(ckpt_path):
            return {"status": "error", "error": f"Checkpoint not found: {ckpt_path}"}

        config = OmegaConf.load(config_path)
        model = load_model(config, ckpt_path, gpu=True, eval_mode=True)
        tokenizer = model.tokenizer
        mc_action_map = MCDataset()
        frame_cache = []
        action_cache = []
        last_pos = 0
        _loaded = True
        context_len = int(
            config.model.params.transformer_config.params.max_position_embeddings
            / (TOKEN_PER_ACTION + TOKEN_PER_IMAGE)
        )
        return {"status": "ok", "context_len": context_len}

    if cmd == "generate_frame":
        if not _loaded:
            return {"status": "error", "error": "Model not loaded"}

        import torch
        from collections import deque
        from mcdataset import MCDataset

        action_dict = req["action"]
        action_dict["camera"] = np.array(action_dict.get("camera", [0, 0]))

        ongoing_act = mc_action_map.get_action_index_from_actiondict(
            action_dict, action_vocab_offset=8192
        )
        ongoing_act = torch.tensor(ongoing_act).unsqueeze(0).to("cuda")
        action_cache.append(ongoing_act)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            next_frame_tokens, new_pos = model.transformer.decode_img_token_for_gradio(
                input_action=ongoing_act,
                position_id=last_pos,
                max_new_tokens=TOKEN_PER_IMAGE + 1,
            )

        last_pos = new_pos[0]
        next_frame_tokens = torch.cat(next_frame_tokens, dim=-1).to("cuda")
        frame_cache.append(next_frame_tokens)

        rgb = tokenizer.token2image(next_frame_tokens)  # uint8 numpy HxWx3

        from wm_platform.engines.worker_protocol import encode_frame

        return {
            "status": "ok",
            "frame": encode_frame(rgb),
            "frame_shape": list(rgb.shape),
        }

    if cmd == "get_latents":
        if not frame_cache:
            return {"status": "ok", "latent": None}
        import torch

        last = frame_cache[-1]
        latent_np = last.cpu().float().numpy()
        from wm_platform.engines.worker_protocol import encode_frame

        return {"status": "ok", "latent": encode_frame(latent_np)}

    if cmd == "reset":
        if _loaded:
            import torch

            model.transformer.refresh_kvcache()
        frame_cache.clear()
        action_cache.clear()
        last_pos = 0
        return {"status": "ok"}

    if cmd == "status":
        return {"status": "ok", "loaded": _loaded, "frames": len(frame_cache)}

    return {"status": "error", "error": f"Unknown command: {cmd}"}


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, root)
    from wm_platform.engines.worker_protocol import worker_main_loop

    worker_main_loop(handle)
