#!/usr/bin/env python3
"""Worker process for Open-Oasis -- runs inside the open-oasis venv."""

import json
import sys
import os

import numpy as np


def _setup_paths():
    repo = os.environ.get("OASIS_REPO", "")
    if repo:
        sys.path.insert(0, repo)


_setup_paths()

dit_model = None
vae_model = None
_loaded = False
_device = "cuda:0"
_scaling_factor = 0.07843137255
_max_noise_level = 1000
_ddim_steps = 10
_stabilization_level = 15
_noise_abs_max = 20
_alphas_cumprod = None
_noise_range = None
_latent_buffer = None


def handle(req: dict) -> dict:
    global dit_model, vae_model, _loaded, _alphas_cumprod, _noise_range, _latent_buffer

    cmd = req["cmd"]

    if cmd == "load":
        import torch
        from dit import DiT_models
        from vae import VAE_models
        from utils import sigmoid_beta_schedule
        from safetensors.torch import load_model

        oasis_ckpt = req["oasis_ckpt"]
        vae_ckpt = req["vae_ckpt"]
        ddim = req.get("ddim_steps", 10)

        if not os.path.isfile(oasis_ckpt):
            return {"status": "error", "error": f"DiT checkpoint not found: {oasis_ckpt}"}
        if not os.path.isfile(vae_ckpt):
            return {"status": "error", "error": f"VAE checkpoint not found: {vae_ckpt}"}

        dit = DiT_models["DiT-S/2"]()
        if oasis_ckpt.endswith(".safetensors"):
            load_model(dit, oasis_ckpt)
        else:
            ckpt = torch.load(oasis_ckpt, weights_only=True)
            dit.load_state_dict(ckpt, strict=False)
        dit_model = dit.to(_device).eval()

        vae = VAE_models["vit-l-20-shallow-encoder"]()
        if vae_ckpt.endswith(".safetensors"):
            load_model(vae, vae_ckpt)
        else:
            vae_ckpt_data = torch.load(vae_ckpt, weights_only=True)
            vae.load_state_dict(vae_ckpt_data)
        vae_model = vae.to(_device).eval()

        global _ddim_steps
        _ddim_steps = ddim
        _noise_range = torch.linspace(-1, _max_noise_level - 1, _ddim_steps + 1)
        betas = sigmoid_beta_schedule(_max_noise_level).float().to(_device)
        alphas = 1.0 - betas
        _alphas_cumprod = torch.cumprod(alphas, dim=0)

        _loaded = True
        return {"status": "ok"}

    if cmd == "generate_video":
        if not _loaded:
            return {"status": "error", "error": "Model not loaded"}

        import torch
        from torch import autocast
        from einops import rearrange
        from utils import load_prompt, load_actions

        prompt_path = req["prompt_path"]
        actions_path = req["actions_path"]
        n_prompt = req.get("n_prompt_frames", 1)
        total = req.get("total_frames", 32)
        video_offset = req.get("video_offset", None)

        x = load_prompt(prompt_path, video_offset=video_offset, n_prompt_frames=n_prompt)
        actions = load_actions(actions_path, action_offset=video_offset)[:, :total]
        x, actions = x.to(_device), actions.to(_device)

        B = x.shape[0]
        H, W = x.shape[-2:]
        x = rearrange(x, "b t c h w -> (b t) c h w")
        with torch.no_grad(), autocast("cuda", dtype=torch.half):
            x = vae_model.encode(x * 2 - 1).mean * _scaling_factor
        x = rearrange(
            x,
            "(b t) (h w) c -> b t c h w",
            t=n_prompt,
            h=H // vae_model.patch_size,
            w=W // vae_model.patch_size,
        )

        ac = rearrange(_alphas_cumprod, "T -> T 1 1 1")

        for i in range(n_prompt, total):
            chunk = torch.randn((B, 1, *x.shape[-3:]), device=_device)
            chunk = torch.clamp(chunk, -_noise_abs_max, _noise_abs_max)
            x = torch.cat([x, chunk], dim=1)
            start = max(0, i + 1 - dit_model.max_frames)

            for ni in reversed(range(1, _ddim_steps + 1)):
                t_ctx = torch.full((B, i), _stabilization_level - 1, dtype=torch.long, device=_device)
                t = torch.full((B, 1), _noise_range[ni], dtype=torch.long, device=_device)
                t_next = torch.full((B, 1), _noise_range[ni - 1], dtype=torch.long, device=_device)
                t_next = torch.where(t_next < 0, t, t_next)
                t_full = torch.cat([t_ctx, t], dim=1)[:, start:]
                t_next_full = torch.cat([t_ctx, t_next], dim=1)[:, start:]

                x_curr = x[:, start:].clone()
                with torch.no_grad(), autocast("cuda", dtype=torch.half):
                    v = dit_model(x_curr, t_full, actions[:, start : i + 1])

                x_start = ac[t_full].sqrt() * x_curr - (1 - ac[t_full]).sqrt() * v
                x_noise = ((1 / ac[t_full]).sqrt() * x_curr - x_start) / (1 / ac[t_full] - 1).sqrt()
                a_next = ac[t_next_full]
                a_next[:, :-1] = 1.0
                if ni == 1:
                    a_next[:, -1:] = 1.0
                x_pred = a_next.sqrt() * x_start + x_noise * (1 - a_next).sqrt()
                x[:, -1:] = x_pred[:, -1:]

        _latent_buffer = x.detach().clone()

        x_dec = rearrange(x, "b t c h w -> (b t) (h w) c")
        with torch.no_grad():
            x_dec = (vae_model.decode(x_dec / _scaling_factor) + 1) / 2
        x_dec = rearrange(x_dec, "(b t) c h w -> b t h w c", t=total)
        frames = torch.clamp(x_dec, 0, 1)
        frames = (frames * 255).byte().cpu().numpy()[0]

        from wm_platform.engines.worker_protocol import encode_frame

        encoded = [encode_frame(f) for f in frames]
        return {
            "status": "ok",
            "frames": encoded,
            "n_frames": len(encoded),
            "shape": list(frames[0].shape),
        }

    if cmd == "generate_interactive":
        if not _loaded:
            return {"status": "error", "error": "Model not loaded"}

        import torch
        from torch import autocast
        from einops import rearrange
        from utils import load_prompt

        prompt_path = req["prompt_path"]
        action_list = req["actions"]  # list of 25-dim lists
        n_prompt = 1
        total = len(action_list) + 1  # +1 for prompt frame

        x = load_prompt(prompt_path, n_prompt_frames=n_prompt)
        actions_tensor = torch.zeros(1, total, 25)
        for i, a in enumerate(action_list):
            actions_tensor[0, i + 1] = torch.tensor(a, dtype=torch.float32)
        x, actions_tensor = x.to(_device), actions_tensor.to(_device)

        B = x.shape[0]
        H, W = x.shape[-2:]
        x = rearrange(x, "b t c h w -> (b t) c h w")
        with torch.no_grad(), autocast("cuda", dtype=torch.half):
            x = vae_model.encode(x * 2 - 1).mean * _scaling_factor
        x = rearrange(
            x, "(b t) (h w) c -> b t c h w",
            t=n_prompt, h=H // vae_model.patch_size, w=W // vae_model.patch_size,
        )

        ac = rearrange(_alphas_cumprod, "T -> T 1 1 1")

        for i in range(n_prompt, total):
            chunk = torch.randn((B, 1, *x.shape[-3:]), device=_device)
            chunk = torch.clamp(chunk, -_noise_abs_max, _noise_abs_max)
            x = torch.cat([x, chunk], dim=1)
            start = max(0, i + 1 - dit_model.max_frames)

            for ni in reversed(range(1, _ddim_steps + 1)):
                t_ctx = torch.full((B, i), _stabilization_level - 1, dtype=torch.long, device=_device)
                t = torch.full((B, 1), _noise_range[ni], dtype=torch.long, device=_device)
                t_next = torch.full((B, 1), _noise_range[ni - 1], dtype=torch.long, device=_device)
                t_next = torch.where(t_next < 0, t, t_next)
                t_full = torch.cat([t_ctx, t], dim=1)[:, start:]
                t_next_full = torch.cat([t_ctx, t_next], dim=1)[:, start:]

                x_curr = x[:, start:].clone()
                with torch.no_grad(), autocast("cuda", dtype=torch.half):
                    v = dit_model(x_curr, t_full, actions_tensor[:, start : i + 1])

                x_start = ac[t_full].sqrt() * x_curr - (1 - ac[t_full]).sqrt() * v
                x_noise = ((1 / ac[t_full]).sqrt() * x_curr - x_start) / (1 / ac[t_full] - 1).sqrt()
                a_next = ac[t_next_full]
                a_next[:, :-1] = 1.0
                if ni == 1:
                    a_next[:, -1:] = 1.0
                x_pred = a_next.sqrt() * x_start + x_noise * (1 - a_next).sqrt()
                x[:, -1:] = x_pred[:, -1:]

        _latent_buffer = x.detach().clone()

        x_dec = rearrange(x, "b t c h w -> (b t) (h w) c")
        with torch.no_grad():
            x_dec = (vae_model.decode(x_dec / _scaling_factor) + 1) / 2
        x_dec = rearrange(x_dec, "(b t) c h w -> b t h w c", t=total)
        frames = torch.clamp(x_dec, 0, 1)
        frames = (frames * 255).byte().cpu().numpy()[0]

        from wm_platform.engines.worker_protocol import encode_frame
        encoded = [encode_frame(f) for f in frames[n_prompt:]]
        return {
            "status": "ok",
            "frames": encoded,
            "n_frames": len(encoded),
            "shape": list(frames[0].shape),
        }

    if cmd == "get_latents":
        if _latent_buffer is None:
            return {"status": "ok", "latent": None}
        from wm_platform.engines.worker_protocol import encode_frame

        lat = _latent_buffer.cpu().float().numpy()
        return {"status": "ok", "latent": encode_frame(lat)}

    if cmd == "reset":
        _latent_buffer = None
        return {"status": "ok"}

    if cmd == "status":
        return {"status": "ok", "loaded": _loaded}

    return {"status": "error", "error": f"Unknown command: {cmd}"}


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, root)
    from wm_platform.engines.worker_protocol import worker_main_loop

    worker_main_loop(handle)
