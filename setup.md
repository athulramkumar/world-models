# Setup Guide

## Prerequisites

- Linux (tested on Ubuntu 22.04, kernel 6.8)
- Python 3.11+
- NVIDIA GPU with CUDA 12.4 (A100 80GB tested; H100/H200 profiles available)
- `nvidia-smi` working, CUDA toolkit installed
- `ffmpeg` installed (`apt install ffmpeg`) -- required for browser-playable H.264 videos
- `huggingface-cli` installed and logged in (`pip install huggingface-hub && huggingface-cli login`)

## Directory Layout

```
/workspace/world_models/
  setup_envs.sh              # Creates all virtual environments
  download_checkpoints.sh    # Downloads model weights
  requirements.txt           # Platform dependencies (FastAPI, Gradio, etc.)
  hardware_profiles/         # GPU-specific configs (a100_80gb.yaml, h100.yaml, h200.yaml)
  envs/                      # Python virtual environments (one per model + platform)
  repos/                     # Cloned model repositories
  checkpoints/               # Downloaded model weights
  wm_platform/               # Main application code
  test_results/              # Generated test outputs (videos, metadata, frames)
```

## Step 1: Clone Repositories

```bash
mkdir -p /workspace/world_models/repos
cd /workspace/world_models/repos
git clone https://github.com/athulramkumar/mineworld.git
git clone https://github.com/athulramkumar/open-oasis.git
git clone https://github.com/athulramkumar/world_engine.git
```

## Step 2: Create Virtual Environments

Each model requires its own venv because they need different PyTorch versions.

```bash
cd /workspace/world_models
bash setup_envs.sh all
```

Or individually:

```bash
bash setup_envs.sh open-oasis      # torch 2.4.1
bash setup_envs.sh world_engine    # torch 2.5.1 (upgraded to 2.6.0 at runtime for CVE fix)
bash setup_envs.sh mineworld       # torch 2.6.0
bash setup_envs.sh platform        # torch 2.4.1 + FastAPI + Gradio
```

### Environments Created

| Environment     | Path                          | PyTorch | Key Deps                                             |
|:---------------|:------------------------------|:--------|:-----------------------------------------------------|
| `open-oasis`   | `envs/open-oasis/`           | 2.4.1   | einops, diffusers, timm, safetensors, av             |
| `world_engine` | `envs/world_engine/`         | 2.6.0+  | einops, transformers, tensordict, triton, accelerate  |
| `mineworld`    | `envs/mineworld/`            | 2.6.0   | omegaconf, transformers, diffusers, gradio            |
| `platform`     | `envs/platform/`             | 2.4.1   | fastapi, gradio, uvicorn, opencv, numpy, Pillow       |

### Known Patches Applied at Runtime

The World Engine repo code required these patches (already applied in-tree):

1. **`repos/world_engine/src/world_engine.py`**: `torch._dynamo.config.recompile_limit` and `capture_scalar_outputs` wrapped in `try/except AttributeError` for PyTorch version compat.
2. **`repos/world_engine/src/model/kv_cache.py`**: `BlockMask.from_kv_blocks` uses `inspect.signature` to conditionally pass `compute_q_blocks=False` (varies across PyTorch versions).
3. **World Engine PyTorch upgraded** from 2.5.1 to 2.6.0+ inside its venv to fix `torch.load` CVE-2025-32434 security restriction.

## Step 3: Download Model Checkpoints

```bash
bash download_checkpoints.sh all
```

Or individually:

```bash
bash download_checkpoints.sh oasis          # Etched/oasis-500m (safetensors)
bash download_checkpoints.sh world_engine   # Overworld/Waypoint-1-Small (auto-downloads on first use too)
bash download_checkpoints.sh mineworld      # microsoft/mineworld (temporarily unavailable on HF)
```

### Checkpoint Locations

| Model        | Checkpoint Dir                    | Files                                                |
|:------------|:---------------------------------|:-----------------------------------------------------|
| Open-Oasis  | `checkpoints/oasis/`             | `oasis500m.safetensors`, `vit-l-20.safetensors`     |
| World Engine| `checkpoints/world_engine/`      | Auto-downloaded from `Overworld/Waypoint-1-Small`   |
| MineWorld   | `checkpoints/mineworld/`         | Currently empty (HF repo temporarily down)           |

## Step 4: Run the Platform

```bash
cd /workspace/world_models
python3 -m uvicorn wm_platform.app:app --host 0.0.0.0 --port 7860
```

Or:

```bash
python3 -m wm_platform.app --host 0.0.0.0 --port 7860
```

Open `http://localhost:7860` in a browser. The Gradio UI has 4 tabs: Dashboard, Model Explorer, MemFlow, Results.

### API Endpoints

- `GET /api/health` -- health check
- `GET /api/profiles` -- list hardware profiles
- `GET /api/profiles/{name}` -- get profile details
- `GET /api/envs` -- check venv readiness
- `GET /api/gpu` -- live GPU stats (name, VRAM, utilization, temp)

## Step 5: Run Tests

```bash
# Unit tests for engines and MemFlow
python3 -m pytest wm_platform/tests/test_engines.py -v
python3 -m pytest wm_platform/tests/test_memflow_kitchen.py -v
python3 -m pytest wm_platform/tests/test_memflow_characters.py -v

# Full integration test (loads models, generates videos, runs MemFlow comparison)
python3 -m pytest wm_platform/tests/test_full_run.py -v

# Generate long side-by-side comparison videos
python3 -m wm_platform.tests.generate_comparison_videos \
  --oasis-durations 30 60 \
  --we-frames 30 \
  --we-prompt "A cozy Minecraft kitchen with a chest containing a diamond"
```

### Generation Times (A100 80GB)

| Engine       | Frames/sec | 30s video (180f) | 60s video (360f) |
|:------------|:-----------|:-----------------|:-----------------|
| Open-Oasis  | ~1.5       | ~2 min           | ~4 min           |
| World Engine| ~0.04      | ~78 min          | ~156 min         |

## Hardware Profiles

Located in `hardware_profiles/`. Auto-detected by GPU name at startup.

| Profile       | GPU                  | VRAM  | Architecture | Notes                                    |
|:-------------|:---------------------|:------|:-------------|:-----------------------------------------|
| `a100_80gb`  | NVIDIA A100-SXM4-80GB| 80 GB | Ampere       | bf16 preferred for WE, fp16 for Oasis    |
| `h100`       | NVIDIA H100-SXM5-80GB| 80 GB | Hopper       | FP8 tensor cores, can use larger models  |
| `h200`       | NVIDIA H200-SXM-141GB| 141GB | Hopper       | Enough VRAM for batch>1 or multi-model   |

## Troubleshooting

| Issue | Solution |
|:------|:---------|
| `No module named 'tensordict'` | Install in WE venv: `envs/world_engine/bin/pip install tensordict` |
| Worker process hangs during model load | The stderr pipe buffer fills. Fixed by the background `_drain_stderr` thread in `worker_protocol.py` |
| Videos don't play in browser | Must be H.264 encoded. Use `ffmpeg -i input.mp4 -c:v libx264 -pix_fmt yuv420p output.mp4` |
| `torch.load` security error | Upgrade torch to 2.6.0+ in the affected venv |
| `BlockMask.from_kv_blocks` unexpected kwarg | Already patched in `kv_cache.py` with `inspect.signature` check |
| World Engine `CtrlInput` type error | Button values must be `Set[int]`, already patched in `world_engine_worker.py` |
