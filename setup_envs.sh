#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ENVS_DIR="$ROOT/envs"
REPOS_DIR="$ROOT/repos"

CUDA_TAG="cu124"
PIP_TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"

log() { echo -e "\n===== $1 =====\n"; }

# ------------------------------------------------------------------ #
# 1. MineWorld  (torch 2.6, gradio, omegaconf, diffusers …)
# ------------------------------------------------------------------ #
setup_mineworld() {
    log "Setting up mineworld venv"
    local VENV="$ENVS_DIR/mineworld"
    python3 -m venv "$VENV"
    source "$VENV/bin/activate"

    pip install --upgrade pip setuptools wheel
    pip install torch==2.6.0 torchvision==0.21.0 --index-url "$PIP_TORCH_INDEX"
    pip install \
        omegaconf==2.3.0 \
        transformers==4.48.1 \
        "opencv-python==4.11.0.86" \
        attrs==25.3.0 \
        diffusers==0.32.2 \
        gradio==5.24.0 \
        einops==0.8.1 \
        scipy==1.15.2 \
        torch-fidelity==0.3.0 \
        scikit-learn==1.6.1

    echo "[mineworld] Verifying torch + CUDA …"
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'torch {torch.__version__}  CUDA {torch.version.cuda}  GPU {torch.cuda.get_device_name(0)}')"
    deactivate
    log "mineworld venv ready at $VENV"
}

# ------------------------------------------------------------------ #
# 2. Open-Oasis  (torch 2.4, einops, diffusers, timm, av)
# ------------------------------------------------------------------ #
setup_oasis() {
    log "Setting up open-oasis venv"
    local VENV="$ENVS_DIR/open-oasis"
    python3 -m venv "$VENV"
    source "$VENV/bin/activate"

    pip install --upgrade pip setuptools wheel
    pip install torch==2.4.1 torchvision==0.19.1 --index-url "$PIP_TORCH_INDEX"
    pip install \
        einops \
        diffusers \
        timm \
        safetensors \
        av

    echo "[open-oasis] Verifying torch + CUDA …"
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'torch {torch.__version__}  CUDA {torch.version.cuda}  GPU {torch.cuda.get_device_name(0)}')"
    deactivate
    log "open-oasis venv ready at $VENV"
}

# ------------------------------------------------------------------ #
# 3. World Engine  (torch 2.5, transformers, triton …)
# ------------------------------------------------------------------ #
setup_world_engine() {
    log "Setting up world_engine venv"
    local VENV="$ENVS_DIR/world_engine"
    python3 -m venv "$VENV"
    source "$VENV/bin/activate"

    pip install --upgrade pip setuptools wheel
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url "$PIP_TORCH_INDEX"
    pip install \
        einops \
        "rotary-embedding-torch>=0.8.8" \
        tensordict==0.6.2 \
        transformers==4.46.3 \
        ftfy \
        diffusers \
        "huggingface-hub>=0.26.0" \
        omegaconf \
        accelerate==1.1.1 \
        triton \
        safetensors

    echo "[world_engine] Verifying torch + CUDA …"
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'torch {torch.__version__}  CUDA {torch.version.cuda}  GPU {torch.cuda.get_device_name(0)}')"
    deactivate
    log "world_engine venv ready at $VENV"
}

# ------------------------------------------------------------------ #
# 4. Platform venv (FastAPI, Gradio, shared utilities)
# ------------------------------------------------------------------ #
setup_platform() {
    log "Setting up platform venv"
    local VENV="$ENVS_DIR/platform"
    python3 -m venv "$VENV"
    source "$VENV/bin/activate"

    pip install --upgrade pip setuptools wheel
    pip install torch==2.4.1 torchvision==0.19.1 --index-url "$PIP_TORCH_INDEX"
    pip install -r "$ROOT/requirements.txt" 2>&1

    echo "[platform] Verifying torch + CUDA …"
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'torch {torch.__version__}  CUDA {torch.version.cuda}  GPU {torch.cuda.get_device_name(0)}')"
    deactivate
    log "platform venv ready at $VENV"
}

# ------------------------------------------------------------------ #
# Run all
# ------------------------------------------------------------------ #
main() {
    mkdir -p "$ENVS_DIR"
    local target="${1:-all}"
    case "$target" in
        mineworld)     setup_mineworld ;;
        open-oasis)    setup_oasis ;;
        world_engine)  setup_world_engine ;;
        platform)      setup_platform ;;
        all)
            setup_mineworld
            setup_oasis
            setup_world_engine
            setup_platform
            ;;
        *)
            echo "Usage: $0 {mineworld|open-oasis|world_engine|platform|all}"
            exit 1
            ;;
    esac
    log "All requested environments ready"
}

main "$@"
