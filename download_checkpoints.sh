#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CKPT_DIR="$ROOT/checkpoints"

log() { echo -e "\n===== $1 =====\n"; }

# ------------------------------------------------------------------ #
# 1. Open-Oasis (Etched/oasis-500m)
# ------------------------------------------------------------------ #
download_oasis() {
    log "Downloading Open-Oasis checkpoints"
    local dir="$CKPT_DIR/oasis"
    mkdir -p "$dir"

    if [ -f "$dir/oasis500m.safetensors" ] && [ -f "$dir/vit-l-20.safetensors" ]; then
        echo "Oasis checkpoints already present, skipping."
        return
    fi

    echo "Requires: pip install huggingface-hub && huggingface-cli login"
    huggingface-cli download Etched/oasis-500m oasis500m.safetensors --local-dir "$dir"
    huggingface-cli download Etched/oasis-500m vit-l-20.safetensors --local-dir "$dir"
    log "Oasis checkpoints downloaded to $dir"
}

# ------------------------------------------------------------------ #
# 2. World Engine (auto-downloads via HF hub, but we can pre-cache)
# ------------------------------------------------------------------ #
download_world_engine() {
    log "Downloading World Engine checkpoints"
    local dir="$CKPT_DIR/world_engine"
    mkdir -p "$dir"

    echo "World Engine auto-downloads models on first use via HuggingFace Hub."
    echo "To pre-cache: pip install huggingface-hub && huggingface-cli download Overworld/Waypoint-1-Small --local-dir $dir"
    echo "Attempting pre-cache download..."
    huggingface-cli download Overworld/Waypoint-1-Small --local-dir "$dir" || echo "Warning: download failed, model will auto-download on first use"
    log "World Engine checkpoint cache at $dir"
}

# ------------------------------------------------------------------ #
# 3. MineWorld (temporarily unavailable)
# ------------------------------------------------------------------ #
download_mineworld() {
    log "MineWorld checkpoints"
    local dir="$CKPT_DIR/mineworld"
    mkdir -p "$dir"

    echo "NOTE: MineWorld checkpoints on HuggingFace (microsoft/mineworld) are"
    echo "temporarily taken down. When they return, download with:"
    echo ""
    echo "  huggingface-cli download microsoft/mineworld 700M_16f.ckpt --local-dir $dir"
    echo "  huggingface-cli download microsoft/mineworld vae/config.json --local-dir $dir/vae/"
    echo "  huggingface-cli download microsoft/mineworld vae/vae.ckpt --local-dir $dir/vae/"
    echo ""
    echo "Available model variants: 300M_16f, 700M_16f, 700M_32f, 1200M_16f, 1200M_32f"

    mkdir -p "$dir/vae"
    log "MineWorld checkpoint directory scaffolded at $dir"
}

# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
main() {
    local target="${1:-all}"
    case "$target" in
        oasis)        download_oasis ;;
        world_engine) download_world_engine ;;
        mineworld)    download_mineworld ;;
        all)
            download_mineworld
            download_oasis
            download_world_engine
            ;;
        *)
            echo "Usage: $0 {mineworld|oasis|world_engine|all}"
            exit 1
            ;;
    esac
}

main "$@"
