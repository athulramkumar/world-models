"""World Models Platform -- FastAPI + Gradio unified application."""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

import gradio as gr
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wm_platform.config import (
    auto_select_profile,
    get_env_status,
    list_profiles,
    load_profile,
)
from wm_platform.engines import ENGINE_CLASSES, EngineState
from wm_platform.frontend.dashboard import build_dashboard_tab
from wm_platform.frontend.model_explorer import build_model_explorer_tab
from wm_platform.frontend.memflow_panel import build_memflow_tab
from wm_platform.frontend.results_viewer import build_results_viewer_tab
from wm_platform.interactive import interactive_ws_handler

# ------------------------------------------------------------------ #
#  FastAPI app
# ------------------------------------------------------------------ #

api = FastAPI(title="World Models Platform", version="0.1.0")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
api.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@api.get("/interactive")
def interactive_page():
    html_path = STATIC_DIR / "interactive.html"
    return HTMLResponse(html_path.read_text())


@api.websocket("/ws/interactive")
async def ws_interactive(websocket: WebSocket):
    await interactive_ws_handler(websocket)


@api.get("/api/health")
def health():
    return {"status": "ok"}


@api.get("/api/profiles")
def api_list_profiles():
    return {"profiles": list_profiles()}


@api.get("/api/profiles/{name}")
def api_get_profile(name: str):
    try:
        p = load_profile(name)
        return {
            "gpu": {"name": p.gpu.name, "vram_gb": p.gpu.vram_gb, "arch": p.gpu.architecture},
            "cuda": p.cuda_version,
            "engines": {k: vars(v) for k, v in p.engines.items()},
        }
    except FileNotFoundError:
        return {"error": f"Profile '{name}' not found"}


@api.get("/api/envs")
def api_env_status():
    return get_env_status()


@api.get("/api/gpu")
def api_gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "name": parts[0],
            "vram_used_mb": int(parts[1]),
            "vram_total_mb": int(parts[2]),
            "utilization_pct": int(parts[3]),
            "temp_c": int(parts[4]),
        }
    except Exception as e:
        return {"error": str(e)}


# ------------------------------------------------------------------ #
#  Gradio UI
# ------------------------------------------------------------------ #

def build_gradio_app() -> gr.Blocks:
    with gr.Blocks(title="World Models Platform") as app:
        gr.Markdown("# World Models Platform")
        gr.Markdown(
            "Unified interface for exploring world models (MineWorld, Open-Oasis, World Engine) "
            "with MemFlow structured state management."
        )

        with gr.Tabs():
            with gr.Tab("Dashboard"):
                build_dashboard_tab()
            with gr.Tab("Model Explorer"):
                build_model_explorer_tab()
            with gr.Tab("MemFlow"):
                build_memflow_tab()
            with gr.Tab("Results"):
                build_results_viewer_tab()

    return app


# Mount Gradio inside FastAPI
gradio_app = build_gradio_app()
app = gr.mount_gradio_app(api, gradio_app, path="/")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="World Models Platform")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    print(f"Starting World Models Platform on {args.host}:{args.port}")
    print(f"Detected GPU: {auto_select_profile().gpu.name}")
    print(f"Environment status: {get_env_status()}")
    print(f"Available profiles: {list_profiles()}")

    uvicorn.run(
        "wm_platform.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
