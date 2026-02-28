"""Dashboard tab -- system status, GPU info, hardware profile, engine quick-launch."""

from __future__ import annotations

import subprocess
from typing import Optional

import gradio as gr

from ..config import (
    HardwareProfile,
    auto_select_profile,
    get_env_status,
    list_profiles,
    load_profile,
)
from ..engines import ENGINE_CLASSES, EngineState

_active_engines: dict = {}
_current_profile: Optional[HardwareProfile] = None


def get_gpu_info() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        parts = out.split(",")
        return (
            f"**GPU**: {parts[0].strip()}\n"
            f"**VRAM**: {parts[1].strip()} / {parts[2].strip()} MiB\n"
            f"**Utilization**: {parts[3].strip()}%"
        )
    except Exception as e:
        return f"GPU info unavailable: {e}"


def get_env_status_md() -> str:
    status = get_env_status()
    lines = ["| Environment | Ready |", "| --- | --- |"]
    for name, ready in status.items():
        icon = "Ready" if ready else "Not set up"
        lines.append(f"| {name} | {icon} |")
    return "\n".join(lines)


def get_engine_status_md() -> str:
    if not _active_engines:
        return "No engines loaded."
    lines = ["| Engine | State | Model | Frames | Last Gen |", "| --- | --- | --- | --- | --- |"]
    for name, engine in _active_engines.items():
        s = engine.status
        lines.append(
            f"| {s.name} | {s.state.value} | {s.model_name} | "
            f"{s.frames_generated} | {s.last_gen_time_ms:.0f}ms |"
        )
    return "\n".join(lines)


def refresh_dashboard():
    return get_gpu_info(), get_env_status_md(), get_engine_status_md()


def select_profile(profile_name: str) -> str:
    global _current_profile
    try:
        _current_profile = load_profile(profile_name)
        gpu = _current_profile.gpu
        return (
            f"**Profile**: {profile_name}\n"
            f"**GPU**: {gpu.name} ({gpu.vram_gb}GB, {gpu.architecture})\n"
            f"**CUDA**: {_current_profile.cuda_version}"
        )
    except Exception as e:
        return f"Error loading profile: {e}"


def build_dashboard_tab() -> gr.Blocks:
    with gr.Blocks() as tab:
        gr.Markdown("## System Dashboard")

        with gr.Row():
            gpu_info = gr.Markdown(get_gpu_info())
            env_status = gr.Markdown(get_env_status_md())

        with gr.Row():
            profile_dropdown = gr.Dropdown(
                choices=list_profiles(),
                label="Hardware Profile",
                value="a100_80gb",
            )
            profile_info = gr.Markdown()
            profile_dropdown.change(select_profile, inputs=profile_dropdown, outputs=profile_info)

        gr.Markdown("### Engine Status")
        engine_status = gr.Markdown(get_engine_status_md())

        refresh_btn = gr.Button("Refresh", variant="secondary")
        refresh_btn.click(refresh_dashboard, outputs=[gpu_info, env_status, engine_status])

    return tab
