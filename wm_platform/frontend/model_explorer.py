"""Model Explorer tab -- per-model interaction UIs for MineWorld, Oasis, and WorldEngine."""

from __future__ import annotations

import io
import os
import tempfile
from typing import Optional

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from ..config import CHECKPOINTS_DIR, REPOS_DIR, auto_select_profile, get_env_status
from ..engines import ENGINE_CLASSES, EngineState
from ..engines.base import BaseWorldEngine, Frame

_engines: dict[str, BaseWorldEngine] = {}


def _get_engine(name: str) -> Optional[BaseWorldEngine]:
    return _engines.get(name)


def _load_engine(name: str, variant: str = "") -> str:
    if name in _engines and _engines[name].status.state == EngineState.READY:
        return f"{name} already loaded."
    try:
        cls = ENGINE_CLASSES[name]
        engine = cls()
        kwargs = {}
        if variant:
            kwargs["model_variant"] = variant
        engine.load(**kwargs)
        _engines[name] = engine
        s = engine.status
        if s.state == EngineState.ERROR:
            return f"Load error: {s.error}"
        return f"{s.model_name} loaded successfully."
    except Exception as e:
        return f"Error: {e}"


def _unload_engine(name: str) -> str:
    if name in _engines:
        _engines[name].unload()
        del _engines[name]
        return f"{name} unloaded."
    return f"{name} not loaded."


# ------------------------------------------------------------------ #
#  MineWorld sub-tab
# ------------------------------------------------------------------ #

MINEWORLD_ACTIONS_LIST = [
    "forward", "back", "left", "right", "jump", "sprint", "sneak",
    "attack", "use", "drop",
]


def mineworld_generate(selected_actions, cam_x, cam_y):
    engine = _get_engine("mineworld")
    if not engine or engine.status.state != EngineState.READY:
        return None, "MineWorld not loaded."

    action_dict = {a: 0 for a in MINEWORLD_ACTIONS_LIST}
    for a in (selected_actions or []):
        if a in action_dict:
            action_dict[a] = 1
    action_dict["camera"] = [int(cam_y or 0), int(cam_x or 0)]

    try:
        frame = engine.generate_frame(action_dict)
        img = Image.fromarray(frame.rgb)
        return img, f"Frame {frame.frame_idx} generated in {engine.status.last_gen_time_ms:.0f}ms"
    except Exception as e:
        return None, f"Error: {e}"


def build_mineworld_subtab() -> gr.Blocks:
    with gr.Blocks() as tab:
        gr.Markdown("### MineWorld -- Llama Autoregressive World Model")

        with gr.Row():
            variant = gr.Dropdown(
                choices=["300M_16f", "700M_16f", "700M_32f", "1200M_16f", "1200M_32f"],
                value="700M_16f",
                label="Model Variant",
            )
            load_btn = gr.Button("Load Model", variant="primary")
            unload_btn = gr.Button("Unload")
            status_txt = gr.Textbox(label="Status", interactive=False)

        load_btn.click(lambda v: _load_engine("mineworld", v), inputs=variant, outputs=status_txt)
        unload_btn.click(lambda: _unload_engine("mineworld"), outputs=status_txt)

        with gr.Row():
            actions = gr.CheckboxGroup(MINEWORLD_ACTIONS_LIST, label="Actions")
            cam_x = gr.Slider(-90, 90, 0, step=1, label="Camera X")
            cam_y = gr.Slider(-90, 90, 0, step=1, label="Camera Y")

        gen_btn = gr.Button("Generate Frame", variant="primary")
        with gr.Row():
            frame_display = gr.Image(label="Generated Frame", width=384, height=224)
            gen_info = gr.Textbox(label="Info", interactive=False)

        gen_btn.click(mineworld_generate, inputs=[actions, cam_x, cam_y], outputs=[frame_display, gen_info])

    return tab


# ------------------------------------------------------------------ #
#  Open-Oasis sub-tab
# ------------------------------------------------------------------ #

OASIS_SAMPLE_DIR = REPOS_DIR / "open-oasis" / "sample_data"
OASIS_DEFAULT_IMAGE = str(OASIS_SAMPLE_DIR / "sample_image_0.png")


def _list_oasis_actions() -> list[str]:
    if not OASIS_SAMPLE_DIR.exists():
        return []
    return sorted(
        f.name for f in OASIS_SAMPLE_DIR.iterdir()
        if f.name.endswith(".one_hot_actions.pt")
    )


def _list_oasis_prompts() -> list[str]:
    if not OASIS_SAMPLE_DIR.exists():
        return []
    prompts = []
    for f in sorted(OASIS_SAMPLE_DIR.iterdir()):
        if f.suffix in (".png", ".jpg", ".mp4"):
            prompts.append(f.name)
    return prompts


def oasis_load_prompt(prompt_name: str):
    path = OASIS_SAMPLE_DIR / prompt_name
    if path.suffix == ".mp4":
        cap = cv2.VideoCapture(str(path))
        ret, frame = cap.read()
        cap.release()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    return np.array(Image.open(path))


def oasis_generate(prompt_img, actions_name, total_frames):
    engine = _get_engine("open_oasis")
    if not engine or engine.status.state != EngineState.READY:
        return None, "Oasis not loaded. Click 'Load Model' first."

    try:
        if prompt_img is not None:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            Image.fromarray(prompt_img).save(tmp.name)
            prompt_path = tmp.name
        else:
            prompt_path = OASIS_DEFAULT_IMAGE

        actions_path = str(OASIS_SAMPLE_DIR / actions_name) if actions_name else \
            str(OASIS_SAMPLE_DIR / "sample_actions_0.one_hot_actions.pt")

        frames = engine.generate_video(
            prompt_path=prompt_path,
            actions_path=actions_path,
            total_frames=int(total_frames or 16),
        )

        if frames:
            gallery = [Image.fromarray(f.rgb) for f in frames]
            return gallery, (
                f"Generated {len(frames)} frames in {engine.status.last_gen_time_ms:.0f}ms\n"
                f"Prompt: {os.path.basename(prompt_path)}\n"
                f"Actions: {actions_name}"
            )
        return None, "No frames generated."
    except Exception as e:
        return None, f"Error: {e}"


def build_oasis_subtab() -> gr.Blocks:
    prompt_choices = _list_oasis_prompts()
    action_choices = _list_oasis_actions()
    default_action = "sample_actions_0.one_hot_actions.pt" if "sample_actions_0.one_hot_actions.pt" in action_choices else (action_choices[0] if action_choices else "")

    with gr.Blocks() as tab:
        gr.Markdown("### Open-Oasis -- Diffusion Transformer World Model (500M)")
        gr.Markdown(
            "Generates Minecraft gameplay video from a starting frame + player actions. "
            "No text prompt -- purely vision + action conditioned."
        )

        with gr.Row():
            load_btn = gr.Button("Load Model", variant="primary")
            unload_btn = gr.Button("Unload")
            status_txt = gr.Textbox(label="Status", interactive=False)

        load_btn.click(lambda: _load_engine("open_oasis"), outputs=status_txt)
        unload_btn.click(lambda: _unload_engine("open_oasis"), outputs=status_txt)

        gr.Markdown("#### Inputs")
        with gr.Row():
            with gr.Column(scale=2):
                prompt_img = gr.Image(
                    label="Prompt Image (starting frame)",
                    type="numpy",
                    value=OASIS_DEFAULT_IMAGE if os.path.isfile(OASIS_DEFAULT_IMAGE) else None,
                )
                prompt_dropdown = gr.Dropdown(
                    choices=prompt_choices,
                    value="sample_image_0.png" if "sample_image_0.png" in prompt_choices else None,
                    label="Load sample prompt",
                )
                prompt_dropdown.change(oasis_load_prompt, inputs=prompt_dropdown, outputs=prompt_img)
            with gr.Column(scale=1):
                actions_dropdown = gr.Dropdown(
                    choices=action_choices,
                    value=default_action,
                    label="Actions file",
                )
                total_frames = gr.Slider(4, 64, 16, step=4, label="Total Frames")

        gen_btn = gr.Button("Generate Video", variant="primary")
        gallery = gr.Gallery(label="Generated Frames", columns=8)
        gen_info = gr.Textbox(label="Info", interactive=False)

        gen_btn.click(
            oasis_generate,
            inputs=[prompt_img, actions_dropdown, total_frames],
            outputs=[gallery, gen_info],
        )

    return tab


# ------------------------------------------------------------------ #
#  World Engine sub-tab
# ------------------------------------------------------------------ #

def we_generate(prompt_text, mouse_x, mouse_y):
    engine = _get_engine("world_engine")
    if not engine or engine.status.state != EngineState.READY:
        return None, "World Engine not loaded."

    try:
        from ..engines.world_engine_adapter import WorldEngineAdapter

        if prompt_text:
            engine.set_prompt(prompt_text)

        frame = engine.generate_frame({
            "button": [],
            "mouse": [float(mouse_x or 0), float(mouse_y or 0)],
            "scroll_wheel": 0,
        })
        return Image.fromarray(frame.rgb), f"Frame {frame.frame_idx} in {engine.status.last_gen_time_ms:.0f}ms"
    except Exception as e:
        return None, f"Error: {e}"


def build_world_engine_subtab() -> gr.Blocks:
    with gr.Blocks() as tab:
        gr.Markdown("### World Engine -- Overworld Inference Engine")

        with gr.Row():
            model_uri = gr.Textbox(value="Overworld/Waypoint-1-Small", label="Model URI")
            load_btn = gr.Button("Load Model", variant="primary")
            unload_btn = gr.Button("Unload")
            status_txt = gr.Textbox(label="Status", interactive=False)

        load_btn.click(
            lambda uri: _load_engine("world_engine", uri),
            inputs=model_uri,
            outputs=status_txt,
        )
        unload_btn.click(lambda: _unload_engine("world_engine"), outputs=status_txt)

        prompt_text = gr.Textbox(label="Text Prompt", value="An explorable Minecraft world")
        with gr.Row():
            mouse_x = gr.Slider(-1, 1, 0, step=0.05, label="Mouse X")
            mouse_y = gr.Slider(-1, 1, 0, step=0.05, label="Mouse Y")

        gen_btn = gr.Button("Generate Frame", variant="primary")
        with gr.Row():
            frame_display = gr.Image(label="Generated Frame")
            gen_info = gr.Textbox(label="Info", interactive=False)

        gen_btn.click(we_generate, inputs=[prompt_text, mouse_x, mouse_y], outputs=[frame_display, gen_info])

    return tab


def build_model_explorer_tab() -> gr.Blocks:
    with gr.Blocks() as tab:
        gr.Markdown("## Model Explorer")
        with gr.Tabs():
            with gr.Tab("MineWorld"):
                build_mineworld_subtab()
            with gr.Tab("Open-Oasis"):
                build_oasis_subtab()
            with gr.Tab("World Engine"):
                build_world_engine_subtab()
    return tab
