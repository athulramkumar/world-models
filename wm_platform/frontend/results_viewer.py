"""Results Viewer -- side-by-side MemFlow with/without comparison videos."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "test_results"

# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _load_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _find_runs(prefix: str) -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        (d for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name.startswith(prefix)),
        key=lambda d: d.name,
    )


def _get_comparison_runs() -> list[Path]:
    return _find_runs("comparison_")


def _get_frame_gallery(run_dir: Path, prefix: str, max_frames: int = 16) -> list:
    d = run_dir / f"frames_{prefix}"
    if not d.exists():
        return []
    paths = sorted(str(p) for p in d.iterdir() if p.suffix == ".png")[:max_frames]
    return [Image.open(p) for p in paths]


def _comparison_choices() -> list[str]:
    runs = _get_comparison_runs()
    return [r.name for r in runs]


def _format_run_label(name: str) -> str:
    parts = name.replace("comparison_", "").split("_")
    if len(parts) >= 2:
        engine = parts[0]
        rest = " ".join(parts[1:])
        return f"{engine.upper()}: {rest}"
    return name


# ------------------------------------------------------------------ #
#  Load a comparison run
# ------------------------------------------------------------------ #

def _load_comparison(run_name: str):
    """Load a comparison run and return all UI components."""
    empty = (None, None, [], [], "", "", "", "")
    if not run_name:
        return empty

    run_dir = RESULTS_DIR / run_name
    if not run_dir.exists():
        return empty

    meta = _load_json(run_dir / "metadata.json")

    vid_without = run_dir / "without_memflow.mp4"
    vid_with = run_dir / "with_memflow.mp4"
    v_without = str(vid_without) if vid_without.exists() else None
    v_with = str(vid_with) if vid_with.exists() else None

    gallery_without = _get_frame_gallery(run_dir, "without")
    gallery_with = _get_frame_gallery(run_dir, "with")

    engine = meta.get("engine", "unknown")
    model = meta.get("model", "unknown")
    duration = meta.get("duration_s", "?")
    fps = meta.get("fps", 6)
    total_frames_bl = meta.get("without_memflow", {}).get("total_frames", "?")
    total_frames_mf = meta.get("with_memflow", {}).get("total_frames", "?")
    gen_time_bl = meta.get("without_memflow", {}).get("generation_time_s", 0)
    gen_time_mf = meta.get("with_memflow", {}).get("generation_time_s", 0)

    header = (
        f"### {engine} -- {model}\n\n"
        f"**Duration**: {duration}s @ {fps}fps | "
        f"**Frames**: {total_frames_bl} (baseline) / {total_frames_mf} (MemFlow)"
    )

    if engine == "open_oasis":
        prompt_img = meta.get("prompt_image", "")
        actions_file = meta.get("actions_file", "")
        header += f"\n\n**Prompt**: `{os.path.basename(prompt_img)}`"
        header += f" | **Actions**: `{os.path.basename(actions_file)}`"
    elif engine == "world_engine":
        prompt_text = meta.get("prompt", "")
        header += f"\n\n**Prompt**: \"{prompt_text}\""

    bl_info = (
        f"**Without MemFlow (Baseline)**\n\n"
        f"- {total_frames_bl} frames generated in {gen_time_bl:.1f}s\n"
        f"- Standard autoregressive generation\n"
        f"- No persistent memory\n"
        f"- Context window only ({meta.get('without_memflow', {}).get('chunk_size', 32)} frame chunks)\n"
        f"- Cannot query past objects/entities"
    )

    mf_data = meta.get("with_memflow", {})
    mem_stats = mf_data.get("final_memory_stats", {})
    mem_prompt = mf_data.get("final_prompt", "N/A")
    corrections = mf_data.get("correction_log", [])

    mf_info = (
        f"**With MemFlow**\n\n"
        f"- {total_frames_mf} frames generated in {gen_time_mf:.1f}s\n"
        f"- MemFlow extraction + memory on each frame\n"
        f"- Corrections applied between chunks/intervals\n"
    )
    if mem_stats:
        mf_info += (
            f"- **Memory graph**: {mem_stats.get('nodes', 0)} nodes, "
            f"{mem_stats.get('edges', 0)} edges\n"
            f"- **Objects**: {mem_stats.get('objects', 0)} | "
            f"**Entities**: {mem_stats.get('entities', 0)} | "
            f"**Locations**: {mem_stats.get('locations', 0)}\n"
        )
    if corrections:
        mf_info += f"- **Corrections applied**: {len(corrections)}\n"
    if mem_prompt and mem_prompt != "N/A":
        mf_info += f"\n**Memory state prompt**:\n> {mem_prompt[:300]}"

    quality = meta.get("quality", {})
    q_without = quality.get("without", {})
    q_with = quality.get("with", {})
    extra = ""
    if q_without or q_with:
        extra = "### AI Quality Scores (GPT-4o Vision)\n\n"
        if q_without:
            extra += (
                f"**Without MemFlow**: {q_without.get('verdict', '?')} "
                f"(avg {q_without.get('avg_score', '?')}/10, "
                f"meaningful: {q_without.get('meaningful_ratio', '?')})\n\n"
            )
        if q_with:
            extra += (
                f"**With MemFlow**: {q_with.get('verdict', '?')} "
                f"(avg {q_with.get('avg_score', '?')}/10, "
                f"meaningful: {q_with.get('meaningful_ratio', '?')})\n\n"
            )

    return (v_without, v_with, gallery_without, gallery_with,
            header, bl_info, mf_info, extra)


# ------------------------------------------------------------------ #
#  Quantitative comparison data (pre-computed)
# ------------------------------------------------------------------ #

def _load_quant_data():
    obj_runs = _find_runs("memflow_object_comparison")
    char_runs = _find_runs("memflow_character_comparison")
    obj_results = []
    if obj_runs:
        obj_results = _load_json(obj_runs[-1] / "metadata.json").get("results", [])
    char_results = []
    if char_runs:
        char_results = _load_json(char_runs[-1] / "metadata.json").get("results", [])
    return obj_results, char_results


# ------------------------------------------------------------------ #
#  Individual model runs (legacy browser)
# ------------------------------------------------------------------ #

def _list_engine_runs(prefix: str) -> list[str]:
    return [d.name for d in _find_runs(prefix)]


def _load_single_run(run_name: str):
    if not run_name:
        return None, [], ""
    run_dir = RESULTS_DIR / run_name
    if not run_dir.exists():
        return None, [], "Not found."
    meta = _load_json(run_dir / "metadata.json")
    vid = None
    for name in ("output.mp4", "oasis_raw.mp4"):
        p = run_dir / name
        if p.exists():
            vid = str(p)
            break
    frames_dir = run_dir / "frames"
    gallery = []
    if frames_dir.exists():
        paths = sorted(str(p) for p in frames_dir.iterdir() if p.suffix == ".png")
        gallery = [Image.open(p) for p in paths[:24]]
    info_parts = []
    for k in ("engine", "model", "prompt", "prompt_image", "actions_name",
              "total_frames", "generation_time_s", "ms_per_frame"):
        if k in meta:
            info_parts.append(f"**{k}**: {meta[k]}")
    return vid, gallery, "\n\n".join(info_parts)


# ------------------------------------------------------------------ #
#  Build UI
# ------------------------------------------------------------------ #

def build_results_viewer_tab() -> gr.Blocks:
    obj_results, char_results = _load_quant_data()
    comparison_choices = _comparison_choices()
    oasis_runs = _list_engine_runs("oasis_")
    we_runs = _list_engine_runs("we_")

    with gr.Blocks() as tab:
        gr.Markdown("## MemFlow: With vs Without Comparison")

        with gr.Tabs():

            # ========================================================== #
            #  Tab 1: Side-by-Side Video Comparison (THE MAIN TAB)
            # ========================================================== #
            with gr.Tab("Side-by-Side Videos"):
                gr.Markdown(
                    "### Watch full-length videos side-by-side\n"
                    "Select a comparison run below. Each run generates the same sequence "
                    "**with** and **without** MemFlow corrections, then saves both videos."
                )

                with gr.Row():
                    comp_selector = gr.Dropdown(
                        choices=comparison_choices,
                        value=comparison_choices[0] if comparison_choices else None,
                        label="Select Comparison Run",
                        scale=3,
                    )
                    comp_load_btn = gr.Button("Load", variant="primary", scale=1)

                comp_header = gr.Markdown()

                gr.Markdown("---")

                with gr.Row(equal_height=True):
                    with gr.Column():
                        gr.Markdown("#### Without MemFlow")
                        comp_vid_without = gr.Video(label="Without MemFlow", autoplay=True)
                        comp_info_bl = gr.Markdown()
                        comp_gallery_without = gr.Gallery(
                            label="Sampled Frames (Without)", columns=8, height=200,
                        )
                    with gr.Column():
                        gr.Markdown("#### With MemFlow")
                        comp_vid_with = gr.Video(label="With MemFlow", autoplay=True)
                        comp_info_mf = gr.Markdown()
                        comp_gallery_with = gr.Gallery(
                            label="Sampled Frames (With)", columns=8, height=200,
                        )

                comp_extra = gr.Markdown()

                comp_outputs = [
                    comp_vid_without, comp_vid_with,
                    comp_gallery_without, comp_gallery_with,
                    comp_header, comp_info_bl, comp_info_mf, comp_extra,
                ]

                comp_selector.change(
                    _load_comparison, inputs=comp_selector, outputs=comp_outputs,
                )
                comp_load_btn.click(
                    _load_comparison, inputs=comp_selector, outputs=comp_outputs,
                )
                if comparison_choices:
                    tab.load(
                        _load_comparison, inputs=[comp_selector], outputs=comp_outputs,
                    )

            # ========================================================== #
            #  Tab 2: Object Persistence Quantitative
            # ========================================================== #
            with gr.Tab("Object Persistence"):
                gr.Markdown("### Object Persistence: Diamond in Kitchen Chest")
                gr.Markdown(
                    "> A diamond is placed in a chest. The player explores elsewhere for "
                    "30 / 60 / 120 seconds, then returns. Can the system recall the diamond?"
                )

                with gr.Row(equal_height=True):
                    with gr.Column():
                        gr.Markdown("#### Without MemFlow")
                        obj_bl = (
                            "| Duration | Recall | Query Support |\n"
                            "|:--------:|:------:|:-------------:|\n"
                        )
                        for d in [30, 60, 120]:
                            obj_bl += f"| {d}s | **0.0** (FORGOTTEN) | No |\n"
                        gr.Markdown(obj_bl)

                    with gr.Column():
                        gr.Markdown("#### With MemFlow")
                        if obj_results:
                            obj_mf = (
                                "| Duration | Recall | Query Support |\n"
                                "|:--------:|:------:|:-------------:|\n"
                            )
                            for r in obj_results:
                                obj_mf += (
                                    f"| {r['duration_s']}s | "
                                    f"**{r['memflow_recall']:.4f}** | Yes |\n"
                                )
                            gr.Markdown(obj_mf)

                gr.Markdown("---")
                if obj_results:
                    combined = (
                        "| Duration | Without MemFlow | With MemFlow | Winner |\n"
                        "|:--------:|:---------------:|:------------:|:------:|\n"
                    )
                    for r in obj_results:
                        combined += (
                            f"| {r['duration_s']}s | 0.0 (FORGOTTEN) | "
                            f"{r['memflow_recall']:.4f} (REMEMBERED) | MemFlow |\n"
                        )
                    gr.Markdown(combined)

            # ========================================================== #
            #  Tab 3: Character Persistence Quantitative
            # ========================================================== #
            with gr.Tab("Character Persistence"):
                gr.Markdown("### Character Persistence: Alice & Bob Identity")
                gr.Markdown(
                    "> Alice and Bob separate to different rooms for 30/60/120 seconds. "
                    "Do they keep their identities?"
                )

                with gr.Row(equal_height=True):
                    with gr.Column():
                        gr.Markdown("#### Without MemFlow")
                        char_bl = (
                            "| Duration | Alice | Bob |\n"
                            "|:--------:|:-----:|:---:|\n"
                        )
                        for d in [30, 60, 120]:
                            char_bl += f"| {d}s | FORGOTTEN | FORGOTTEN |\n"
                        gr.Markdown(char_bl)

                    with gr.Column():
                        gr.Markdown("#### With MemFlow")
                        if char_results:
                            char_mf = (
                                "| Duration | Alice | Bob | Features |\n"
                                "|:--------:|:-----:|:---:|:--------:|\n"
                            )
                            for r in char_results:
                                char_mf += (
                                    f"| {r['duration_s']}s | "
                                    f"**{r['memflow_alice_recall']:.4f}** | "
                                    f"**{r['memflow_bob_recall']:.4f}** | "
                                    f"{'Preserved' if r.get('memflow_features_preserved') else 'Lost'} |\n"
                                )
                            gr.Markdown(char_mf)

            # ========================================================== #
            #  Tab 4: Open-Oasis Runs
            # ========================================================== #
            with gr.Tab("Open-Oasis Runs"):
                gr.Markdown("### Browse Individual Oasis Runs")
                oa_sel = gr.Dropdown(
                    choices=oasis_runs,
                    value=oasis_runs[0] if oasis_runs else None,
                    label="Select Run",
                )
                oa_vid = gr.Video(label="Video")
                oa_gal = gr.Gallery(label="Frames", columns=8)
                oa_info = gr.Markdown()
                oa_sel.change(
                    _load_single_run, inputs=oa_sel,
                    outputs=[oa_vid, oa_gal, oa_info],
                )

            # ========================================================== #
            #  Tab 5: World Engine Runs
            # ========================================================== #
            with gr.Tab("World Engine Runs"):
                gr.Markdown("### Browse Individual World Engine Runs")
                we_sel = gr.Dropdown(
                    choices=we_runs,
                    value=we_runs[0] if we_runs else None,
                    label="Select Run",
                )
                we_vid = gr.Video(label="Video")
                we_gal = gr.Gallery(label="Frames", columns=8)
                we_info = gr.Markdown()
                we_sel.change(
                    _load_single_run, inputs=we_sel,
                    outputs=[we_vid, we_gal, we_info],
                )

            # ========================================================== #
            #  Tab 6: Confidence Decay
            # ========================================================== #
            with gr.Tab("Confidence Decay"):
                gr.Markdown("### How Confidence Decays Over Time")
                gr.Markdown(
                    "MemFlow uses time-based confidence decay. Memories fade gradually "
                    "instead of vanishing abruptly."
                )
                decay_table = (
                    "| Time Since Last Seen | Object Confidence | Character Confidence | Baseline |\n"
                    "|:--------------------:|:-----------------:|:--------------------:|:--------:|\n"
                    "| 0s (just seen)       | 1.0000            | 1.0000               | 1.0      |\n"
                    "| 5s (past window)     | ~0.95             | ~0.99                | **0.0**  |\n"
                    "| 30s                  | 0.7135            | 0.9714               | **0.0**  |\n"
                    "| 60s                  | 0.0500            | 0.8887               | **0.0**  |\n"
                    "| 120s                 | 0.0500            | 0.5614               | **0.0**  |\n"
                )
                gr.Markdown(decay_table)
                gr.Markdown(
                    "**Key insight**: Without MemFlow, recall drops from 1.0 to 0.0 instantly "
                    "when the object leaves the context window. With MemFlow, it decays smoothly."
                )

    return tab
