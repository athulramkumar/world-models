#!/usr/bin/env python3
"""Generate comparison videos: with vs without MemFlow for Oasis and World Engine.

Key fixes over v1:
  - Oasis: advance video_offset between chunks so actions align with the scene
  - World Engine: minimal camera movement for stable generation
  - AI quality judge validates output frames via GPT-4o vision

Outputs go to test_results/comparison_<engine>_<label>/ with:
  - without_memflow.mp4 / with_memflow.mp4
  - metadata.json  (includes quality scores)
  - frames_without/ frames_with/  (sampled PNGs)
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from wm_platform.engines.oasis_engine import OasisEngine
from wm_platform.engines.world_engine_adapter import WorldEngineAdapter
from wm_platform.memflow.corrector import Corrector
from wm_platform.memflow.extractor import StateExtractor
from wm_platform.memflow.memory import StructuredMemory
from wm_platform.memflow.types import CorrectionStrategy, Observation

RESULTS_DIR = ROOT / "test_results"
RESULTS_DIR.mkdir(exist_ok=True)
SAMPLE_DIR = ROOT / "repos" / "open-oasis" / "sample_data"
FPS = 6


def save_video_h264(frames: list[np.ndarray], path: str, fps: int = FPS):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    tmp = path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp, fourcc, fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    ret = subprocess.run(
        ["ffmpeg", "-y", "-i", tmp, "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", path],
        capture_output=True,
    )
    if ret.returncode == 0:
        os.remove(tmp)
    else:
        os.rename(tmp, path)


def save_metadata(run_dir: Path, data: dict):
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(data, f, indent=2, default=str)


def save_frames_gallery(run_dir: Path, frames: list[np.ndarray], prefix: str, sample_every: int = 1):
    d = run_dir / f"frames_{prefix}"
    d.mkdir(exist_ok=True)
    for i, f in enumerate(frames):
        if i % sample_every == 0:
            Image.fromarray(f).save(str(d / f"{prefix}_{i:04d}.png"))


# ------------------------------------------------------------------ #
#  Oasis: chunked generation with proper action offset advancement
# ------------------------------------------------------------------ #

def generate_oasis_chunked(
    engine: OasisEngine,
    prompt_path: str,
    actions_path: str,
    total_frames: int,
    chunk_size: int = 32,
    use_memflow: bool = False,
) -> tuple[list[np.ndarray], dict]:
    """Generate frames in chunks, advancing action offset each time."""
    extractor = StateExtractor() if use_memflow else None
    memory = StructuredMemory(decay_rate=0.0001, min_confidence=0.05) if use_memflow else None
    corrector = Corrector(
        strategy=CorrectionStrategy.FRAME_INJECTION,
        injection_interval=1,
    ) if use_memflow else None

    all_frames: list[np.ndarray] = []
    current_prompt = prompt_path
    correction_log = []
    chunk_idx = 0
    action_offset = 0

    while len(all_frames) < total_frames:
        remaining = total_frames - len(all_frames)
        n = min(chunk_size, remaining)
        if n < 2:
            break

        tag = "MF" if use_memflow else "BL"
        print(f"  [{tag}] Chunk {chunk_idx}: frames {action_offset}..{action_offset+n} "
              f"({len(all_frames)}/{total_frames})", flush=True)

        frames = engine.generate_video(
            prompt_path=current_prompt,
            actions_path=actions_path,
            total_frames=n,
            n_prompt_frames=1,
            video_offset=action_offset if action_offset > 0 else None,
        )
        chunk_rgb = [f.rgb for f in frames]
        all_frames.extend(chunk_rgb)

        if use_memflow and extractor and memory and corrector:
            for i, rgb in enumerate(chunk_rgb):
                global_idx = len(all_frames) - len(chunk_rgb) + i
                obs = Observation(
                    frame_idx=global_idx,
                    timestamp=time.time(),
                    rgb=rgb,
                    engine_id="oasis",
                )
                scene = extractor.classify_scene(obs)
                memory.ingest_scene(scene)
                if scene.scene_id not in corrector._reference_frames:
                    corrector.store_reference_frame(scene.scene_id, rgb)

            mem_state = memory.snapshot()
            correction_log.append({
                "chunk": chunk_idx,
                "frame_range": [action_offset, action_offset + len(chunk_rgb)],
                "memory_nodes": memory.stats()["nodes"],
                "memory_edges": memory.stats()["edges"],
                "prompt_from_memory": mem_state.to_prompt()[:200],
            })

        last_frame = chunk_rgb[-1]
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            Image.fromarray(last_frame).save(tmp.name)
            current_prompt = tmp.name

        action_offset += n
        chunk_idx += 1

    info = {
        "total_frames": len(all_frames),
        "chunk_size": chunk_size,
        "num_chunks": chunk_idx,
        "use_memflow": use_memflow,
    }
    if use_memflow and memory:
        info["correction_log"] = correction_log
        info["final_memory_stats"] = memory.stats()
        info["final_prompt"] = memory.snapshot().to_prompt()

    return all_frames, info


def run_oasis_comparison(duration_s: int, prompt_name: str = "default", quality_check: bool = True):
    """Run Oasis comparison for a given duration."""
    total_frames = duration_s * FPS
    print(f"\n{'='*60}")
    print(f"  OASIS COMPARISON: {duration_s}s ({total_frames} frames)")
    print(f"  Prompt: {prompt_name}")
    print(f"{'='*60}", flush=True)

    if prompt_name == "treechop":
        mp4_path = SAMPLE_DIR / "treechop-f153ac423f61-20210916-183423.chunk_000.mp4"
        actions_path = str(SAMPLE_DIR / "treechop-f153ac423f61-20210916-183423.chunk_000.one_hot_actions.pt")
    elif prompt_name == "snippy":
        mp4_path = SAMPLE_DIR / "snippy-chartreuse-mastiff-f79998db196d-20220401-224517.chunk_001.mp4"
        actions_path = str(SAMPLE_DIR / "snippy-chartreuse-mastiff-f79998db196d-20220401-224517.chunk_001.one_hot_actions.pt")
    else:
        mp4_path = SAMPLE_DIR / "Player729-f153ac423f61-20210806-224813.chunk_000.mp4"
        actions_path = str(SAMPLE_DIR / "Player729-f153ac423f61-20210806-224813.chunk_000.one_hot_actions.pt")

    # Extract first frame from the MP4 as the prompt image
    prompt_path = str(RESULTS_DIR / f"_oasis_prompt_{prompt_name}.png")
    cap = cv2.VideoCapture(str(mp4_path))
    ret, bgr = cap.read()
    cap.release()
    if ret:
        Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(prompt_path)
    else:
        prompt_path = str(SAMPLE_DIR / "sample_image_0.png")

    run_dir = RESULTS_DIR / f"comparison_oasis_{prompt_name}_{duration_s}s"
    run_dir.mkdir(exist_ok=True)

    engine = OasisEngine()
    print("Loading Oasis engine...", flush=True)
    engine.load()

    # --- WITHOUT MemFlow ---
    t0 = time.time()
    print(f"\nGenerating WITHOUT MemFlow ({duration_s}s)...", flush=True)
    frames_bl, info_bl = generate_oasis_chunked(
        engine, prompt_path, actions_path, total_frames,
        chunk_size=32, use_memflow=False,
    )
    bl_time = time.time() - t0
    info_bl["generation_time_s"] = round(bl_time, 1)
    print(f"  Done: {len(frames_bl)} frames in {bl_time:.1f}s", flush=True)

    save_video_h264(frames_bl, str(run_dir / "without_memflow.mp4"))
    save_frames_gallery(run_dir, frames_bl, "without", sample_every=max(1, len(frames_bl) // 16))

    engine.reset()

    # --- WITH MemFlow ---
    t0 = time.time()
    print(f"\nGenerating WITH MemFlow ({duration_s}s)...", flush=True)
    frames_mf, info_mf = generate_oasis_chunked(
        engine, prompt_path, actions_path, total_frames,
        chunk_size=32, use_memflow=True,
    )
    mf_time = time.time() - t0
    info_mf["generation_time_s"] = round(mf_time, 1)
    print(f"  Done: {len(frames_mf)} frames in {mf_time:.1f}s", flush=True)

    save_video_h264(frames_mf, str(run_dir / "with_memflow.mp4"))
    save_frames_gallery(run_dir, frames_mf, "with", sample_every=max(1, len(frames_mf) // 16))

    engine.unload()

    # --- Quality check ---
    quality = {}
    if quality_check:
        try:
            from wm_platform.tests.frame_judge import judge_video_frames
            print("\nJudging quality (Without MemFlow)...", flush=True)
            quality["without"] = judge_video_frames(frames_bl, sample_count=4, context="Minecraft game world")
            print(f"  Verdict: {quality['without']['verdict']} (avg score: {quality['without']['avg_score']})", flush=True)

            print("Judging quality (With MemFlow)...", flush=True)
            quality["with"] = judge_video_frames(frames_mf, sample_count=4, context="Minecraft game world")
            print(f"  Verdict: {quality['with']['verdict']} (avg score: {quality['with']['avg_score']})", flush=True)
        except Exception as e:
            print(f"  Quality check failed: {e}", flush=True)

    meta = {
        "engine": "open_oasis",
        "model": "Oasis 500M (DiT-S/2)",
        "duration_s": duration_s,
        "fps": FPS,
        "prompt_image": prompt_path,
        "prompt_name": prompt_name,
        "actions_file": actions_path,
        "without_memflow": info_bl,
        "with_memflow": info_mf,
    }
    if quality:
        meta["quality"] = quality
    save_metadata(run_dir, meta)

    print(f"\nSaved to {run_dir}", flush=True)
    return run_dir


# ------------------------------------------------------------------ #
#  World Engine: frame-by-frame with/without MemFlow
# ------------------------------------------------------------------ #

def generate_we_sequence(
    engine: WorldEngineAdapter,
    prompt: str,
    total_frames: int,
    use_memflow: bool = False,
) -> tuple[list[np.ndarray], dict]:
    """Generate frames one by one. With MemFlow: prompt conditioning every N frames."""
    extractor = StateExtractor() if use_memflow else None
    memory = StructuredMemory(decay_rate=0.0001, min_confidence=0.05) if use_memflow else None
    corrector = Corrector(
        strategy=CorrectionStrategy.PROMPT_CONDITIONING,
        injection_interval=5,
    ) if use_memflow else None

    all_frames: list[np.ndarray] = []
    correction_log = []

    engine.set_prompt(prompt)

    for i in range(total_frames):
        # Gentle camera pan for visual variety
        mouse_x = 0.02 * np.sin(i * 0.15)
        actions = {"button": [], "mouse": [float(mouse_x), 0.0], "scroll_wheel": 0}

        if use_memflow and corrector and memory:
            if corrector.should_correct(i):
                mem_state = memory.snapshot()
                log = corrector.apply(engine, mem_state, frame_idx=i)
                if log:
                    correction_log.append({**log, "frame": i})
                    print(f"    [MF] Corrected at frame {i}: prompt={mem_state.to_prompt()[:80]}", flush=True)

        frame = engine.generate_frame(actions)
        all_frames.append(frame.rgb)

        if use_memflow and extractor and memory:
            obs = Observation(
                frame_idx=i, timestamp=time.time(),
                rgb=frame.rgb, engine_id="world_engine",
            )
            scene = extractor.classify_scene(obs)
            memory.ingest_scene(scene)

        if i % 5 == 0:
            tag = "MF" if use_memflow else "BL"
            print(f"  [{tag}] Frame {i}/{total_frames}", flush=True)

    info = {
        "total_frames": len(all_frames),
        "use_memflow": use_memflow,
        "prompt": prompt,
    }
    if use_memflow and memory:
        info["correction_log"] = correction_log
        info["final_memory_stats"] = memory.stats()
        info["final_prompt"] = memory.snapshot().to_prompt()

    return all_frames, info


def run_we_comparison(prompt: str, total_frames: int, label: str = "default", quality_check: bool = True):
    """Run World Engine comparison."""
    duration_s = total_frames / FPS
    print(f"\n{'='*60}")
    print(f"  WORLD ENGINE COMPARISON: {duration_s:.0f}s ({total_frames} frames)")
    print(f"  Prompt: {prompt}")
    print(f"{'='*60}", flush=True)

    run_dir = RESULTS_DIR / f"comparison_we_{label}_{total_frames}f"
    run_dir.mkdir(exist_ok=True)

    engine = WorldEngineAdapter()
    print("Loading World Engine...", flush=True)
    engine.load()

    t0 = time.time()
    print(f"\nGenerating WITHOUT MemFlow...", flush=True)
    frames_bl, info_bl = generate_we_sequence(engine, prompt, total_frames, use_memflow=False)
    bl_time = time.time() - t0
    info_bl["generation_time_s"] = round(bl_time, 1)
    print(f"  Done: {len(frames_bl)} frames in {bl_time:.1f}s", flush=True)

    save_video_h264(frames_bl, str(run_dir / "without_memflow.mp4"))
    save_frames_gallery(run_dir, frames_bl, "without", sample_every=max(1, len(frames_bl) // 16))

    engine.reset()

    t0 = time.time()
    print(f"\nGenerating WITH MemFlow...", flush=True)
    frames_mf, info_mf = generate_we_sequence(engine, prompt, total_frames, use_memflow=True)
    mf_time = time.time() - t0
    info_mf["generation_time_s"] = round(mf_time, 1)
    print(f"  Done: {len(frames_mf)} frames in {mf_time:.1f}s", flush=True)

    save_video_h264(frames_mf, str(run_dir / "with_memflow.mp4"))
    save_frames_gallery(run_dir, frames_mf, "with", sample_every=max(1, len(frames_mf) // 16))

    engine.unload()

    quality = {}
    if quality_check:
        try:
            from wm_platform.tests.frame_judge import judge_video_frames
            print("\nJudging quality (Without MemFlow)...", flush=True)
            quality["without"] = judge_video_frames(frames_bl, sample_count=4, context="FPS/adventure game world")
            print(f"  Verdict: {quality['without']['verdict']} (avg score: {quality['without']['avg_score']})", flush=True)

            print("Judging quality (With MemFlow)...", flush=True)
            quality["with"] = judge_video_frames(frames_mf, sample_count=4, context="FPS/adventure game world")
            print(f"  Verdict: {quality['with']['verdict']} (avg score: {quality['with']['avg_score']})", flush=True)
        except Exception as e:
            print(f"  Quality check failed: {e}", flush=True)

    meta = {
        "engine": "world_engine",
        "model": "Overworld/Waypoint-1-Small",
        "duration_s": duration_s,
        "fps": FPS,
        "prompt": prompt,
        "total_frames": total_frames,
        "without_memflow": info_bl,
        "with_memflow": info_mf,
    }
    if quality:
        meta["quality"] = quality
    save_metadata(run_dir, meta)

    print(f"\nSaved to {run_dir}", flush=True)
    return run_dir


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--oasis-durations", nargs="+", type=int, default=[10],
                        help="Oasis video durations in seconds")
    parser.add_argument("--oasis-prompts", nargs="+", default=["default"],
                        choices=["default", "treechop", "snippy"],
                        help="Which sample video/action pairs to use")
    parser.add_argument("--we-frames", type=int, default=30,
                        help="World Engine frames to generate")
    parser.add_argument("--we-prompt", type=str,
                        default="A dark indoor corridor with debris and dim lighting")
    parser.add_argument("--skip-oasis", action="store_true")
    parser.add_argument("--skip-we", action="store_true")
    parser.add_argument("--no-quality-check", action="store_true")
    args = parser.parse_args()

    results = []

    if not args.skip_oasis:
        for prompt_name in args.oasis_prompts:
            for dur in args.oasis_durations:
                rd = run_oasis_comparison(dur, prompt_name=prompt_name,
                                          quality_check=not args.no_quality_check)
                results.append(str(rd))

    if not args.skip_we:
        rd = run_we_comparison(args.we_prompt, args.we_frames, label="v2",
                               quality_check=not args.no_quality_check)
        results.append(str(rd))

    print(f"\n{'='*60}")
    print("ALL DONE. Results:")
    for r in results:
        print(f"  {r}")
    print(f"{'='*60}", flush=True)
