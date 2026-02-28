#!/usr/bin/env python3
"""Generate long comparison videos: with vs without MemFlow for both Oasis and World Engine.

Outputs go to test_results/comparison_<engine>_<duration>s/ with:
  - without_memflow.mp4
  - with_memflow.mp4
  - metadata.json
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
#  Oasis: generate in chunks with/without MemFlow
# ------------------------------------------------------------------ #

def generate_oasis_chunked(
    engine: OasisEngine,
    prompt_path: str,
    actions_path: str,
    total_frames: int,
    chunk_size: int = 32,
    use_memflow: bool = False,
) -> tuple[list[np.ndarray], dict]:
    """Generate frames in chunks. With MemFlow: apply corrections between chunks."""
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

    while len(all_frames) < total_frames:
        remaining = total_frames - len(all_frames)
        n = min(chunk_size, remaining)
        if n < 2:
            break

        print(f"  [{'MF' if use_memflow else 'BL'}] Chunk {chunk_idx}: generating {n} frames "
              f"({len(all_frames)}/{total_frames})")

        frames = engine.generate_video(
            prompt_path=current_prompt,
            actions_path=actions_path,
            total_frames=n,
            n_prompt_frames=1,
        )
        chunk_rgb = [f.rgb for f in frames]
        all_frames.extend(chunk_rgb)

        if use_memflow and extractor and memory and corrector:
            for i, rgb in enumerate(chunk_rgb):
                obs = Observation(
                    frame_idx=len(all_frames) - len(chunk_rgb) + i,
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
                "frame_range": [len(all_frames) - len(chunk_rgb), len(all_frames)],
                "memory_nodes": memory.stats()["nodes"],
                "prompt_from_memory": mem_state.to_prompt()[:200],
            })

        last_frame = chunk_rgb[-1]
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            Image.fromarray(last_frame).save(tmp.name)
            current_prompt = tmp.name

        chunk_idx += 1

    info = {
        "total_frames": len(all_frames),
        "chunk_size": chunk_size,
        "num_chunks": chunk_idx,
        "use_memflow": use_memflow,
    }
    if use_memflow:
        info["correction_log"] = correction_log
        info["final_memory_stats"] = memory.stats()
        info["final_prompt"] = memory.snapshot().to_prompt()

    return all_frames, info


def run_oasis_comparison(duration_s: int):
    """Run Oasis comparison for a given duration."""
    total_frames = duration_s * FPS
    print(f"\n{'='*60}")
    print(f"  OASIS COMPARISON: {duration_s}s ({total_frames} frames)")
    print(f"{'='*60}")

    prompt_path = str(SAMPLE_DIR / "sample_image_0.png")
    actions_path = str(SAMPLE_DIR / "Player729-f153ac423f61-20210806-224813.chunk_000.one_hot_actions.pt")

    run_dir = RESULTS_DIR / f"comparison_oasis_{duration_s}s"
    run_dir.mkdir(exist_ok=True)

    engine = OasisEngine()
    print("Loading Oasis engine...")
    engine.load()

    t0 = time.time()
    print(f"\nGenerating WITHOUT MemFlow ({duration_s}s)...")
    frames_bl, info_bl = generate_oasis_chunked(
        engine, prompt_path, actions_path, total_frames,
        chunk_size=32, use_memflow=False,
    )
    bl_time = time.time() - t0
    info_bl["generation_time_s"] = bl_time
    print(f"  Done: {len(frames_bl)} frames in {bl_time:.1f}s")

    save_video_h264(frames_bl, str(run_dir / "without_memflow.mp4"))
    save_frames_gallery(run_dir, frames_bl, "without", sample_every=max(1, len(frames_bl) // 16))

    engine.reset()

    t0 = time.time()
    print(f"\nGenerating WITH MemFlow ({duration_s}s)...")
    frames_mf, info_mf = generate_oasis_chunked(
        engine, prompt_path, actions_path, total_frames,
        chunk_size=32, use_memflow=True,
    )
    mf_time = time.time() - t0
    info_mf["generation_time_s"] = mf_time
    print(f"  Done: {len(frames_mf)} frames in {mf_time:.1f}s")

    save_video_h264(frames_mf, str(run_dir / "with_memflow.mp4"))
    save_frames_gallery(run_dir, frames_mf, "with", sample_every=max(1, len(frames_mf) // 16))

    engine.unload()

    save_metadata(run_dir, {
        "engine": "open_oasis",
        "model": "Oasis 500M (DiT-S/2)",
        "duration_s": duration_s,
        "fps": FPS,
        "prompt_image": prompt_path,
        "actions_file": actions_path,
        "without_memflow": info_bl,
        "with_memflow": info_mf,
    })

    print(f"\nSaved to {run_dir}")
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
        mouse_x = 0.05 * np.sin(i * 0.3)
        actions = {"button": [], "mouse": [float(mouse_x), 0.0], "scroll_wheel": 0}

        if use_memflow and corrector and memory:
            if corrector.should_correct(i):
                mem_state = memory.snapshot()
                log = corrector.apply(engine, mem_state, frame_idx=i)
                if log:
                    correction_log.append({**log, "frame": i})
                    print(f"    [MF] Corrected at frame {i}: prompt={mem_state.to_prompt()[:80]}")

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
            print(f"  [{tag}] Frame {i}/{total_frames}")

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


def run_we_comparison(prompt: str, total_frames: int, label: str = "default"):
    """Run World Engine comparison."""
    duration_s = total_frames / FPS
    print(f"\n{'='*60}")
    print(f"  WORLD ENGINE COMPARISON: {duration_s:.0f}s ({total_frames} frames)")
    print(f"  Prompt: {prompt}")
    print(f"{'='*60}")

    run_dir = RESULTS_DIR / f"comparison_we_{label}_{total_frames}f"
    run_dir.mkdir(exist_ok=True)

    engine = WorldEngineAdapter()
    print("Loading World Engine...")
    engine.load()

    t0 = time.time()
    print(f"\nGenerating WITHOUT MemFlow...")
    frames_bl, info_bl = generate_we_sequence(engine, prompt, total_frames, use_memflow=False)
    bl_time = time.time() - t0
    info_bl["generation_time_s"] = bl_time
    print(f"  Done: {len(frames_bl)} frames in {bl_time:.1f}s")

    save_video_h264(frames_bl, str(run_dir / "without_memflow.mp4"))
    save_frames_gallery(run_dir, frames_bl, "without", sample_every=max(1, len(frames_bl) // 16))

    engine.reset()

    t0 = time.time()
    print(f"\nGenerating WITH MemFlow...")
    frames_mf, info_mf = generate_we_sequence(engine, prompt, total_frames, use_memflow=True)
    mf_time = time.time() - t0
    info_mf["generation_time_s"] = mf_time
    print(f"  Done: {len(frames_mf)} frames in {mf_time:.1f}s")

    save_video_h264(frames_mf, str(run_dir / "with_memflow.mp4"))
    save_frames_gallery(run_dir, frames_mf, "with", sample_every=max(1, len(frames_mf) // 16))

    engine.unload()

    save_metadata(run_dir, {
        "engine": "world_engine",
        "model": "Overworld/Waypoint-1-Small",
        "duration_s": duration_s,
        "fps": FPS,
        "prompt": prompt,
        "total_frames": total_frames,
        "without_memflow": info_bl,
        "with_memflow": info_mf,
    })

    print(f"\nSaved to {run_dir}")
    return run_dir


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--oasis-durations", nargs="+", type=int, default=[30, 60],
                        help="Oasis video durations in seconds")
    parser.add_argument("--we-frames", type=int, default=30,
                        help="World Engine frames to generate")
    parser.add_argument("--we-prompt", type=str,
                        default="A cozy Minecraft kitchen with a chest containing a diamond")
    parser.add_argument("--skip-oasis", action="store_true")
    parser.add_argument("--skip-we", action="store_true")
    args = parser.parse_args()

    results = []

    if not args.skip_oasis:
        for dur in args.oasis_durations:
            rd = run_oasis_comparison(dur)
            results.append(str(rd))

    if not args.skip_we:
        rd = run_we_comparison(args.we_prompt, args.we_frames, label="kitchen")
        results.append(str(rd))

    print(f"\n{'='*60}")
    print("ALL DONE. Results:")
    for r in results:
        print(f"  {r}")
    print(f"{'='*60}")
