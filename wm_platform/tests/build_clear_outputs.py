#!/usr/bin/env python3
"""Build clear, viewable output videos and a comprehensive results manifest.

1. Re-encodes all comparison mp4s at 30fps (frame interpolation) and 1280x720
2. Creates frame montage grids (early/mid/late) as single PNG images
3. Writes results_manifest.json with prompts, judge scores, and all paths
4. Writes results_manifest.txt as a human-readable summary

Run:
    cd /workspace/world_models
    python3 -u -m wm_platform.tests.build_clear_outputs
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "test_results"
OUTPUT = RESULTS / "final_outputs"
OUTPUT.mkdir(exist_ok=True)


def reencode_video(src: Path, dst: Path, target_fps: int = 30, width: int = 1280, height: int = 720):
    """Re-encode video: upscale to 720p, interpolate to 30fps for smooth playback."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"scale={width}:{height}:flags=lanczos,fps={target_fps}",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  WARN: ffmpeg failed for {src}: {r.stderr[:200]}")
    return dst


def make_montage(video_path: Path, out_path: Path, n_frames: int = 8, cols: int = 4, cell_w: int = 480, cell_h: int = 270):
    """Extract n_frames evenly spaced from a video and tile them into a montage."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, bgr = cap.read()
        if ret:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append((idx, Image.fromarray(rgb).resize((cell_w, cell_h), Image.LANCZOS)))
    cap.release()

    if not frames:
        return

    rows_count = (len(frames) + cols - 1) // cols
    label_h = 28
    img_w = cols * cell_w
    img_h = rows_count * (cell_h + label_h)
    montage = Image.new("RGB", (img_w, img_h), (30, 30, 30))
    draw = ImageDraw.Draw(montage)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, (frame_idx, thumb) in enumerate(frames):
        col = i % cols
        row = i // cols
        x = col * cell_w
        y = row * (cell_h + label_h)
        montage.paste(thumb, (x, y))
        time_s = frame_idx / 6.0
        label = f"Frame {frame_idx} ({time_s:.1f}s)"
        draw.text((x + 5, y + cell_h + 2), label, fill=(220, 220, 220), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(str(out_path), quality=95)
    return out_path


def load_metadata(run_dir: Path) -> dict:
    meta_path = run_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def build_run_entry(run_dir: Path, label: str) -> dict:
    """Build a manifest entry for one comparison run."""
    meta = load_metadata(run_dir)

    entry = {
        "label": label,
        "source_dir": str(run_dir),
    }

    # Engine info
    entry["engine"] = meta.get("engine", "unknown")
    entry["model"] = meta.get("model", "unknown")
    entry["duration_s"] = meta.get("duration_s", 0)
    entry["fps"] = meta.get("fps", 6)
    entry["total_frames"] = meta.get("with_memflow", {}).get("total_frames", 0)

    # Prompt
    if "prompt_image" in meta:
        prompt_img = meta["prompt_image"]
        entry["prompt_image"] = prompt_img
        # Copy prompt to final outputs
        prompt_src = Path(prompt_img)
        if prompt_src.exists():
            prompt_dst = OUTPUT / f"prompt_{label}.png"
            Image.open(prompt_src).save(str(prompt_dst))
            entry["prompt_image_output"] = str(prompt_dst)
    if "prompt" in meta:
        entry["prompt_text"] = meta["prompt"]
    if "prompt_name" in meta:
        entry["prompt_name"] = meta["prompt_name"]
    if "actions_file" in meta:
        entry["actions_file"] = meta["actions_file"]

    # Source videos
    bl_src = run_dir / "without_memflow.mp4"
    mf_src = run_dir / "with_memflow.mp4"

    videos = {}
    for tag, src in [("without_memflow", bl_src), ("with_memflow", mf_src)]:
        if src.exists():
            # Re-encode
            dst = OUTPUT / f"{label}_{tag}_720p_30fps.mp4"
            print(f"  Re-encoding {tag} -> {dst.name}...", flush=True)
            reencode_video(src, dst)
            videos[tag] = {
                "original": str(src),
                "reencoded": str(dst),
            }
            # Montage
            mont = OUTPUT / f"{label}_{tag}_montage.png"
            make_montage(src, mont, n_frames=8, cols=4)
            if mont.exists():
                videos[tag]["montage"] = str(mont)

    entry["videos"] = videos

    # Quality judge results
    quality = meta.get("quality", {})
    for tag in ["without", "with"]:
        q = quality.get(tag, {})
        if q:
            key = f"quality_{tag}_memflow"
            entry[key] = {
                "verdict": q.get("verdict", "N/A"),
                "avg_score": q.get("avg_score", 0),
                "meaningful_ratio": q.get("meaningful_ratio", "N/A"),
            }
            frame_results = q.get("frame_results", [])
            entry[key]["frame_descriptions"] = [
                {
                    "frame_idx": fr.get("frame_idx", 0),
                    "score": fr.get("score", 0),
                    "meaningful": fr.get("is_meaningful", False),
                    "description": fr.get("description", ""),
                    "issues": fr.get("issues", ""),
                }
                for fr in frame_results
            ]

    # MemFlow stats
    mf_info = meta.get("with_memflow", {})
    if "final_memory_stats" in mf_info:
        entry["memflow_graph"] = mf_info["final_memory_stats"]
    if "correction_log" in mf_info:
        entry["memflow_corrections"] = len(mf_info["correction_log"])

    return entry


def write_text_manifest(entries: list[dict], path: Path):
    """Write a human-readable text manifest."""
    lines = []
    lines.append("=" * 80)
    lines.append("  MEMFLOW COMPARISON RESULTS - COMPLETE MANIFEST")
    lines.append("=" * 80)
    lines.append("")

    for entry in entries:
        lines.append("-" * 80)
        lines.append(f"RUN: {entry['label']}")
        lines.append("-" * 80)
        lines.append(f"  Engine:     {entry.get('engine', 'N/A')}")
        lines.append(f"  Model:      {entry.get('model', 'N/A')}")
        lines.append(f"  Duration:   {entry.get('duration_s', 'N/A')}s ({entry.get('total_frames', 'N/A')} frames @ {entry.get('fps', 6)}fps)")

        if "prompt_name" in entry:
            lines.append(f"  Prompt:     {entry['prompt_name']}")
        if "prompt_text" in entry:
            lines.append(f"  Prompt txt: {entry['prompt_text']}")
        if "prompt_image_output" in entry:
            lines.append(f"  Prompt img: {entry['prompt_image_output']}")
        if "actions_file" in entry:
            lines.append(f"  Actions:    {entry['actions_file']}")

        lines.append("")

        # Videos
        for tag_key, tag_label in [("without_memflow", "WITHOUT MemFlow"), ("with_memflow", "WITH MemFlow")]:
            v = entry.get("videos", {}).get(tag_key, {})
            if v:
                lines.append(f"  [{tag_label}]")
                lines.append(f"    Original video: {v.get('original', 'N/A')}")
                lines.append(f"    HD video (720p 30fps): {v.get('reencoded', 'N/A')}")
                if "montage" in v:
                    lines.append(f"    Frame montage: {v['montage']}")

        lines.append("")

        # Quality
        for tag_key, tag_label in [("quality_without_memflow", "WITHOUT MemFlow"), ("quality_with_memflow", "WITH MemFlow")]:
            q = entry.get(tag_key, {})
            if q:
                lines.append(f"  [GPT-4o Quality Judge - {tag_label}]")
                lines.append(f"    Verdict: {q.get('verdict', 'N/A')}")
                lines.append(f"    Average Score: {q.get('avg_score', 'N/A')}/10")
                lines.append(f"    Meaningful: {q.get('meaningful_ratio', 'N/A')}")
                for fd in q.get("frame_descriptions", []):
                    lines.append(f"    Frame {fd['frame_idx']:>4}: {fd['score']}/10 {'[Y]' if fd['meaningful'] else '[N]'} {fd['description']}")
                    if fd.get("issues") and fd["issues"] not in ("", "NONE"):
                        lines.append(f"              Issues: {fd['issues']}")
                lines.append("")

        # MemFlow graph
        if "memflow_graph" in entry:
            g = entry["memflow_graph"]
            lines.append(f"  [MemFlow Graph (final)]")
            lines.append(f"    Nodes: {g.get('nodes', 0)}, Edges: {g.get('edges', 0)}")
            lines.append(f"    Locations: {g.get('locations', 0)}, Objects: {g.get('objects', 0)}, Entities: {g.get('entities', 0)}")
            lines.append(f"    Corrections applied: {entry.get('memflow_corrections', 0)}")

        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    print("Building clear outputs...\n", flush=True)

    # Discover all comparison runs
    comparison_dirs = sorted(RESULTS.glob("comparison_*"))
    entries = []

    for d in comparison_dirs:
        if not d.is_dir():
            continue
        bl = d / "without_memflow.mp4"
        mf = d / "with_memflow.mp4"
        if not bl.exists() and not mf.exists():
            continue

        label = d.name.replace("comparison_", "")
        print(f"\nProcessing: {label}", flush=True)
        entry = build_run_entry(d, label)
        entries.append(entry)

    # Also check for permanence probe results
    for probe_dir in sorted(RESULTS.glob("permanence_probes_*")):
        if not probe_dir.is_dir():
            continue
        for sub in sorted(probe_dir.iterdir()):
            if sub.is_dir() and sub.name.startswith("probe_"):
                results_json = sub / "probe_results.json"
                if results_json.exists():
                    with open(results_json) as f:
                        probe_data = json.load(f)
                    # Copy sample frames to output
                    frames = sorted(sub.glob("frame_*.png"))
                    if frames:
                        montage_cells = []
                        for fp in frames[:8]:
                            montage_cells.append(Image.open(str(fp)))
                        if montage_cells:
                            cell_w, cell_h = 480, 270
                            cols = min(4, len(montage_cells))
                            rows = (len(montage_cells) + cols - 1) // cols
                            label_h = 28
                            mont = Image.new("RGB", (cols * cell_w, rows * (cell_h + label_h)), (30, 30, 30))
                            draw = ImageDraw.Draw(mont)
                            try:
                                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
                            except (OSError, IOError):
                                font = ImageFont.load_default()
                            for i, cell_img in enumerate(montage_cells):
                                c, r_idx = i % cols, i // cols
                                cell_img = cell_img.resize((cell_w, cell_h), Image.LANCZOS)
                                mont.paste(cell_img, (c * cell_w, r_idx * (cell_h + label_h)))
                                fn = frames[i].stem
                                draw.text((c * cell_w + 5, r_idx * (cell_h + label_h) + cell_h + 2),
                                          fn, fill=(220, 220, 220), font=font)
                            mont_path = OUTPUT / f"probe_{sub.name}_montage.png"
                            mont.save(str(mont_path), quality=95)

    # Write JSON manifest
    manifest_json = OUTPUT / "results_manifest.json"
    with open(manifest_json, "w") as f:
        json.dump({"runs": entries, "output_dir": str(OUTPUT)}, f, indent=2, default=str)
    print(f"\nJSON manifest: {manifest_json}", flush=True)

    # Write text manifest
    manifest_txt = OUTPUT / "results_manifest.txt"
    write_text_manifest(entries, manifest_txt)
    print(f"Text manifest: {manifest_txt}", flush=True)

    # Summary
    print(f"\n{'='*60}")
    print(f"  DONE - {len(entries)} runs processed")
    print(f"  Output: {OUTPUT}")
    print(f"{'='*60}")
    print(f"\nFiles in output:")
    for f in sorted(OUTPUT.iterdir()):
        sz = f.stat().st_size
        if sz > 1_000_000:
            print(f"  {f.name:60s} {sz/1_000_000:.1f} MB")
        else:
            print(f"  {f.name:60s} {sz/1_000:.0f} KB")


if __name__ == "__main__":
    main()
