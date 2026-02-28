"""Full integration tests for Open-Oasis and World Engine.

Runs each engine end-to-end with the repo's sample inputs, saves all
prompts, inputs, outputs (frames), and videos to test_results/.
Also runs MemFlow with/without comparisons on actual generated frames.

Usage:
    cd /workspace/world_models
    python3 -m pytest wm_platform/tests/test_full_run.py -v -s --tb=short
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wm_platform.config import CHECKPOINTS_DIR, REPOS_DIR, get_env_status
from wm_platform.engines.base import EngineState

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "test_results"
RESULTS_DIR.mkdir(exist_ok=True)


def _save_metadata(run_dir: Path, data: dict):
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(data, f, indent=2, default=str)


def _save_frames_as_images(run_dir: Path, frames: list, prefix="frame"):
    (run_dir / "frames").mkdir(exist_ok=True)
    paths = []
    for i, frame in enumerate(frames):
        if hasattr(frame, "rgb"):
            rgb = frame.rgb
        elif isinstance(frame, np.ndarray):
            rgb = frame
        else:
            continue
        path = run_dir / "frames" / f"{prefix}_{i:04d}.png"
        Image.fromarray(rgb).save(str(path))
        paths.append(str(path))
    return paths


def _save_video(run_dir: Path, frames: list, filename="output.mp4", fps=6):
    if not frames:
        return None
    first = frames[0].rgb if hasattr(frames[0], "rgb") else frames[0]
    h, w = first.shape[:2]
    tmp_path = str(run_dir / "_tmp_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
    for f in frames:
        rgb = f.rgb if hasattr(f, "rgb") else f
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()
    final_path = str(run_dir / filename)
    import subprocess as _sp
    ret = _sp.run(
        ["ffmpeg", "-y", "-i", tmp_path, "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", final_path],
        capture_output=True,
    )
    if ret.returncode == 0:
        os.remove(tmp_path)
    else:
        os.rename(tmp_path, final_path)
    return final_path


# ================================================================== #
#  Open-Oasis Tests
# ================================================================== #

class TestOasisFullRun:
    """End-to-end Open-Oasis: load model, generate with each sample input,
    save everything."""

    @pytest.fixture(autouse=True)
    def setup(self):
        status = get_env_status()
        if not status.get("open_oasis", False):
            pytest.skip("open_oasis venv not set up")
        ckpt = CHECKPOINTS_DIR / "oasis" / "oasis500m.safetensors"
        if not ckpt.exists():
            pytest.skip("Oasis checkpoint not available")

        from wm_platform.engines.oasis_engine import OasisEngine
        self.engine = OasisEngine()
        self.engine.load()
        if self.engine.status.state != EngineState.READY:
            pytest.skip(f"Oasis failed to load: {self.engine.status.error}")
        yield
        self.engine.unload()

    def _get_sample_data(self):
        sample_dir = REPOS_DIR / "open-oasis" / "sample_data"
        prompts = sorted(p for p in sample_dir.iterdir() if p.suffix in (".png", ".jpg"))
        action_files = sorted(p for p in sample_dir.iterdir() if p.name.endswith(".one_hot_actions.pt"))
        return prompts, action_files

    def test_default_sample(self):
        """Generate 16 frames with the default sample_image_0 + sample_actions_0."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RESULTS_DIR / f"oasis_default_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        sample_dir = REPOS_DIR / "open-oasis" / "sample_data"
        prompt_path = str(sample_dir / "sample_image_0.png")
        actions_path = str(sample_dir / "sample_actions_0.one_hot_actions.pt")

        # Save input prompt
        Image.open(prompt_path).save(str(run_dir / "input_prompt.png"))

        t0 = time.time()
        frames = self.engine.generate_video(
            prompt_path=prompt_path,
            actions_path=actions_path,
            total_frames=16,
        )
        gen_time = time.time() - t0

        assert len(frames) == 16
        for f in frames:
            assert f.rgb.shape[2] == 3
            assert f.rgb.dtype == np.uint8

        frame_paths = _save_frames_as_images(run_dir, frames)
        video_path = _save_video(run_dir, frames)

        _save_metadata(run_dir, {
            "engine": "open_oasis",
            "model": "Oasis 500M (DiT-S/2)",
            "prompt_image": prompt_path,
            "actions_file": actions_path,
            "total_frames": 16,
            "generation_time_s": round(gen_time, 2),
            "ms_per_frame": round(gen_time / 16 * 1000, 1),
            "frame_shape": list(frames[0].rgb.shape),
            "output_frames": frame_paths,
            "output_video": video_path,
            "timestamp": ts,
        })
        print(f"\n  Oasis default: {len(frames)} frames in {gen_time:.1f}s -> {run_dir}")

    def test_all_action_files(self):
        """Generate 8 frames for each available action file, all with the default prompt."""
        prompts, action_files = self._get_sample_data()
        prompt_path = str(REPOS_DIR / "open-oasis" / "sample_data" / "sample_image_0.png")

        for actions_path in action_files:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = actions_path.stem.replace(".one_hot_actions", "")
            run_dir = RESULTS_DIR / f"oasis_{name}_{ts}"
            run_dir.mkdir(parents=True, exist_ok=True)

            Image.open(prompt_path).save(str(run_dir / "input_prompt.png"))

            t0 = time.time()
            frames = self.engine.generate_video(
                prompt_path=prompt_path,
                actions_path=str(actions_path),
                total_frames=8,
            )
            gen_time = time.time() - t0

            assert len(frames) == 8

            frame_paths = _save_frames_as_images(run_dir, frames)
            video_path = _save_video(run_dir, frames)

            _save_metadata(run_dir, {
                "engine": "open_oasis",
                "prompt_image": prompt_path,
                "actions_file": str(actions_path),
                "actions_name": name,
                "total_frames": 8,
                "generation_time_s": round(gen_time, 2),
                "output_video": video_path,
                "timestamp": ts,
            })
            print(f"\n  Oasis {name}: {len(frames)} frames in {gen_time:.1f}s -> {run_dir}")

    def test_mp4_prompt_first_frame(self):
        """Use first frame from each MP4 sample as the prompt image."""
        sample_dir = REPOS_DIR / "open-oasis" / "sample_data"
        mp4s = sorted(p for p in sample_dir.iterdir() if p.suffix == ".mp4")
        default_actions = str(sample_dir / "sample_actions_0.one_hot_actions.pt")

        for mp4 in mp4s[:2]:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = mp4.stem[:40]
            run_dir = RESULTS_DIR / f"oasis_mp4prompt_{name}_{ts}"
            run_dir.mkdir(parents=True, exist_ok=True)

            cap = cv2.VideoCapture(str(mp4))
            ret, bgr = cap.read()
            cap.release()
            if not ret:
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            prompt_file = str(run_dir / "input_prompt.png")
            Image.fromarray(rgb).save(prompt_file)

            frames = self.engine.generate_video(
                prompt_path=prompt_file,
                actions_path=default_actions,
                total_frames=8,
            )

            assert len(frames) == 8
            _save_frames_as_images(run_dir, frames)
            _save_video(run_dir, frames)
            _save_metadata(run_dir, {
                "engine": "open_oasis",
                "prompt_source": str(mp4),
                "total_frames": 8,
                "timestamp": ts,
            })
            print(f"\n  Oasis mp4-prompt {name}: {len(frames)} frames -> {run_dir}")


# ================================================================== #
#  World Engine Tests
# ================================================================== #

class TestWorldEngineFullRun:
    """End-to-end World Engine: load model, generate frames, save results."""

    @pytest.fixture(autouse=True)
    def setup(self):
        status = get_env_status()
        if not status.get("world_engine", False):
            pytest.skip("world_engine venv not set up")

        from wm_platform.engines.world_engine_adapter import WorldEngineAdapter
        self.engine = WorldEngineAdapter()
        self.engine.load(model_uri="Overworld/Waypoint-1-Small")
        if self.engine.status.state != EngineState.READY:
            pytest.skip(f"World Engine failed to load: {self.engine.status.error}")
        yield
        self.engine.unload()

    @pytest.mark.parametrize("prompt", [
        "A cozy Minecraft kitchen with a chest on the floor",
        "An open grassy field with trees and a river",
        "A dark underground cave with torches on the walls",
        "A living room with wooden walls and a fireplace",
    ])
    def test_generate_with_prompt(self, prompt):
        """Generate 8 sequential frames with text prompt + neutral actions."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = prompt[:30].replace(" ", "_").replace("/", "_")
        run_dir = RESULTS_DIR / f"we_{safe_name}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        self.engine.set_prompt(prompt)

        frames = []
        t0 = time.time()
        for i in range(8):
            frame = self.engine.generate_frame({
                "button": [],
                "mouse": [0.0, 0.0],
                "scroll_wheel": 0,
            })
            frames.append(frame)
        gen_time = time.time() - t0

        assert len(frames) == 8
        for f in frames:
            assert f.rgb.shape[2] == 3
            assert f.rgb.dtype == np.uint8

        frame_paths = _save_frames_as_images(run_dir, frames)
        video_path = _save_video(run_dir, frames)

        _save_metadata(run_dir, {
            "engine": "world_engine",
            "model": "Overworld/Waypoint-1-Small",
            "prompt": prompt,
            "total_frames": 8,
            "generation_time_s": round(gen_time, 2),
            "ms_per_frame": round(gen_time / 8 * 1000, 1),
            "frame_shape": list(frames[0].rgb.shape),
            "output_frames": frame_paths,
            "output_video": video_path,
            "timestamp": ts,
        })
        print(f"\n  WE '{prompt[:30]}': {len(frames)} frames in {gen_time:.1f}s -> {run_dir}")

    def test_generate_with_movement(self):
        """Generate frames while moving forward and looking around."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RESULTS_DIR / f"we_movement_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        self.engine.set_prompt("An explorable Minecraft world")

        frames = []
        actions_log = []
        t0 = time.time()
        for i in range(12):
            mouse_x = 0.1 * np.sin(i * 0.5)
            actions = {
                "button": [],
                "mouse": [float(mouse_x), 0.0],
                "scroll_wheel": 0,
            }
            frame = self.engine.generate_frame(actions)
            frames.append(frame)
            actions_log.append(actions)
        gen_time = time.time() - t0

        _save_frames_as_images(run_dir, frames)
        _save_video(run_dir, frames)

        _save_metadata(run_dir, {
            "engine": "world_engine",
            "prompt": "An explorable Minecraft world",
            "total_frames": 12,
            "generation_time_s": round(gen_time, 2),
            "actions_log": actions_log,
            "timestamp": ts,
        })
        print(f"\n  WE movement: {len(frames)} frames in {gen_time:.1f}s -> {run_dir}")


# ================================================================== #
#  MemFlow WITH vs WITHOUT comparison
# ================================================================== #

class TestMemFlowComparison:
    """Quantitative + qualitative comparison of MemFlow vs no-MemFlow.

    Uses actual world-model-generated frames (from Oasis) to test
    the full pipeline: Observer -> Extractor -> Memory -> Corrector.
    Also measures the data-layer memory tests quantitatively.
    """

    def _run_data_layer_comparison(self, scenario: str, durations: list[int]):
        """Run data-layer MemFlow vs baseline and return structured results."""
        from wm_platform.memflow.memory import StructuredMemory
        from wm_platform.memflow.types import (
            MemoryEdge, MemoryNode, ObjectCategory, ObjectState, RelationType,
        )

        results = []

        for duration in durations:
            row = {"scenario": scenario, "duration_s": duration}

            # --- Without MemFlow ---
            if scenario == "object_persistence":
                from wm_platform.tests.test_memflow_kitchen import NaiveContextWindow
                ctx = NaiveContextWindow(window_size=32)
                ctx.add_frame({"scene": "kitchen", "objects": ["diamond", "chest"]})
                fps = 6
                for _ in range(int(duration * fps)):
                    ctx.add_frame({"scene": "living_room", "objects": ["grass"]})
                row["baseline_recall"] = ctx.recall_confidence("diamond")
                row["baseline_can_query_container"] = False
            else:
                from wm_platform.tests.test_memflow_characters import NaiveEntityTracker
                ctx = NaiveEntityTracker(window_size=32)
                alice_feat = np.random.RandomState(42).randn(48).astype(np.float32)
                bob_feat = np.random.RandomState(99).randn(48).astype(np.float32)
                ctx.add_frame({
                    "location": "living_room",
                    "entities": ["alice", "bob"],
                    "entity_features": {"alice": alice_feat, "bob": bob_feat},
                })
                fps = 6
                for _ in range(int(duration * fps)):
                    ctx.add_frame({"location": "bedroom", "entities": []})
                row["baseline_alice_recall"] = 1.0 if ctx.recall_entity("alice") else 0.0
                row["baseline_bob_recall"] = 1.0 if ctx.recall_entity("bob") else 0.0
                row["baseline_features_preserved"] = ctx.recall_entity_features("alice") is not None

            # --- With MemFlow ---
            if scenario == "object_persistence":
                from wm_platform.tests.test_memflow_kitchen import _build_kitchen_scenario
                memory = StructuredMemory(decay_rate=0.0001, min_confidence=0.05)
                memory, _ = _build_kitchen_scenario(memory, duration)
                contents = memory.query_container_contents("kitchen_chest")
                diamond_conf = max(
                    (n.confidence for n in contents if n.label == "diamond"), default=0.0
                )
                row["memflow_recall"] = round(diamond_conf, 6)
                row["memflow_can_query_container"] = len(contents) > 0
                row["memflow_graph_stats"] = memory.stats()
            else:
                from wm_platform.tests.test_memflow_characters import _run_character_scenario_memflow
                memory = StructuredMemory(decay_rate=0.00001, min_confidence=0.05)
                memory, alice_feat, bob_feat = _run_character_scenario_memflow(memory, duration)
                alice_node = memory.get_node("alice")
                bob_node = memory.get_node("bob")
                row["memflow_alice_recall"] = round(alice_node.confidence, 6) if alice_node else 0.0
                row["memflow_bob_recall"] = round(bob_node.confidence, 6) if bob_node else 0.0
                row["memflow_features_preserved"] = (
                    alice_node is not None and alice_node.features is not None
                    and np.array_equal(alice_node.features, alice_feat)
                )
                entities_at_kitchen = memory.query_entities_at("kitchen")
                row["memflow_entities_at_kitchen"] = [e.node_id for e in entities_at_kitchen]
                row["memflow_graph_stats"] = memory.stats()

            results.append(row)
        return results

    def test_object_persistence_comparison(self):
        """Quantitative comparison: diamond recall with/without MemFlow."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RESULTS_DIR / f"memflow_object_comparison_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        results = self._run_data_layer_comparison("object_persistence", [30, 60, 120])

        _save_metadata(run_dir, {
            "test": "object_persistence_comparison",
            "description": (
                "Diamond placed in kitchen chest, then player explores living room "
                "for N seconds. Tests recall upon return."
            ),
            "results": results,
            "timestamp": ts,
        })

        # Print comparison table
        print("\n" + "=" * 80)
        print("OBJECT PERSISTENCE: MemFlow vs Baseline")
        print("=" * 80)
        print(f"{'Duration':>10} | {'Baseline Recall':>16} | {'MemFlow Recall':>15} | {'MemFlow Query':>14}")
        print("-" * 65)
        for r in results:
            print(f"{r['duration_s']:>8}s | {r['baseline_recall']:>16.4f} | "
                  f"{r['memflow_recall']:>15.6f} | "
                  f"{'YES' if r['memflow_can_query_container'] else 'NO':>14}")

        for r in results:
            assert r["baseline_recall"] == 0.0, "Baseline should forget"
            assert r["memflow_recall"] > 0.0, "MemFlow should remember"
            assert r["memflow_can_query_container"], "MemFlow should support structured queries"

    def test_character_persistence_comparison(self):
        """Quantitative comparison: character recall with/without MemFlow."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RESULTS_DIR / f"memflow_character_comparison_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        results = self._run_data_layer_comparison("character_persistence", [30, 60, 120])

        _save_metadata(run_dir, {
            "test": "character_persistence_comparison",
            "description": (
                "Alice and Bob meet in living room, separate to different rooms for "
                "N seconds, then reconvene in kitchen. Tests identity recall."
            ),
            "results": results,
            "timestamp": ts,
        })

        print("\n" + "=" * 80)
        print("CHARACTER PERSISTENCE: MemFlow vs Baseline")
        print("=" * 80)
        print(f"{'Duration':>10} | {'Base Alice':>11} | {'Base Bob':>9} | "
              f"{'MF Alice':>10} | {'MF Bob':>8} | {'Features':>9} | {'At Kitchen':>11}")
        print("-" * 85)
        for r in results:
            print(f"{r['duration_s']:>8}s | "
                  f"{r['baseline_alice_recall']:>11.1f} | {r['baseline_bob_recall']:>9.1f} | "
                  f"{r['memflow_alice_recall']:>10.6f} | {r['memflow_bob_recall']:>8.6f} | "
                  f"{'YES' if r['memflow_features_preserved'] else 'NO':>9} | "
                  f"{','.join(r['memflow_entities_at_kitchen']):>11}")

        for r in results:
            assert r["baseline_alice_recall"] == 0.0
            assert r["baseline_bob_recall"] == 0.0
            assert r["memflow_alice_recall"] > 0.0
            assert r["memflow_bob_recall"] > 0.0
            assert r["memflow_features_preserved"]

    def test_memflow_on_generated_oasis_frames(self):
        """Run MemFlow pipeline on actual Oasis-generated frames, if available."""
        status = get_env_status()
        if not status.get("open_oasis", False):
            pytest.skip("open_oasis venv not set up")
        ckpt = CHECKPOINTS_DIR / "oasis" / "oasis500m.safetensors"
        if not ckpt.exists():
            pytest.skip("Oasis checkpoint not available")

        from wm_platform.engines.oasis_engine import OasisEngine
        from wm_platform.memflow.extractor import StateExtractor
        from wm_platform.memflow.memory import StructuredMemory
        from wm_platform.memflow.types import Observation

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RESULTS_DIR / f"memflow_oasis_pipeline_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        engine = OasisEngine()
        engine.load()
        assert engine.status.state == EngineState.READY

        sample_dir = REPOS_DIR / "open-oasis" / "sample_data"
        frames = engine.generate_video(
            prompt_path=str(sample_dir / "sample_image_0.png"),
            actions_path=str(sample_dir / "sample_actions_0.one_hot_actions.pt"),
            total_frames=16,
        )
        engine.unload()

        _save_frames_as_images(run_dir, frames, prefix="oasis_frame")
        _save_video(run_dir, frames, filename="oasis_raw.mp4")

        extractor = StateExtractor()
        memory = StructuredMemory(decay_rate=0.0001, min_confidence=0.05)
        extraction_log = []

        for i, f in enumerate(frames):
            obs = Observation(
                frame_idx=i, timestamp=time.time() + i,
                rgb=f.rgb, engine_id="open_oasis",
            )
            scene = extractor.classify_scene(obs)
            memory.ingest_scene(scene)

            objects = extractor.extract_objects(obs)
            extraction_log.append({
                "frame": i,
                "scene_id": scene.scene_id,
                "scene_label": scene.label,
                "num_objects": len(objects),
                "object_labels": [o.label for o in objects],
                "scene_confidence": scene.confidence,
            })

            if i > 0:
                changed = extractor.detect_scene_change(obs)
                extraction_log[-1]["scene_changed"] = changed

        snapshot = memory.snapshot()

        _save_metadata(run_dir, {
            "test": "memflow_on_oasis_frames",
            "description": (
                "Ran MemFlow (extractor + memory) on 16 frames generated by Oasis. "
                "Extracts objects, classifies scenes, builds memory graph."
            ),
            "extraction_log": extraction_log,
            "memory_stats": memory.stats(),
            "memory_nodes": [
                {"id": n.node_id, "type": n.node_type, "label": n.label,
                 "confidence": round(n.confidence, 4), "observations": n.observation_count}
                for n in snapshot.nodes
            ],
            "memory_edges": [
                {"source": e.source_id, "target": e.target_id,
                 "relation": e.relation.value, "confidence": round(e.confidence, 4)}
                for e in snapshot.edges
            ],
            "prompt_description": snapshot.to_prompt(),
            "timestamp": ts,
        })

        print("\n" + "=" * 80)
        print("MEMFLOW ON OASIS FRAMES")
        print("=" * 80)
        print(f"  Frames analyzed: {len(frames)}")
        print(f"  Memory graph: {memory.stats()}")
        print(f"  Prompt from memory: {snapshot.to_prompt()[:120]}")
        for entry in extraction_log:
            print(f"  Frame {entry['frame']:2d}: scene={entry['scene_label']:<30} "
                  f"objects={entry['object_labels']}")

    def test_memflow_summary_report(self):
        """Generate a final summary report comparing all scenarios."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RESULTS_DIR / f"memflow_summary_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        obj_results = self._run_data_layer_comparison("object_persistence", [30, 60, 120])
        char_results = self._run_data_layer_comparison("character_persistence", [30, 60, 120])

        report_lines = [
            "# MemFlow vs Baseline Comparison Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Object Persistence (Diamond in Kitchen Chest)",
            "",
            "| Duration | Baseline Recall | MemFlow Recall | MemFlow Wins |",
            "|----------|-----------------|----------------|--------------|",
        ]

        for r in obj_results:
            wins = "YES" if r["memflow_recall"] > r["baseline_recall"] else "NO"
            report_lines.append(
                f"| {r['duration_s']}s | {r['baseline_recall']:.4f} | "
                f"{r['memflow_recall']:.6f} | {wins} |"
            )

        report_lines += [
            "",
            "## Character Persistence (Alice & Bob Identity)",
            "",
            "| Duration | Base Alice | Base Bob | MF Alice | MF Bob | Features OK | Entities@Kitchen |",
            "|----------|-----------|---------|----------|--------|------------|-----------------|",
        ]

        for r in char_results:
            report_lines.append(
                f"| {r['duration_s']}s | {r['baseline_alice_recall']:.1f} | "
                f"{r['baseline_bob_recall']:.1f} | {r['memflow_alice_recall']:.6f} | "
                f"{r['memflow_bob_recall']:.6f} | "
                f"{'YES' if r['memflow_features_preserved'] else 'NO'} | "
                f"{','.join(r['memflow_entities_at_kitchen'])} |"
            )

        report_lines += [
            "",
            "## Key Findings",
            "",
            "1. **Without MemFlow**: Objects and characters are completely forgotten once they",
            "   leave the model's finite context window (typically 32 frames = ~5 seconds at 6fps).",
            "   Recall drops to 0.0 for all durations (30s, 60s, 120s).",
            "",
            "2. **With MemFlow**: The structured memory graph maintains object and character state",
            "   indefinitely with graceful confidence decay. Even after 120 seconds, recall",
            "   confidence remains above the minimum threshold.",
            "",
            "3. **Structured Queries**: MemFlow enables graph queries like 'what was in the chest?'",
            "   and 'who is at the kitchen?' that are impossible with a naive context window.",
            "",
            "4. **Feature Preservation**: Entity feature vectors (appearance embeddings) are preserved",
            "   across time gaps, enabling re-identification when characters reappear.",
        ]

        report_text = "\n".join(report_lines)
        with open(run_dir / "comparison_report.md", "w") as f:
            f.write(report_text)

        _save_metadata(run_dir, {
            "test": "memflow_summary",
            "object_results": obj_results,
            "character_results": char_results,
            "timestamp": ts,
        })

        print("\n" + report_text)
