"""Smoke tests for engine adapters -- verify load, generate, status."""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wm_platform.config import get_env_status, CHECKPOINTS_DIR
from wm_platform.engines.base import EngineState
from wm_platform.engines.mineworld_engine import MineWorldEngine
from wm_platform.engines.oasis_engine import OasisEngine
from wm_platform.engines.world_engine_adapter import WorldEngineAdapter


def _skip_if_env_missing(env_name: str):
    status = get_env_status()
    if not status.get(env_name, False):
        pytest.skip(f"{env_name} venv not set up")


def _skip_if_checkpoint_missing(*paths):
    for p in paths:
        if not os.path.isfile(p):
            pytest.skip(f"Checkpoint not found: {p}")


class TestMineWorldEngine:
    def test_status_before_load(self):
        engine = MineWorldEngine()
        assert engine.status.state == EngineState.UNLOADED

    def test_load_missing_checkpoint(self):
        _skip_if_env_missing("mineworld")
        engine = MineWorldEngine()
        engine.load("700M_16f")
        assert engine.status.state == EngineState.ERROR
        assert "not found" in engine.status.error.lower() or "unavailable" in engine.status.error.lower()

    @pytest.mark.skipif(
        not os.path.isfile(CHECKPOINTS_DIR / "mineworld" / "700M_16f.ckpt"),
        reason="MineWorld checkpoint not available",
    )
    def test_load_and_generate(self):
        _skip_if_env_missing("mineworld")
        engine = MineWorldEngine()
        engine.load("700M_16f")
        assert engine.status.state == EngineState.READY

        frame = engine.generate_frame({"forward": 1, "camera": [0, 0]})
        assert frame.rgb.shape[2] == 3
        assert frame.rgb.dtype == np.uint8
        engine.unload()


class TestOasisEngine:
    def test_status_before_load(self):
        engine = OasisEngine()
        assert engine.status.state == EngineState.UNLOADED

    @pytest.mark.skipif(
        os.path.isfile(CHECKPOINTS_DIR / "oasis" / "oasis500m.safetensors"),
        reason="Oasis checkpoints are present -- missing-checkpoint test not applicable",
    )
    def test_load_missing_checkpoint(self):
        _skip_if_env_missing("open_oasis")
        engine = OasisEngine()
        engine.load()
        assert engine.status.state == EngineState.ERROR
        assert "not found" in engine.status.error.lower()

    @pytest.mark.skipif(
        not os.path.isfile(CHECKPOINTS_DIR / "oasis" / "oasis500m.safetensors"),
        reason="Oasis checkpoint not available",
    )
    def test_load_and_generate(self):
        _skip_if_env_missing("open_oasis")
        engine = OasisEngine()
        engine.load()
        assert engine.status.state == EngineState.READY

        repo = engine.repo_path
        frames = engine.generate_video(
            prompt_path=str(repo / "sample_data" / "sample_image_0.png"),
            actions_path=str(repo / "sample_data" / "sample_actions_0.one_hot_actions.pt"),
            total_frames=4,
        )
        assert len(frames) == 4
        assert frames[0].rgb.shape[2] == 3
        engine.unload()


class TestWorldEngineAdapter:
    def test_status_before_load(self):
        engine = WorldEngineAdapter()
        assert engine.status.state == EngineState.UNLOADED

    @pytest.mark.skipif(
        os.environ.get("SKIP_WORLD_ENGINE_LOAD", "1") == "1",
        reason="WorldEngine load requires HF download; set SKIP_WORLD_ENGINE_LOAD=0 to run",
    )
    def test_load_and_generate(self):
        _skip_if_env_missing("world_engine")
        engine = WorldEngineAdapter()
        engine.load(model_uri="Overworld/Waypoint-1-Small")
        assert engine.status.state == EngineState.READY

        engine.set_prompt("A fun Minecraft world")
        frame = engine.generate_frame({"button": [], "mouse": [0.0, 0.0]})
        assert frame.rgb.shape[2] == 3
        assert frame.rgb.dtype == np.uint8
        engine.unload()
