"""Hardware profile management and environment registry."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
PROFILES_DIR = ROOT / "hardware_profiles"
ENVS_DIR = ROOT / "envs"
REPOS_DIR = ROOT / "repos"
CHECKPOINTS_DIR = ROOT / "checkpoints"


@dataclass
class EngineHWConfig:
    precision: str = "fp16"
    batch_size: int = 1
    compile: bool = False
    recommended_model: Optional[str] = None
    max_context_frames: Optional[int] = None
    ddim_steps: Optional[int] = None
    max_frames: Optional[int] = None
    quantization: Optional[str] = None
    model_uri: Optional[str] = None
    notes: str = ""


@dataclass
class GPUInfo:
    name: str = "unknown"
    vram_gb: int = 0
    compute_capability: str = "0.0"
    architecture: str = "unknown"


@dataclass
class HardwareProfile:
    gpu: GPUInfo = field(default_factory=GPUInfo)
    cuda_version: str = "0.0"
    engines: dict[str, EngineHWConfig] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> HardwareProfile:
        with open(path) as f:
            data = yaml.safe_load(f)
        gpu = GPUInfo(**data.get("gpu", {}))
        cuda_ver = data.get("cuda", {}).get("version", "0.0")
        engines = {}
        for name, cfg in data.get("engines", {}).items():
            engines[name] = EngineHWConfig(**cfg)
        return cls(gpu=gpu, cuda_version=cuda_ver, engines=engines)


def detect_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip()
        return out.split("\n")[0]
    except Exception:
        return "unknown"


def auto_select_profile() -> HardwareProfile:
    """Pick the hardware profile matching the detected GPU."""
    gpu_name = detect_gpu_name().lower()
    mapping = {
        "a100": "a100_80gb.yaml",
        "h100": "h100.yaml",
        "h200": "h200.yaml",
    }
    for key, filename in mapping.items():
        if key in gpu_name:
            return HardwareProfile.from_yaml(PROFILES_DIR / filename)
    if (PROFILES_DIR / "a100_80gb.yaml").exists():
        return HardwareProfile.from_yaml(PROFILES_DIR / "a100_80gb.yaml")
    return HardwareProfile()


def load_profile(name: str) -> HardwareProfile:
    path = PROFILES_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Hardware profile not found: {path}")
    return HardwareProfile.from_yaml(path)


def list_profiles() -> list[str]:
    return [p.stem for p in PROFILES_DIR.glob("*.yaml")]


@dataclass
class EnvInfo:
    name: str
    venv_path: Path
    repo_path: Path
    python_bin: Path
    ready: bool = False

    def check_ready(self) -> bool:
        self.ready = self.python_bin.exists()
        return self.ready


ENV_REGISTRY: dict[str, EnvInfo] = {
    "mineworld": EnvInfo(
        name="mineworld",
        venv_path=ENVS_DIR / "mineworld",
        repo_path=REPOS_DIR / "mineworld",
        python_bin=ENVS_DIR / "mineworld" / "bin" / "python",
    ),
    "open_oasis": EnvInfo(
        name="open_oasis",
        venv_path=ENVS_DIR / "open-oasis",
        repo_path=REPOS_DIR / "open-oasis",
        python_bin=ENVS_DIR / "open-oasis" / "bin" / "python",
    ),
    "world_engine": EnvInfo(
        name="world_engine",
        venv_path=ENVS_DIR / "world_engine",
        repo_path=REPOS_DIR / "world_engine",
        python_bin=ENVS_DIR / "world_engine" / "bin" / "python",
    ),
}


def get_env_status() -> dict[str, bool]:
    return {name: info.check_ready() for name, info in ENV_REGISTRY.items()}
