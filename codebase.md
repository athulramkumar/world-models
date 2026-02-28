# Codebase Reference

## Architecture Overview

```
wm_platform/
├── app.py                 # FastAPI + Gradio unified app (entry point)
├── config.py              # Hardware profiles, env registry, path constants
├── engines/
│   ├── base.py            # BaseWorldEngine ABC, Frame, EngineState, EngineStatus
│   ├── worker_protocol.py # JSON-over-stdin/stdout IPC for subprocess workers
│   ├── oasis_engine.py    # Open-Oasis adapter (batch video generation)
│   ├── oasis_worker.py    # Subprocess running in open-oasis venv
│   ├── world_engine_adapter.py  # World Engine adapter (frame-by-frame generation)
│   ├── world_engine_worker.py   # Subprocess running in world_engine venv
│   ├── mineworld_engine.py      # MineWorld adapter (frame-by-frame, Llama-based)
│   └── mineworld_worker.py      # Subprocess running in mineworld venv
├── memflow/
│   ├── types.py           # Core data types: Observation, ObjectState, SceneState, MemoryNode, etc.
│   ├── observer.py        # Wraps engine; records every frame as an Observation
│   ├── extractor.py       # HSV-based object detection + scene classification
│   ├── memory.py          # Graph-based StructuredMemory (nodes + edges + decay)
│   ├── corrector.py       # Reinjects memory state into generation (3 strategies)
│   └── pipeline.py        # Orchestrates Observer → Extractor → Memory → Corrector
├── frontend/
│   ├── dashboard.py       # System status, GPU info, hardware profile, env checks
│   ├── model_explorer.py  # Per-model interactive UIs (MineWorld, Oasis, World Engine)
│   ├── memflow_panel.py   # Memory visualization, interactive persistence tests
│   └── results_viewer.py  # Side-by-side MemFlow comparison viewer + legacy run browser
└── tests/
    ├── test_engines.py          # Unit tests for engine adapters
    ├── test_memflow_kitchen.py  # Object persistence tests (diamond in chest)
    ├── test_memflow_characters.py  # Character persistence tests (Alice & Bob)
    ├── test_full_run.py         # Full integration: engine gen + MemFlow comparison
    └── generate_comparison_videos.py  # Produces long side-by-side comparison videos
```

## Application Entry Point

`wm_platform/app.py` creates a **FastAPI** application with a **Gradio** frontend mounted at `/`.

- FastAPI provides REST API endpoints (`/api/health`, `/api/profiles`, `/api/envs`, `/api/gpu`)
- Gradio provides the interactive web UI with 4 tabs: Dashboard, Model Explorer, MemFlow, Results
- Run via: `python3 -m uvicorn wm_platform.app:app --host 0.0.0.0 --port 7860`

## Engine Architecture

### The Process Isolation Pattern

Each world model runs in **its own Python subprocess** inside its own virtual environment. This is necessary because the models require different PyTorch versions:

| Model        | PyTorch | Virtual Env                |
|:------------|:--------|:---------------------------|
| Open-Oasis  | 2.4.1   | `envs/open-oasis/`        |
| World Engine| 2.6.0   | `envs/world_engine/`      |
| MineWorld   | 2.6.0   | `envs/mineworld/`         |
| Platform    | 2.4.1   | `envs/platform/`          |

The **worker protocol** (`worker_protocol.py`) uses JSON over stdin/stdout:

```
Platform process (platform venv)
  │
  ├──▶ OasisEngine ──stdin/stdout──▶ oasis_worker.py (open-oasis venv)
  ├──▶ WorldEngineAdapter ──stdin/stdout──▶ world_engine_worker.py (world_engine venv)
  └──▶ MineWorldEngine ──stdin/stdout──▶ mineworld_worker.py (mineworld venv)
```

**EngineWorkerClient** manages the subprocess lifecycle:
- `start()`: spawns `subprocess.Popen` with the venv's python binary
- `send_command(cmd, **kwargs)`: writes JSON to stdin, reads JSON from stdout
- `stop()`: sends "shutdown" command then terminates
- A background `_drain_stderr` thread prevents pipe buffer deadlocks

**Frames are transferred as base64-encoded numpy arrays** using `encode_frame()` / `decode_frame()`.

### BaseWorldEngine Interface

All engines implement this abstract interface:

```python
class BaseWorldEngine(ABC):
    def load(model_variant=None, **kwargs) -> None      # Load weights to GPU
    def generate_frame(actions: dict) -> Frame           # Generate one frame
    def get_latents() -> Optional[np.ndarray]            # Get latest latent state
    def inject_conditioning(memory_state) -> None        # MemFlow → engine hook
    def append_context_frame(frame: Frame) -> None       # Add reference frame
    def reset() -> None                                  # Clear state
    def unload() -> None                                 # Release GPU
```

The `Frame` dataclass holds: `rgb` (uint8 H×W×3), optional `latent`, `frame_idx`, `timestamp`, `metadata`.

### Open-Oasis Engine

**Model**: DiT-S/2 (500M params) + ViT-L/20 VAE encoder/decoder

**Generation mode**: **Batch** — generates N frames at once via DDIM sampling.

```python
engine.generate_video(
    prompt_path="path/to/image.png",
    actions_path="path/to/actions.one_hot_actions.pt",
    total_frames=32,
    n_prompt_frames=1,
)
```

- Input: starting frame image + pre-recorded action sequence (24 action channels)
- No text prompt — purely vision + action conditioned
- Generates in chunks of up to 32 frames (limited by `dit_model.max_frames`)
- The worker loads from `.safetensors` checkpoint files
- `generate_frame()` raises `NotImplementedError` — use `generate_video()` instead

**Oasis action keys** (24 channels): `inventory, ESC, hotbar.1-9, forward, back, left, right, cameraX, cameraY, jump, sneak, sprint, swapHands, attack, use, pickItem, drop`

**Chunked generation for long videos**: The `generate_comparison_videos.py` script chains multiple 32-frame chunks together by saving the last frame of each chunk as the prompt image for the next chunk.

### World Engine

**Model**: Overworld/Waypoint-1-Small (DiT + autoencoder + text encoder)

**Generation mode**: **Frame-by-frame** — generates one frame per call.

```python
engine.set_prompt("A cozy Minecraft kitchen")
frame = engine.generate_frame({
    "button": [],          # Set[int] — keyboard buttons
    "mouse": [0.0, 0.0],  # [x, y] mouse delta
    "scroll_wheel": 0,
})
```

- Input: text prompt + control inputs (mouse, keyboard, scroll)
- Text-conditioned — prompt can describe the scene
- `inject_conditioning(memory_state)` calls `set_prompt(memory_state.to_prompt())`
- Much slower than Oasis (~25s per frame vs ~0.7s per frame)

**Known patches required**:
- `CtrlInput.button` must be `Set[int]` — the worker converts incoming lists
- `torch._dynamo.config` attributes wrapped in try/except
- `BlockMask.from_kv_blocks` parameter detection via `inspect.signature`

### MineWorld Engine

**Model**: Llama-based transformer + VQGAN tokenizer

**Generation mode**: Frame-by-frame, action-conditioned (same as MineWorld's action space)

**Status**: Checkpoints temporarily unavailable on HuggingFace (microsoft/mineworld). The engine adapter and worker are fully implemented but cannot be tested until checkpoints return.

---

## MemFlow: Structured State Management

MemFlow is a pipeline that sits between the world model and the user, providing **persistent structured memory** that survives beyond the model's finite context window.

### The Problem

World models have a limited context window (e.g., Open-Oasis: 32 frames ≈ 5 seconds at 6fps). After an object leaves the context window, the model completely forgets it. If a player places a diamond in a chest, walks away for 30 seconds, and returns, the model has no memory of the diamond.

### The Solution: MemFlow Pipeline

```
Engine.generate_frame(actions)
        │
        ▼
  ┌─────────────┐
  │  Observer    │  Records every frame as an Observation
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  Extractor   │  Detects objects, classifies scenes, extracts features
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  Memory      │  Graph database: nodes (objects/locations/entities) + edges (relationships)
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  Corrector   │  Reinjects memory state into the engine
  └─────────────┘
```

### MemFlow Types (`types.py`)

| Type                | Purpose                                         |
|:-------------------|:------------------------------------------------|
| `Observation`      | A single frame + latent + action from the engine |
| `ObjectState`      | Detected object with bbox, features, confidence  |
| `SceneState`       | Scene classification (indoor_wood, outdoor_grass, underground) + objects |
| `ObjectCategory`   | Enum: BLOCK, ITEM, ENTITY, CONTAINER, STRUCTURE, UNKNOWN |
| `MemoryNode`       | Node in memory graph (object, location, or entity) |
| `MemoryEdge`       | Relationship edge: IN, AT, INSIDE, NEAR, OWNS, SAME_AS |
| `MemoryState`      | Snapshot of full graph for handoff to Corrector  |
| `CorrectionStrategy` | Enum: LATENT_NUDGE, FRAME_INJECTION, PROMPT_CONDITIONING |

### Observer (`observer.py`)

Wraps a `BaseWorldEngine` and records every frame:

```python
observer = Observer(engine, window_size=128)
frame, obs = observer.generate_and_observe(actions)
```

- Maintains a sliding window of `Observation` objects (deque, default 128)
- Supports callbacks (`on_observation`)
- Can also passively observe external frames via `observe_frame(frame, action)`

### State Extractor (`extractor.py`)

Derives semantic state from frames using **HSV color segmentation** (lightweight, no ML model):

```python
extractor = StateExtractor(min_object_area=100)
objects = extractor.extract_objects(obs)      # List[ObjectState]
scene = extractor.classify_scene(obs)         # SceneState
changed = extractor.detect_scene_change(obs)  # bool
```

**Object detection**: Matches pixels against HSV color profiles:
- `chest`: orange-brown tones → CONTAINER
- `diamond`: cyan-blue tones → ITEM
- `grass_block`: green tones → BLOCK
- `wood_plank`: light brown tones → BLOCK
- `stone`: gray tones → BLOCK
- `entity_skin`: skin tones → ENTITY

For each match, it finds contours, computes bounding boxes, and creates a **color-histogram feature vector** (48-dim: 16 bins × 3 channels HSV) for re-identification.

**Scene classification**: Computes dominant hue histogram:
- Hue 15-35 → `indoor_wood` (wooden room / kitchen / house)
- Hue 35-85 → `outdoor_grass` (outdoor / grassy area)
- Hue 0-15 → `underground` (cave)

Also computes a **spatial scene feature** (4×4 grid, 24-dim per cell = 384-dim total) for scene change detection.

### Structured Memory (`memory.py`)

An in-memory **graph database** with temporal tracking:

```python
memory = StructuredMemory(decay_rate=0.001, min_confidence=0.05)
```

**Nodes** (`MemoryNode`):
- Have `node_id`, `node_type` (object/location/entity), `label`, `features` (embedding), `confidence`, `observation_count`
- Re-observing a node boosts confidence by +0.1 and updates features/properties

**Edges** (`MemoryEdge`):
- Relationship types: IN, AT, INSIDE, NEAR, OWNS, SAME_AS
- Re-observing an edge boosts confidence by +0.1

**Ingestion from Extractor**:
```python
memory.ingest_scene(scene)                              # Add scene + all its objects
memory.ingest_object_in_container(obj, container_id)    # Diamond INSIDE chest
memory.ingest_entity_at_location(entity, location_id)   # Alice AT kitchen
```

**Confidence Decay**:
```python
memory.decay(current_time=now)
```
- Linear decay: `confidence -= decay_rate × (now - last_seen)`
- Floor at `min_confidence` (default 0.05) — memories never fully vanish
- Without MemFlow, recall drops from 1.0 → 0.0 **instantly** when the object leaves the context window
- With MemFlow, it decays **gradually** over time

**Structured Queries**:
```python
memory.query_objects_at("kitchen")          # All objects at a location (including nested in containers)
memory.query_entities_at("kitchen")         # All entities at a location
memory.query_container_contents("chest_1")  # What's inside a container?
memory.find_nodes(node_type="entity")       # Find all entities
```

**Snapshot**:
```python
state = memory.snapshot()   # → MemoryState
prompt = state.to_prompt()  # → "Current location: kitchen. In the kitchen: chest, diamond. Alice is at the kitchen"
```

**`stats()` returns** (note: key is `nodes`, not `total_nodes`):
```python
{
    "nodes": int,      # total node count
    "edges": int,      # total edge count
    "current_scene": str | None,
    "locations": int,
    "objects": int,
    "entities": int,
}
```

### Corrector (`corrector.py`)

Reinjects memory state into the generation loop using one of **three strategies**:

#### 1. LATENT_NUDGE
- Modifies the current latent tensor toward stored reference patterns
- `delta = reference_latent - current_latent; current += nudge_strength × delta`
- Requires matching latent shapes
- Best for models that expose latent space

#### 2. FRAME_INJECTION (used for Open-Oasis)
- Periodically injects a stored **reference RGB frame** via `engine.append_context_frame()`
- Controlled by `injection_interval` (number of frames between injections)
- Stores reference frames keyed by scene_id
- Anchors the model's visual memory to a known-good frame

#### 3. PROMPT_CONDITIONING (used for World Engine)
- Calls `memory.snapshot().to_prompt()` to render memory as natural language
- Passes to `engine.inject_conditioning()` → `engine.set_prompt(prompt_text)`
- Updates every `injection_interval` frames (default: 5)
- Example output: "Current location: wooden room (kitchen/house). In the wooden room: chest, diamond, wood_plank"

**Usage**:
```python
corrector = Corrector(strategy=CorrectionStrategy.FRAME_INJECTION, injection_interval=30)
corrector.store_reference_frame("kitchen", reference_rgb)

if corrector.should_correct(frame_idx):
    log = corrector.apply(engine, memory_snapshot, current_obs, frame_idx)
```

### MemFlowPipeline (`pipeline.py`)

Orchestrates all components in a single `step()` call:

```python
pipeline = MemFlowPipeline(
    engine,
    strategy=CorrectionStrategy.FRAME_INJECTION,
    enable_correction=True,
    injection_interval=30,
)
pipeline.start()

for action in action_stream:
    frame, obs, scene = pipeline.step(action)

stats = pipeline.get_stats()
snapshot = pipeline.memory.snapshot()
```

Each `step()`:
1. Generates a frame via Observer (which calls `engine.generate_frame`)
2. Extracts scene state via Extractor
3. Ingests into Memory
4. Detects scene changes
5. Applies confidence decay
6. Applies correction if due

---

## Frontend (Gradio UI)

### Dashboard (`dashboard.py`)
- Shows GPU info (from `nvidia-smi`), environment readiness, engine status
- Hardware profile selector dropdown
- Refresh button for live updates

### Model Explorer (`model_explorer.py`)
Three sub-tabs, one per model:

- **MineWorld**: Action checkboxes + camera sliders → generate single frame
- **Open-Oasis**: Prompt image selector + action file selector + frame slider → generate video (gallery)
- **World Engine**: Text prompt + mouse sliders → generate single frame

Each has Load/Unload buttons that manage the subprocess workers.

### MemFlow Panel (`memflow_panel.py`)
- **Memory Visualization**: Renders current memory graph as Markdown tables
- **Object Persistence Test**: Interactive slider (10-300s) → runs diamond-in-chest scenario
- **Character Persistence Test**: Interactive slider (10-300s) → runs Alice-Bob identity scenario

These tests run entirely in the data layer (no GPU needed) — they simulate the passage of time and test recall.

### Results Viewer (`results_viewer.py`)
Main tab for **side-by-side video comparison**:

- Dropdown selects comparison runs (e.g., `comparison_oasis_30s`, `comparison_oasis_60s`, `comparison_we_kitchen_30f`)
- "Load" button triggers data loading (also auto-loads on first visit)
- Two `gr.Video` components side by side: "Without MemFlow" and "With MemFlow"
- Detailed metadata displayed below each video
- Sampled frame galleries for visual inspection

Additional tabs: Object Persistence table, Character Persistence table, individual run browsers for Oasis and World Engine, Confidence Decay explanation.

---

## Test Infrastructure

### Test Results Directory

All outputs go to `test_results/` with standardized structure:

```
test_results/
├── comparison_oasis_30s/        # Side-by-side comparison run
│   ├── without_memflow.mp4      # Baseline video (H.264)
│   ├── with_memflow.mp4         # MemFlow-corrected video (H.264)
│   ├── frames_without/          # Sampled PNG frames
│   ├── frames_with/             # Sampled PNG frames
│   └── metadata.json            # Full generation metadata
├── comparison_we_kitchen_30f/   # World Engine comparison
├── oasis_default_YYYYMMDD/      # Individual Oasis run
├── we_A_cozy_Minecraft_..../    # Individual World Engine run
├── memflow_object_comparison/   # Quantitative object persistence data
├── memflow_character_comparison/# Quantitative character persistence data
└── memflow_summary_YYYYMMDD/    # Summary report with comparison tables
```

### Metadata JSON Structure (Comparison Runs)

```json
{
  "engine": "open_oasis",
  "model": "Oasis 500M (DiT-S/2)",
  "duration_s": 30,
  "fps": 6,
  "prompt_image": "path/to/image.png",
  "actions_file": "path/to/actions.pt",
  "without_memflow": {
    "total_frames": 180,
    "chunk_size": 32,
    "num_chunks": 6,
    "use_memflow": false,
    "generation_time_s": 120.5
  },
  "with_memflow": {
    "total_frames": 180,
    "chunk_size": 32,
    "num_chunks": 6,
    "use_memflow": true,
    "generation_time_s": 135.2,
    "correction_log": [...],
    "final_memory_stats": {"nodes": 42, "edges": 38, ...},
    "final_prompt": "Current location: ..."
  }
}
```

### Video Encoding

All videos **must** be H.264 encoded for browser playback:

```python
# Write raw frames via cv2
writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
# Convert to H.264 via ffmpeg
subprocess.run(["ffmpeg", "-y", "-i", tmp, "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", output])
```

Videos encoded with `mp4v` (MPEG-4 Part 2) will **not play** in browsers. The `ffmpeg` post-processing step is required.

---

## Configuration System (`config.py`)

### Path Constants

```python
ROOT = Path(__file__).parent.parent         # /workspace/world_models
PROFILES_DIR = ROOT / "hardware_profiles"
ENVS_DIR = ROOT / "envs"
REPOS_DIR = ROOT / "repos"
CHECKPOINTS_DIR = ROOT / "checkpoints"
```

### Hardware Profile Auto-Detection

`auto_select_profile()` runs `nvidia-smi` and matches the GPU name against known profiles:
- "a100" → `a100_80gb.yaml`
- "h100" → `h100.yaml`
- "h200" → `h200.yaml`
- Fallback → `a100_80gb.yaml`

Each profile specifies per-engine settings: precision, batch_size, compile, recommended_model, etc.

### Environment Registry

```python
ENV_REGISTRY = {
    "mineworld": EnvInfo(venv_path=ENVS_DIR/"mineworld", repo_path=REPOS_DIR/"mineworld", ...),
    "open_oasis": EnvInfo(venv_path=ENVS_DIR/"open-oasis", repo_path=REPOS_DIR/"open-oasis", ...),
    "world_engine": EnvInfo(venv_path=ENVS_DIR/"world_engine", repo_path=REPOS_DIR/"world_engine", ...),
}
```

`get_env_status()` checks if each venv's python binary exists.

---

## Key Design Decisions and Gotchas

### Worker Stderr Draining
Worker processes print debug output to stderr. If stderr isn't read, the pipe buffer fills (typically 64KB) and the worker **hangs on write**. A daemon thread (`_drain_stderr`) continuously reads stderr to prevent this.

### Action File Format (Oasis)
Oasis action files are `.one_hot_actions.pt` files containing tensors. They must be loaded with `torch.load(..., weights_only=False)` because they contain numpy arrays internally (triggers the `numpy.core.multiarray._reconstruct` unpickling path).

### World Engine CtrlInput
The `CtrlInput` dataclass requires `button` to be `Set[int]`, not `List[int]`. The worker converts incoming JSON arrays to sets. Mouse values are `Tuple[float, float]`.

### Memory Decay Rates
- Object persistence tests use `decay_rate=0.0001` (objects fade in ~60s)
- Character persistence tests use `decay_rate=0.00001` (characters persist much longer)
- `min_confidence=0.05` ensures nothing fully vanishes
- The decay is **linear**: `new_confidence = old_confidence - decay_rate × elapsed_seconds`

### Generation Speed
- **Open-Oasis**: ~0.7s/frame on A100 (batch of 32 frames ≈ 22s)
- **World Engine**: ~25s/frame on A100 (autoregressive, much heavier model)
- MineWorld: untested (checkpoints unavailable)

### Python Buffering
When running long generation scripts, stdout buffering can hide output. Use `PYTHONUNBUFFERED=1 python3 -u script.py` or `print(..., flush=True)`.

---

## How MemFlow Integrates with Each Model

### Open-Oasis + FRAME_INJECTION

```
For each 32-frame chunk:
  1. Generate chunk with engine.generate_video()
  2. For each frame in chunk:
     a. Create Observation from frame RGB
     b. extractor.classify_scene(obs)
     c. memory.ingest_scene(scene)
     d. Store first frame of each scene_id as reference: corrector.store_reference_frame(scene_id, rgb)
  3. Save last frame as PNG → use as prompt for next chunk
  4. Between chunks, the corrector's stored reference frames are available for future injection
```

The FRAME_INJECTION strategy calls `engine.append_context_frame()` which adds a known-good reference frame to Oasis's context window, anchoring its visual memory.

### World Engine + PROMPT_CONDITIONING

```
For each frame:
  1. If corrector.should_correct(frame_idx):
     a. snapshot = memory.snapshot()
     b. prompt = snapshot.to_prompt()
     c. engine.set_prompt(prompt)    ← updates the text conditioning
  2. frame = engine.generate_frame(actions)
  3. obs = Observation(frame_idx, rgb, ...)
  4. scene = extractor.classify_scene(obs)
  5. memory.ingest_scene(scene)
```

The PROMPT_CONDITIONING strategy renders the memory graph as a natural-language description and passes it as the text prompt to World Engine. This keeps the model "aware" of objects and entities that have left the visual field.

### Example Memory Graph After 30s of Oasis Generation

```
Nodes:
  indoor_wood (location, "wooden room (kitchen/house)", confidence=0.85)
  chest_12    (container, "chest", confidence=0.72)
  wood_23     (block, "wood_plank", confidence=0.65)
  entity_5    (entity, "entity_skin", confidence=0.45)

Edges:
  chest_12 —IN→ indoor_wood
  wood_23  —IN→ indoor_wood
  entity_5 —AT→ indoor_wood

Prompt: "Current location: wooden room (kitchen/house). In the wooden room: chest, wood_plank. entity_skin is at the wooden room"
```
