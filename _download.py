import os
import sys
import subprocess

result = subprocess.run(
    ["bash", "-c", "source ~/.bash_aliases && echo $HF_TOKEN"],
    capture_output=True, text=True,
)
token = result.stdout.strip()
if not token:
    print("ERROR: No HF_TOKEN found")
    sys.exit(1)
print(f"Token: {token[:10]}...")

from huggingface_hub import hf_hub_download, list_repo_files

CKPT = "/workspace/world_models/checkpoints"

print("\n=== Oasis ===")
for fname in ["oasis500m.safetensors", "vit-l-20.safetensors"]:
    dest = os.path.join(CKPT, "oasis", fname)
    if os.path.isfile(dest):
        print(f"  Skip (exists): {fname}")
        continue
    try:
        p = hf_hub_download(
            "Etched/oasis-500m", fname,
            local_dir=os.path.join(CKPT, "oasis"),
            token=token,
        )
        sz = os.path.getsize(p) / 1e9
        print(f"  OK {fname}: {sz:.2f} GB")
    except Exception as e:
        print(f"  FAIL {fname}: {e}")

print("\n=== World Engine ===")
try:
    files = list_repo_files("Overworld/Waypoint-1-Small", token=token)
    print(f"  {len(files)} files in repo")
    for fname in files:
        dest = os.path.join(CKPT, "world_engine", fname)
        if os.path.isfile(dest):
            print(f"  Skip (exists): {fname}")
            continue
        try:
            p = hf_hub_download(
                "Overworld/Waypoint-1-Small", fname,
                local_dir=os.path.join(CKPT, "world_engine"),
                token=token,
            )
            sz = os.path.getsize(p) / 1e6
            print(f"  OK {fname}: {sz:.1f} MB")
        except Exception as e:
            print(f"  FAIL {fname}: {e}")
except Exception as e:
    print(f"  Error: {e}")

print("\n=== MineWorld ===")
try:
    files = list_repo_files("microsoft/mineworld", token=token)
    print(f"  {len(files)} files found")
except Exception as e:
    print(f"  Not accessible: {type(e).__name__}")

print("\n=== Totals ===")
for sub in ["oasis", "world_engine", "mineworld"]:
    d = os.path.join(CKPT, sub)
    total = 0
    count = 0
    for root, dirs, fnames in os.walk(d):
        for f in fnames:
            total += os.path.getsize(os.path.join(root, f))
            count += 1
    print(f"  {sub}: {count} files, {total / 1e9:.2f} GB")
