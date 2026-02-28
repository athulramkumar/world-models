"""AI-powered frame quality judge using OpenAI GPT-4o vision."""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import numpy as np
from PIL import Image


def _encode_image_b64(img: np.ndarray | Image.Image, max_size: int = 512) -> str:
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def judge_frame(
    frame: np.ndarray | Image.Image,
    context: str = "Minecraft-style game world",
    api_key: str | None = None,
) -> dict:
    """Judge a single frame for quality using GPT-4o.

    Returns dict with keys: score (1-10), is_meaningful (bool), description, issues.
    """
    import httpx

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OPENAI_API_KEY not set")

    b64 = _encode_image_b64(frame)

    resp = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "gpt-4o",
            "max_tokens": 300,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a video game frame quality judge. Evaluate frames from "
                        "AI world models that generate game-like scenes. Be concise."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Rate this AI-generated game frame (expected context: {context}).\n\n"
                                "Respond in EXACTLY this format (no extra text):\n"
                                "SCORE: <1-10>\n"
                                "MEANINGFUL: <YES/NO>\n"
                                "DESCRIPTION: <one sentence describing what you see>\n"
                                "ISSUES: <one sentence about problems, or NONE>\n\n"
                                "Scoring guide:\n"
                                "- 8-10: Clear, recognizable game scene with objects/terrain\n"
                                "- 5-7: Somewhat recognizable but blurry/degraded\n"
                                "- 3-4: Barely recognizable, mostly flat textures or fog\n"
                                "- 1-2: Complete gibberish, noise, or blank"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                },
            ],
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()

    result = {"raw": text, "score": 5, "is_meaningful": False, "description": "", "issues": ""}
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("SCORE:"):
            try:
                result["score"] = int(line.split(":")[1].strip().split("/")[0].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("MEANINGFUL:"):
            result["is_meaningful"] = "YES" in line.upper()
        elif line.startswith("DESCRIPTION:"):
            result["description"] = line.split(":", 1)[1].strip()
        elif line.startswith("ISSUES:"):
            result["issues"] = line.split(":", 1)[1].strip()

    return result


def judge_video_frames(
    frames: list[np.ndarray],
    sample_count: int = 5,
    context: str = "Minecraft-style game world",
    api_key: str | None = None,
) -> dict:
    """Judge a sample of frames from a video sequence.

    Returns summary with per-frame scores and overall verdict.
    """
    if not frames:
        return {"verdict": "EMPTY", "avg_score": 0, "frame_results": []}

    indices = np.linspace(0, len(frames) - 1, sample_count, dtype=int)
    results = []

    for idx in indices:
        print(f"    Judging frame {idx}/{len(frames)-1}...", flush=True)
        r = judge_frame(frames[idx], context=context, api_key=api_key)
        r["frame_idx"] = int(idx)
        results.append(r)

    scores = [r["score"] for r in results]
    avg = sum(scores) / len(scores) if scores else 0
    meaningful_count = sum(1 for r in results if r["is_meaningful"])

    verdict = "GOOD" if avg >= 6 and meaningful_count >= len(results) // 2 else \
              "ACCEPTABLE" if avg >= 4 else "BAD"

    return {
        "verdict": verdict,
        "avg_score": round(avg, 1),
        "meaningful_ratio": f"{meaningful_count}/{len(results)}",
        "frame_results": results,
    }
