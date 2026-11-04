"""Stage 3: 1fps frame extraction + deterministic heuristics -> suspects.

Heuristics only *nominate* suspects; the VLM (stage 4) convicts.
All processing is per-frame streaming: frames are written to disk as jpg
and iterated one at a time. Peak memory = a handful of frames.

Detectors:
  - fire:      HSV orange/bright region ratio (flicker handled by VLM review)
  - smoke:     large low-saturation mid-brightness gray mass
  - red flag:  large saturated red region (red flag = hard block)
  - yellow flag: large saturated yellow region (mark only, not blocked)
  - freeze:    near-identical consecutive frames (crash aftermath / marshal
               intervention often goes static under broadcast direction)
  - slow replay: sustained low-motion segments (candidate for review if
               combined with close-up / stillness)
Also emits per-second motion scores reused by the EDL as highlight signal.
"""
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .joblog import jlog

FRAME_DIR = "frames"
W = 640  # analysis proxy width


def extract_frames(video: Path, workdir: Path, fps: int):
    d = workdir / FRAME_DIR
    d.mkdir(exist_ok=True)
    r = subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-i", str(video),
        "-vf", f"fps={fps},scale={W}:-1", "-q:v", "4",
        str(d / "f%06d.jpg"),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"frame extract failed: {r.stderr[-500:]}")
    return sorted(d.glob("f*.jpg"))


def _hsv_frac(hsv, lo, hi):
    return float(cv2.inRange(hsv, lo, hi).mean() / 255.0)


def analyze_frame(img: np.ndarray) -> dict:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    fire = _hsv_frac(hsv, np.array([5, 200, 220]), np.array([35, 255, 255]))
    red = _hsv_frac(hsv, np.array([0, 180, 150]), np.array([10, 255, 255])) \
        + _hsv_frac(hsv, np.array([170, 180, 150]), np.array([179, 255, 255]))
    yellow = _hsv_frac(hsv, np.array([20, 200, 200]), np.array([35, 255, 255]))
    # smoke: low saturation, mid-high value, large contiguous area
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = hsv[:, :, 1]
    smoke_mask = ((s < 60) & (gray > 120)).astype(np.uint8)
    smoke = float(smoke_mask.mean())
    return {"fire": fire, "red": red, "yellow": yellow, "smoke": smoke,
            "brightness": float(gray.mean())}


def run(video: Path, workdir: Path, duration: float, fps: int = 1) -> dict:
    frames = extract_frames(video, workdir, fps)
    jlog(workdir, f"extracted {len(frames)} frames @ {fps}fps")
    suspects = []      # per-frame flags
    motion = []        # per-second mean abs diff (highlight signal)
    prev_small = None
    freeze_run = 0
    streak = {"fire": 0, "smoke": 0}  # require 2 consecutive frames

    for i, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        feats = analyze_frame(img)
        small = cv2.resize(img, (160, 90))
        diff = float(cv2.absdiff(small, prev_small).mean()) if prev_small is not None else 0.0
        prev_small = small
        motion.append(diff)

        t = i / fps
        flags = []
        # persistence: fire/smoke need 2 consecutive frames to nominate
        for k, cond in (("fire", feats["fire"] > 0.03),
                        ("smoke", feats["smoke"] > 0.55 and
                                     feats["brightness"] > 100)):
            streak[k] = streak[k] + 1 if cond else 0
            if streak[k] >= 2:
                flags.append(k)
        if feats["red"] > 0.25:
            flags.append("red_flag")
        if feats["yellow"] > 0.25:
            flags.append("yellow_flag")
        if diff < 1.5:
            freeze_run += 1
            if freeze_run >= 4:
                flags.append("freeze")
        else:
            freeze_run = 0
        if 0.1 < diff < 1.5:
            flags.append("low_motion")  # possible slow replay / still
        if flags:
            suspects.append({"t": round(t, 2), "flags": flags,
                             **{k: round(v, 3) for k, v in feats.items()},
                             "frame": str(fp)})
        if i % 300 == 299:
            jlog(workdir, f"scanning {i+1}/{len(frames)} frames, "
                          f"{len(suspects)} suspects so far")

    out = {"suspects": suspects, "motion": motion, "fps": fps,
           "n_frames": len(frames)}
    np.save(workdir / "motion.npy", np.array(motion, dtype=np.float32))
    (workdir / "vision_suspects.json").write_text(json.dumps(out, indent=2))
    return out
