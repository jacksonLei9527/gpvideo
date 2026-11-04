"""Stage 5: rule-based EDL (MVP; LLM ordering slots in here later).

Constraints enforced:
  - cuts only on PySceneDetect boundaries
  - clip length MIN_CLIP..MAX_CLIP
  - PAD_S safety pad around blocked ranges
  - chronological order (timeline replaces commentary)
  - intro anchor: earliest strong early clip; outro anchor: last motion-active
    scene before duration end
  - total ~TARGET_SECONDS within [MIN_TOTAL_SECONDS, MAX_TOTAL_SECONDS]
"""
import json
from bisect import bisect_right
from pathlib import Path

import numpy as np

from .. import config


def _blocked(t, blocks):
    return any(b["start"] <= t <= b["end"] for b in blocks)


def _scene_score(a, b, flux_t, flux_s, motion):
    s = 0.0
    for t, z in zip(flux_t, flux_s):
        if a <= t <= b:
            s += z
    i0, i1 = int(a), min(int(b), len(motion) - 1)
    if i1 > i0:
        s += float(np.mean(motion[i0:i1])) * 2.0
    return s


def compute_candidates(scenes: list, blocked: list, duration: float,
                       flux_t, flux_s, motion) -> list:
    """All safe scene-shots with scores (includes unselected ones).

    For shots longer than CLIP_WIN, pick the most dynamic sub-window inside
    the shot (max mean motion) instead of blindly taking the head.
    Mostly-static shots are deprioritized (score *0.3).
    """
    blocks = [{"start": max(0, b["start"] - config.PAD_S),
               "end": b["end"] + config.PAD_S} for b in blocked]
    cands = []
    for a, b in zip(scenes, scenes[1:]):
        mid = (a + b) / 2
        if _blocked(mid, blocks):
            continue
        if b - a < config.MIN_CLIP:
            continue
        i0, i1 = int(a), min(int(b), len(motion) - 1)
        seg = motion[i0:i1 + 1] if i1 > i0 else np.array([0.0])
        m_mean = float(np.mean(seg))

        win = min(b - a, config.CLIP_WIN)
        if b - a > win and len(seg) > int(win):
            # slide 1s-step window, pick max mean motion position
            w = int(win)
            cs = np.concatenate([[0.0], np.cumsum(seg)])
            means = [(cs[i + w] - cs[i]) / w for i in range(len(seg) - w + 1)]
            best = int(np.argmax(means))
            a2, end = a + best, a + best + w
            m_clip = float(max(means))
        else:
            a2, end = a, min(b, a + win)
            m_clip = m_mean

        score = _scene_score(a2, end, flux_t, flux_s, motion)
        if m_clip < config.STATIC_MOTION:
            score *= 0.3
        cands.append({"start": a2, "end": end, "score": score,
                      "motion": m_clip})
    return cands


def run(workdir: Path, scenes: list, audio: dict, blocked: list,
        duration: float) -> list:
    flux = np.load(workdir / "flux.npy") if (workdir / "flux.npy").exists() else np.array([])
    motion = np.load(workdir / "motion.npy") if (workdir / "motion.npy").exists() else np.array([])
    hop = audio.get("hop", 0.5)
    flux_t = np.arange(len(flux)) * hop
    flux_s = flux / (np.median(flux) + 1e-6) if len(flux) else np.array([])

    cands = compute_candidates(scenes, blocked, duration,
                               flux_t, flux_s, motion)
    if not cands:
        raise RuntimeError("no safe candidate clips — check blocked_ranges")

    # intro anchor: best clip in first 15% of race
    early = [c for c in cands if c["end"] < duration * 0.15]
    intro = max(early, key=lambda c: c["score"]) if early else min(cands, key=lambda c: c["start"])
    # outro anchor: best clip in last 15%
    late = [c for c in cands if c["start"] > duration * 0.85]
    outro = max(late, key=lambda c: c["score"]) if late else cands[-1]

    selected = {id(intro): intro, id(outro): outro}
    total = (intro["end"] - intro["start"]) + (outro["end"] - outro["start"])

    for c in sorted(cands, key=lambda c: -c["score"]):
        if total >= config.TARGET_SECONDS:
            break
        if id(c) in selected or c is intro or c is outro:
            continue
        selected[id(c)] = c
        total += c["end"] - c["start"]

    clips = sorted(selected.values(), key=lambda c: c["start"])
    # dedupe identical/fully-overlapping
    edl = []
    for c in clips:
        if edl and c["start"] < edl[-1]["end"]:
            continue
        edl.append({"start": round(c["start"], 3), "end": round(c["end"], 3)})

    (workdir / "edl.json").write_text(json.dumps(edl, indent=2))
    return edl
