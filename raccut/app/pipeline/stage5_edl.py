"""Build an action-first EDL and reserve a short closing podium section."""
import json
import math
from pathlib import Path

import numpy as np

from .. import config
from .joblog import jlog
from .cancel import check_cancel

ROLE_CONFIDENCE = 0.65
PODIUM_MAX_SECONDS = 8.0
SELECTION_GAP = 1.0
EXCITING_ACTIONS = {
    "overtake", "close_battle", "pack_racing", "race_start",
    "race_finish", "onboard", "replay",
}


def _overlaps_block(start: float, end: float, blocks: list) -> bool:
    return any(start < block["end"] and end > block["start"]
               for block in blocks)


def _scene_score(a, b, flux_t, flux_s, motion):
    audio = [float(z) for t, z in zip(flux_t, flux_s) if a <= t <= b]
    i0, i1 = max(0, int(a)), min(int(b), len(motion))
    visual = float(np.mean(motion[i0:i1])) if i1 > i0 else 0.0
    # Keep audio transients useful without letting one spike dominate a shot.
    return (float(np.mean(audio)) if audio else 0.0) + visual * 2.0, visual


def compute_candidates(scenes: list, blocked: list, duration: float,
                       flux_t, flux_s, motion,
                       scene_labels: dict | None = None,
                       workdir: Path | None = None,
                       min_action_motion: float | None = None) -> list:
    """Score only confidently identified race-action and podium shots."""
    labels = scene_labels or {}
    motion_floor = (config.get_min_action_motion() if min_action_motion is None
                    else min_action_motion)
    blocks = [{"start": max(0.0, b["start"] - config.PAD_S),
               "end": min(duration, b["end"] + config.PAD_S)}
              for b in blocked]
    candidates = []

    for scene_id, (start, end) in enumerate(zip(scenes, scenes[1:])):
        if workdir is not None and scene_id % 25 == 0:
            check_cancel(workdir)
        label = labels.get(scene_id, {})
        category = label.get("category", "unknown")
        confidence = float(label.get("confidence", 0.0))
        if category not in ("race_action", "podium") or confidence < ROLE_CONFIDENCE:
            continue
        if end - start < config.MIN_CLIP:
            continue

        # Pick the most visually active window while keeping the shot's label.
        window = min(end - start, config.CLIP_WIN)
        if end - start > window:
            step = 1.0
            starts = np.arange(start, end - window + 0.001, step)
            if len(starts) == 0:
                starts = np.array([start])
            motion_means = []
            for pos in starts:
                lo, hi = max(0, int(pos)), min(int(pos + window), len(motion))
                motion_means.append(float(np.mean(motion[lo:hi])) if hi > lo else 0.0)
            clip_start = float(starts[int(np.argmax(motion_means))])
            clip_end = min(end, clip_start + window)
        else:
            clip_start, clip_end = start, end

        if _overlaps_block(clip_start, clip_end, blocks):
            continue
        score, visual_motion = _scene_score(
            clip_start, clip_end, flux_t, flux_s, motion)
        # Exclude slow/neutralized racing shots such as safety-car processions.
        # Podium footage is exempt because it is intentionally a short outro.
        if category == "race_action" and visual_motion < motion_floor:
            continue
        if visual_motion < config.STATIC_MOTION:
            score *= 0.2

        action = label.get("action", "unknown")
        if category == "race_action":
            if action not in EXCITING_ACTIONS:
                continue
            # Make clearly exciting moments outrank routine race coverage.
            if action == "overtake":
                score = score * 1.8 + 1.0
            elif action == "close_battle":
                score = score * 1.6 + 0.8
            elif action == "pack_racing":
                score = score * 1.35 + 0.5
            elif action in ("race_start", "race_finish"):
                score = score * 1.3 + 0.4
            else:
                score *= 1.15
        candidates.append({
            "start": float(clip_start), "end": float(clip_end),
            "score": float(score), "motion": visual_motion,
            "scene_id": scene_id, "category": category, "action": action,
        })
    return candidates


def _can_add(candidate: dict, selected: list, target: float,
             current: float) -> bool:
    length = candidate["end"] - candidate["start"]
    if current + length > target + 1e-6:
        return False
    return not any(candidate["start"] < item["end"] + SELECTION_GAP
                   and candidate["end"] > item["start"] - SELECTION_GAP
                   for item in selected)


def _select_action(candidates: list, target: float, duration: float) -> list:
    """Favor strong shots while spreading the first pass across the race."""
    ranked = sorted(
        candidates,
        key=lambda c: c["score"] * math.sqrt(max(c["end"] - c["start"], 1.0)),
        reverse=True,
    )
    selected = []
    current = 0.0
    bins = max(1, math.ceil(duration / 300.0))
    quota = target * 0.55 / bins
    bucket_use = [0.0] * bins

    # First take the best action from across the full race timeline.
    for candidate in ranked:
        if current >= target:
            break
        bucket = min(bins - 1, int(candidate["start"] / max(duration, 1) * bins))
        length = candidate["end"] - candidate["start"]
        if bucket_use[bucket] + length > quota:
            continue
        if _can_add(candidate, selected, target, current):
            selected.append(candidate)
            current += length
            bucket_use[bucket] += length

    # Use any remaining exciting race footage; interviews/pit shots never enter.
    for candidate in ranked:
        if current >= target:
            break
        if _can_add(candidate, selected, target, current):
            selected.append(candidate)
            current += candidate["end"] - candidate["start"]
    return sorted(selected, key=lambda c: c["start"])


def _select_podium(candidates: list, duration: float) -> list:
    """Take up to 20 seconds from the latest confirmed ceremony shots."""
    selected = []
    used = 0.0
    for candidate in sorted(candidates, key=lambda c: c["start"], reverse=True):
        if candidate["start"] < duration * 0.6:
            continue
        if used >= PODIUM_MAX_SECONDS:
            break
        item = dict(candidate)
        if selected:
            item["end"] = min(item["end"], min(c["start"] for c in selected) - SELECTION_GAP)
        length = item["end"] - item["start"]
        if length < config.MIN_CLIP:
            continue
        allowed = min(length, PODIUM_MAX_SECONDS - used)
        if allowed < config.MIN_CLIP:
            continue
        item["start"] = item["end"] - allowed
        selected.append(item)
        used += allowed
    return sorted(selected, key=lambda c: c["start"])


def run(workdir: Path, scenes: list, audio: dict, blocked: list,
        duration: float, min_action_motion: float | None = None) -> list:
    check_cancel(workdir)
    flux = (np.load(workdir / "flux.npy")
            if (workdir / "flux.npy").exists() else np.array([]))
    motion = (np.load(workdir / "motion.npy")
              if (workdir / "motion.npy").exists() else np.array([]))
    hop = audio.get("hop", 0.5)
    flux_t = np.arange(len(flux)) * hop
    flux_s = flux / (np.median(flux) + 1e-6) if len(flux) else np.array([])

    labels_file = workdir / "race_scenes.json"
    labels = {}
    if labels_file.exists():
        labels = {int(item["scene_id"]): item
                  for item in json.loads(labels_file.read_text())}

    motion_floor = (config.get_min_action_motion()
                    if min_action_motion is None else float(min_action_motion))
    candidates = compute_candidates(
        scenes, blocked, duration, flux_t, flux_s, motion, labels, workdir,
        min_action_motion=motion_floor)
    check_cancel(workdir)
    action_candidates = [c for c in candidates if c["category"] == "race_action"]
    jlog(workdir, f"fast-action motion floor {motion_floor:.1f}; "
         f"{len(action_candidates)} eligible racing clips")
    podium_candidates = [c for c in candidates if c["category"] == "podium"]
    podium = _select_podium(podium_candidates, duration)
    podium_duration = sum(c["end"] - c["start"] for c in podium)
    action_target = max(0.0, config.TARGET_SECONDS - podium_duration)
    action = _select_action(action_candidates, action_target, duration)

    edl = []
    for item in action + podium:
        edl.append({"start": round(item["start"], 3),
                    "end": round(item["end"], 3),
                    "type": item["category"],
                    "action": item["action"]})
    output_duration = sum(item["end"] - item["start"] for item in edl)
    if not action:
        raise RuntimeError(
            "no sufficiently fast, confidently classified racing-action clips; "
            "slow safety-car/neutralized footage, interviews and pit footage "
            "are excluded")

    (workdir / "edl.json").write_text(json.dumps(edl, indent=2))
    podium_note = f", podium outro {podium_duration:.0f}s" if podium else ", no podium detected"
    shortfall = max(0.0, config.TARGET_SECONDS - output_duration)
    jlog(workdir, f"selected {len(action)} racing clips and {len(podium)} podium clips; "
         f"output {output_duration:.0f}s{podium_note}")
    if shortfall > 1.0:
        jlog(workdir, f"only {output_duration:.0f}s of confidently classified action found; "
             "left the rest out instead of adding interviews")
    return edl
