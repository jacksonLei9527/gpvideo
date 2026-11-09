"""Stage 4: VLM second review of suspect segments via OpenRouter.

Groups suspect frames into contiguous ranges, samples 3-5 frames per range,
asks a strict yes/no question, collects verdicts -> blocked_ranges.md.

Hard rule: 'red_flag'/'fire'/'smoke' flags with VLM=yes, OR 'freeze'
confirmed as incident -> blocked. yellow_flag marks but never blocks alone.
Unconfirmed freeze/low_motion are NOT blocked (racing has legit stills).
"""
import base64
import json
import os
import time
from pathlib import Path

import httpx

VLM_WORKERS = int(os.environ.get("RACCUT_VLM_WORKERS", "3"))

from .. import config
from .joblog import jlog
from . import llmstats
from .cancel import JobCancelled, check_cancel

PROMPT = (
    "These are frames from a motorcycle/car racing broadcast. Answer strictly "
    "in JSON: {\"verdict\": \"yes\" or \"no\", \"reason_en\": \"short text\", "
    "\"reason_zh_hant\": \"繁體中文\", \"reason_zh_hans\": \"简体中文\", "
    "\"reason_pt\": \"português\"}. "
    "Say yes if ANY of these appear: heavy smoke; fire/flames; a crashed or "
    "damaged vehicle lying/stopped on the track or gravel; a rider/driver "
    "down, sitting, or walking away from a crashed vehicle on the racing "
    "surface; marshals, officials, or medical staff attending on the track "
    "or runoff area; debris or vehicle parts scattered on track; recovery "
    "vehicle/crane actively working on the track or runoff (lifting a "
    "vehicle, crew gathered around it); stretcher; blood; close-up of "
    "medical rescue; or any post-crash cleanup scene. "
    "Say no for: normal pit stops inside pit lane/garage with crew around an "
    "intact vehicle; podium; crowd; flags alone; a stationary crane or "
    "recovery vehicle merely parked beside the circuit with no incident "
    "activity around it; normal racing footage."
)

def _wait_cancel(workdir: Path, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        check_cancel(workdir)
        time.sleep(min(0.25, deadline - time.monotonic()))


CLEANUP_TAIL_S = 45.0  # post-incident cleanup window

SCENE_CLASSIFY_BATCH = 8
SCENE_CLASSIFY_PROMPT = (
    "Classify each numbered scene from these racing-broadcast frames. Frames "
    "with the same scene ID are from one shot. Return only JSON in this "
    "schema: {\"scenes\":[{\"scene_id\":0,\"category\":\"race_action\","
    "\"action\":\"overtake\",\"confidence\":0.0}]}. Allowed categories: "
    "race_action, podium, interview, pit, other. Allowed actions: overtake, "
    "close_battle, pack_racing, race_start, race_finish, onboard, replay, safety_car, "
    "unknown. Use race_action only for visually compelling competition: "
    "overtakes, close battles, pack racing, starts/finishes, or an engaging "
    "onboard. A distant lone vehicle circulating without a contest is other. "
    "Replays count only when they show an exciting racing moment. Identify a safety car when a "
    "roof-mounted flashing beacon or light bar is visible AND that car is leading a pack with race cars "
    "following. When both cues are visible, label category other and action safety_car even if many cars "
    "are in frame. Do not classify a neutralized procession, formation lap, or yellow-flag controlled-speed "
    "following as race_action or pack_racing. A group of cars alone or an unclear roof light is not enough; "
    "require visible high-speed competition, close battles, or overtaking for race_action. Use interview for a driver, "
    "team member, or reporter speaking to camera or another person rather "
    "than showing active competition, even if wearing racing "
    "gear. Use pit for garage, pit-lane service, or stationary preparations. "
    "Use podium only for the actual post-race podium/award ceremony with "
    "winners, trophies, or champagne; post-race interviews are interviews. "
    "Do not infer action from graphics or a person merely wearing a helmet. "
    "Set confidence from 0 to 1; use other/unknown when uncertain. Include "
    "exactly one result for every scene ID provided."
)
SCENE_CATEGORIES = {"race_action", "podium", "interview", "pit", "other"}


class SceneClassificationFormatError(ValueError):
    """A successful API response could not be mapped reliably to scene IDs."""


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _call_vlm(workdir: Path, images: list[Path]) -> dict:
    check_cancel(workdir)
    key = config.load_openrouter_key()
    content = [{"type": "text", "text": PROMPT}]
    for p in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_b64(p)}"},
        })
    r = None
    with llmstats.timer() as t:
        err = None
        for attempt in range(5):
            check_cancel(workdir)
            try:
                r = httpx.post(
                    config.OPENROUTER_BASE,
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": config.VLM_MODEL,
                          "messages": [{"role": "user", "content": content}],
                          "temperature": 0},
                    timeout=30,
                )
                check_cancel(workdir)
                r.raise_for_status()
                break
            except Exception as e:
                err = e
                resp = getattr(e, "response", None)
                if resp is not None and resp.status_code == 429:
                    ra = resp.headers.get("retry-after")
                    wait = int(ra) if ra and ra.isdigit() else \
                        min(60, 8 * (attempt + 1))
                    _wait_cancel(workdir, wait)
                    continue
                if attempt == 0:
                    continue  # one free retry for transient errors
                llmstats.record(workdir, "vlm_review", len(images),
                                t.latency, False)
                raise
        else:
            llmstats.record(workdir, "vlm_review", len(images),
                            t.latency, False)
            raise err
    llmstats.record(workdir, "vlm_review", len(images), t.latency, True)
    text = r.json()["choices"][0]["message"]["content"]
    try:
        start = text.find("{")
        v = json.loads(text[start:text.rfind("}") + 1])
        v.setdefault("reason", v.get("reason_en", ""))
        return v
    except Exception:
        return {"verdict": "no", "reason": f"unparseable: {text[:80]}",
                "reason_en": f"unparseable: {text[:80]}"}


def group_ranges(suspects: list, fps: int, gap_s: float = 5.0) -> list:
    """Merge per-frame suspects into [start,end,flags] ranges."""
    ranges = []
    for s in suspects:
        # low_motion alone is not review-worthy
        real = [f for f in s["flags"] if f != "low_motion"]
        if not real:
            continue
        if ranges and s["t"] - ranges[-1]["end"] <= gap_s:
            ranges[-1]["end"] = s["t"]
            ranges[-1]["flags"] = sorted(set(ranges[-1]["flags"]) | set(real))
            ranges[-1]["frames"].append(s["frame"])
        else:
            ranges.append({"start": s["t"], "end": s["t"] + 1.0 / fps,
                           "flags": real, "frames": [s["frame"]]})
    return ranges


def _sample(frames: list, n: int = 4) -> list:
    if len(frames) <= n:
        return frames
    step = len(frames) / n
    return [frames[int(i * step)] for i in range(n)]


def _json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model response did not contain a JSON object")
    return json.loads(text[start:end + 1])


def _json_fallback(value):
    """Serialize filesystem and numeric scalar values safely in job artifacts."""
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _classify_scene_batch(workdir: Path, batch: list) -> list:
    check_cancel(workdir)
    key = config.load_openrouter_key()
    content = [{"type": "text", "text": SCENE_CLASSIFY_PROMPT}]
    n_images = 0
    for item in batch:
        for frame in item["frames"]:
            content.append({"type": "text", "text": f"Scene ID {item['scene_id']} reference frame:"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_b64(frame)}"},
            })
            n_images += 1

    with llmstats.timer() as timer:
        try:
            response = None
            error = None
            for attempt in range(5):
                check_cancel(workdir)
                try:
                    response = httpx.post(
                        config.OPENROUTER_BASE,
                        headers={"Authorization": f"Bearer {key}"},
                        json={"model": config.VLM_MODEL,
                              "messages": [{"role": "user", "content": content}],
                              "temperature": 0, "max_tokens": 1600},
                        timeout=30,
                    )
                    check_cancel(workdir)
                    response.raise_for_status()
                    break
                except Exception as exc:
                    error = exc
                    resp = getattr(exc, "response", None)
                    if resp is not None and resp.status_code == 429:
                        retry_after = resp.headers.get("retry-after")
                        wait = int(retry_after) if retry_after and retry_after.isdigit() \
                            else min(30, 2 ** (attempt + 1))
                        _wait_cancel(workdir, wait)
                    elif attempt < 4:
                        _wait_cancel(workdir, min(8, 2 ** attempt))
            else:
                raise error
            try:
                parsed = _json_object(
                    response.json()["choices"][0]["message"]["content"])
                rows = parsed.get("scenes") if isinstance(parsed, dict) else None
                if not isinstance(rows, list):
                    raise ValueError("response has no scenes list")
            except Exception as exc:
                raise SceneClassificationFormatError(str(exc)) from exc
            llmstats.record(workdir, "scene_classification", n_images,
                            timer.latency, True)
            return rows
        except Exception:
            llmstats.record(workdir, "scene_classification", n_images,
                            timer.latency, False)
            raise


def classify_scenes(workdir: Path, scenes: list, vision: dict) -> list:
    """Label each usable shot so interviews cannot enter the highlight EDL."""
    frame_paths = sorted((workdir / "frames").glob("f*.jpg"))
    fps = max(1, int(vision.get("fps", 1)))
    samples = []
    for scene_id, (start, end) in enumerate(zip(scenes, scenes[1:])):
        if scene_id % 25 == 0:
            check_cancel(workdir)
        if end - start < config.MIN_CLIP or not frame_paths:
            continue
        times = [(start + end) / 2]
        if end - start >= 4:
            times.append(start + (end - start) * 0.75)
        selected = []
        for t in times:
            index = max(0, min(len(frame_paths) - 1, round(t * fps)))
            frame = frame_paths[index]
            if frame not in selected:
                selected.append(frame)
        samples.append({"scene_id": scene_id, "start": start, "end": end,
                        "frames": selected})

    jlog(workdir, f"classifying {len(samples)} scene shots for race action, interviews and podium")
    batches = [samples[i:i + SCENE_CLASSIFY_BATCH]
               for i in range(0, len(samples), SCENE_CLASSIFY_BATCH)]
    results = {}
    expected_scene_ids = {sample["scene_id"] for sample in samples}

    def classify_one(batch):
        try:
            rows = _classify_scene_batch(workdir, batch)
            expected = {sample["scene_id"] for sample in batch}
            valid_ids = set()
            valid_rows = []
            for row in rows:
                try:
                    scene_id = int(row["scene_id"])
                    confidence = float(row["confidence"])
                    category = str(row["category"])
                    if (scene_id in expected and 0 <= confidence <= 1
                            and category in SCENE_CATEGORIES):
                        valid_ids.add(scene_id)
                        valid_rows.append(row)
                except (KeyError, TypeError, ValueError):
                    continue
            missing = expected - valid_ids
            if not missing:
                return rows
            jlog(workdir, f"batch labels incomplete for IDs "
                 f"{sorted(missing)}; retrying those scenes singly")
            recovered = valid_rows
            for sample in batch:
                if sample["scene_id"] not in missing:
                    continue
                try:
                    recovered.extend(_classify_scene_batch(workdir, [sample]))
                except Exception as single_exc:
                    jlog(workdir, f"single-scene classification failed "
                         f"for ID {sample['scene_id']}: {single_exc}")
            return recovered
        except SceneClassificationFormatError as exc:
            # A smaller request often fixes truncated or malformed batch JSON.
            jlog(workdir, f"batch labels incomplete for IDs "
                 f"{batch[0]['scene_id']}-{batch[-1]['scene_id']}; retrying singly: {exc}")
            rows = []
            for sample in batch:
                try:
                    rows.extend(_classify_scene_batch(workdir, [sample]))
                except Exception as single_exc:
                    jlog(workdir, f"single-scene classification failed "
                         f"for ID {sample['scene_id']}: {single_exc}")
            return rows
        except JobCancelled:
            raise
        except Exception as exc:
            jlog(workdir, f"scene classification failed for IDs "
                 f"{batch[0]['scene_id']}-{batch[-1]['scene_id']}: {exc}; "
                 "unclassified shots will be excluded")
            return []

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=VLM_WORKERS) as pool:
        futures = [pool.submit(classify_one, batch) for batch in batches]
        completed = 0
        for future in as_completed(futures):
            check_cancel(workdir)
            completed += 1
            for item in future.result():
                try:
                    scene_id = int(item["scene_id"])
                    confidence = float(item["confidence"])
                    category = str(item.get("category", "other"))
                    if (0 <= confidence <= 1 and category in SCENE_CATEGORIES
                            and scene_id in expected_scene_ids):
                        results[scene_id] = {
                            "scene_id": scene_id,
                            "category": category,
                            "action": str(item.get("action", "unknown")),
                            "confidence": confidence,
                        }
                except (KeyError, TypeError, ValueError):
                    continue
            if completed % 10 == 0 or completed == len(futures):
                jlog(workdir, f"scene classification batches {completed}/{len(futures)}")

    output = []
    for sample in samples:
        label = results.get(sample["scene_id"], {
            "scene_id": sample["scene_id"], "category": "unknown",
            "action": "unknown", "confidence": 0.0,
        })
        output.append({**sample, **label,
                       "frames": [os.fspath(frame) for frame in sample["frames"]]})
    target = workdir / "race_scenes.json"
    temp = workdir / "race_scenes.json.tmp"
    temp.write_text(json.dumps(output, indent=2, default=_json_fallback))
    temp.replace(target)
    counts = {}
    for item in output:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
    jlog(workdir, f"scene labels: {counts}")
    return output


def run(workdir: Path, vision: dict, duration: float) -> list:
    suspects = vision["suspects"]
    fps = vision["fps"]
    ranges = group_ranges(suspects, fps)

    jlog(workdir, f"VLM review: {len(ranges)} suspect ranges")

    def review_one(i, rg):
        imgs = [Path(f) for f in _sample(rg["frames"]) if Path(f).exists()]
        if not imgs:
            return i, rg, None
        try:
            return i, rg, _call_vlm(workdir, imgs)
        except JobCancelled:
            raise
        except Exception as e:
            jlog(workdir, f"VLM call failed @{rg['start']:.0f}s: {e}")
            # fail-closed: treat review failure as blocked
            return i, rg, {"verdict": "yes",
                           "reason": f"VLM review failed: {e} (fail-closed)",
                           "reason_en": f"VLM review failed: {e}"}

    blocked = []
    reviews = [None] * len(ranges)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    done = 0
    with ThreadPoolExecutor(max_workers=VLM_WORKERS) as ex:
        futs = [ex.submit(review_one, i, rg) for i, rg in enumerate(ranges)]
        for f in as_completed(futs):
            check_cancel(workdir)
            i, rg, verdict = f.result()
            done += 1
            if verdict is None:
                continue
            reviews[i] = {**{k: rg[k] for k in ("start", "end", "flags",
                                               "frames")},
                          "verdict": verdict}
            jlog(workdir, f"range {done}/{len(ranges)} "
                          f"[{rg['start']:.0f}-{rg['end']:.0f}s "
                          f"{','.join(rg['flags'])}] -> {verdict.get('verdict')}")
            if verdict.get("verdict") == "yes":
                blocked.append({"start": rg["start"], "end": rg["end"],
                                "reason": verdict.get(
                                    "reason", verdict.get("reason_en", "")),
                                "flags": rg["flags"]})
    reviews = [r for r in reviews if r]

    # incident tail extension: extend crash blocks 30-60s until scene
    # "returns to normal" — MVP: fixed CLEANUP_TAIL_S window
    extended = []
    for b in blocked:
        if any(f in b["flags"] for f in ("fire", "smoke", "freeze", "red_flag")):
            extended.append({"start": b["start"],
                             "end": min(b["end"] + CLEANUP_TAIL_S, duration),
                             "reason": b["reason"] + " (+cleanup tail)",
                             "flags": b["flags"]})
        else:
            extended.append(b)

    # merge overlaps
    extended.sort(key=lambda x: x["start"])
    merged = []
    for b in extended:
        if merged and b["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], b["end"])
        else:
            merged.append(dict(b))

    (workdir / "vlm_reviews.json").write_text(json.dumps(reviews, indent=2))
    lines = ["# blocked_ranges", ""]
    for b in merged:
        lines.append(f"- [{b['start']:.2f} - {b['end']:.2f}] "
                     f"{','.join(b['flags'])} :: {b['reason']}")
    (workdir / "blocked_ranges.md").write_text("\n".join(lines))
    return merged
