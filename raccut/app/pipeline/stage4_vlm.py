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

CLEANUP_TAIL_S = 45.0  # post-incident cleanup window


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _call_vlm(workdir: Path, images: list[Path]) -> dict:
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
            try:
                r = httpx.post(
                    config.OPENROUTER_BASE,
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": config.VLM_MODEL,
                          "messages": [{"role": "user", "content": content}],
                          "temperature": 0},
                    timeout=120,
                )
                r.raise_for_status()
                break
            except Exception as e:
                err = e
                resp = getattr(e, "response", None)
                if resp is not None and resp.status_code == 429:
                    ra = resp.headers.get("retry-after")
                    wait = int(ra) if ra and ra.isdigit() else \
                        min(60, 8 * (attempt + 1))
                    time.sleep(wait)
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
