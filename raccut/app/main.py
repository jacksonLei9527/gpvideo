import json
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               StreamingResponse)

from . import config
from .pipeline.runner import run_pipeline
from .pipeline.cancel import (is_cancelled, register_job, request_cancel,
                              unregister_job)

app = FastAPI(title="raccut")
JOBS: dict[str, Path] = {}
lock = threading.Lock()


def _job_dir(job_id: str) -> Path:
    return config.WORKDIR / job_id


def range_response(path: Path, request: Request,
                   media_type: str = "video/mp4"):
    """Minimal HTTP Range support so <video> seeking works everywhere."""
    size = path.stat().st_size
    rng = request.headers.get("range")
    if rng:
        import re
        m = re.match(r"bytes=(\d*)-(\d*)", rng)
        start = int(m.group(1) or 0)
        end = min(int(m.group(2) or size - 1), size - 1)
        length = end - start + 1

        def gen():
            with open(path, "rb") as f:
                f.seek(start)
                left = length
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    left -= len(chunk)
                    yield chunk

        return StreamingResponse(
            gen(), status_code=206, media_type=media_type, headers={
                "Content-Range": f"bytes {start}-{end}/{size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length)})
    return FileResponse(path, media_type=media_type,
                        headers={"Accept-Ranges": "bytes"})


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(
        (config.ROOT / "web" / "index.html").read_text(),
        headers={"Cache-Control": "no-store"})


@app.get("/api/sys")
def sys_info():
    import psutil
    vm = psutil.virtual_memory()
    proc = psutil.Process()
    running = 0
    for d in config.WORKDIR.iterdir():
        sf = d / "status.json"
        if not sf.exists():
            continue
        try:
            state = json.loads(sf.read_text()).get("state")
        except (OSError, json.JSONDecodeError, AttributeError):
            continue
        running += state in ("queued", "running")
    return {
        "mem_total_gb": round(vm.total / 1e9, 1),
        "mem_used_pct": vm.percent,
        "proc_rss_mb": round(proc.memory_info().rss / 1e6),
        "cpu_pct": psutil.cpu_percent(interval=0.2),
        "jobs_running": running,
    }


@app.get("/api/settings")
def get_settings():
    return {"min_action_motion": config.get_min_action_motion(),
            "default_min_action_motion": config.DEFAULT_MIN_ACTION_MOTION}


@app.post("/api/settings")
def save_settings(body: dict):
    try:
        if body.get("reset"):
            saved = config.clear_min_action_motion()
        else:
            value = float(body.get("min_action_motion"))
            saved = config.set_min_action_motion(value)
    except (TypeError, ValueError) as exc:
        return JSONResponse({"error": str(exc) or "invalid motion threshold"}, 400)
    return {"ok": True, "min_action_motion": saved}


def _start(job_id: str, video: Path, started_at: float | None = None,
           source_filename: str | None = None):
    jobdir = _job_dir(job_id)
    jobdir.mkdir(parents=True, exist_ok=True)
    (jobdir / "status.json").write_text(json.dumps({
        "state": "queued", "pct": 0, "started_at": started_at or time.time(),
        "source_filename": source_filename or video.name,
        "min_action_motion": config.get_min_action_motion(),
    }))
    out = config.OUTPUT / job_id
    out.mkdir(parents=True, exist_ok=True)
    final = out / "final.mp4"
    JOBS[job_id] = jobdir
    register_job(job_id)
    t = threading.Thread(target=_run_safe, args=(job_id, video, jobdir, final),
                         daemon=True)
    t.start()


def _run_safe(job_id, video, jobdir, final):
    try:
        run_pipeline(video, jobdir, final)
    except Exception:
        pass  # status already recorded


@app.post("/api/jobs/upload")
def job_from_upload(f: UploadFile = File(...), started_at: float | None = Form(None)):
    filename = (f.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    if not filename.lower().endswith(".mp4"):
        return JSONResponse({"error": "Please upload an MP4 video."}, 400)
    job_id = uuid.uuid4().hex[:10]
    jobdir = _job_dir(job_id)
    jobdir.mkdir(parents=True, exist_ok=True)
    dst = jobdir / "upload.mp4"
    with dst.open("wb") as w:
        shutil.copyfileobj(f.file, w, length=1 << 22)
    if dst.stat().st_size == 0:
        shutil.rmtree(jobdir, ignore_errors=True)
        return JSONResponse({"error": "The uploaded video is empty."}, 400)
    _start(job_id, dst, started_at, filename)
    return {"job_id": job_id}


@app.get("/api/jobs")
def list_jobs(q: str = "", limit: int = 20):
    jobs = []
    query = q.strip().casefold()
    limit = min(max(limit, 1), 100)
    for d in sorted(config.WORKDIR.iterdir(), key=lambda d: -d.stat().st_mtime):
        sf = d / "status.json"
        if not sf.exists():
            continue
        try:
            st = json.loads(sf.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(st, dict):
            continue
        source_name = str(st.get("source_filename") or "")
        if query and query not in source_name.casefold() and query not in d.name.casefold():
            continue
        now = time.time()
        started = st.get("started_at") or now
        finished = st.get("finished_at")
        elapsed = st.get("elapsed_seconds")
        if elapsed is None:
            elapsed = max(0, (finished or now) - started)
        detail_file = d / "detail.txt"
        jobs.append({"id": d.name, "source_filename": st.get("source_filename"),
                     "state": st.get("state"), "pct": st.get("pct"),
                     "stage": st.get("stage"),
                     "started_at": st.get("started_at"),
                     "finished_at": st.get("finished_at"),
                     "elapsed_seconds": elapsed,
                     "cancel_requested": is_cancelled(d),
                     "detail": detail_file.read_text().strip()
                     if detail_file.exists() else "",
                     "mtime": d.stat().st_mtime,
                     "has_final": (config.OUTPUT / d.name / "final.mp4").exists()})
    return jobs[:limit]


def _parse_blocked(d: Path) -> list:
    """blocked.json (new jobs) or parse blocked_ranges.md (old jobs)."""
    bf = d / "blocked.json"
    if bf.exists():
        return json.loads(bf.read_text())
    mf = d / "blocked_ranges.md"
    out = []
    if mf.exists():
        import re
        for line in mf.read_text().splitlines():
            m = re.match(r"- \[([\d.]+) - ([\d.]+)\]", line)
            if m:
                out.append({"start": float(m.group(1)),
                            "end": float(m.group(2))})
    return out


def _job_safe_candidates(d: Path):
    import numpy as np
    from .pipeline import stage5_edl
    scenes_f, meta_f = d / "scenes.json", d / "probe.json"
    if not scenes_f.exists() or not meta_f.exists():
        raise FileNotFoundError("job data incomplete")
    scenes = json.loads(scenes_f.read_text())
    duration = json.loads(meta_f.read_text())["duration"]
    blocked = _parse_blocked(d)
    flux = np.load(d / "flux.npy") if (d / "flux.npy").exists() else np.array([])
    motion = np.load(d / "motion.npy") if (d / "motion.npy").exists() else np.array([])
    flux_t = np.arange(len(flux)) * 0.5
    flux_s = flux / (np.median(flux) + 1e-6) if len(flux) else np.array([])
    labels_f = d / "race_scenes.json"
    labels = ({int(item["scene_id"]): item
               for item in json.loads(labels_f.read_text())}
              if labels_f.exists() else {})
    status_file = d / "status.json"
    saved_floor = (json.loads(status_file.read_text()).get("min_action_motion")
                   if status_file.exists() else None)
    cands = stage5_edl.compute_candidates(
        scenes, blocked, duration, flux_t, flux_s, motion, labels, d,
        min_action_motion=saved_floor)
    return cands, duration


@app.get("/api/jobs/{job_id}/candidates")
def job_candidates(job_id: str):
    """All safe candidate clips with scores; `selected` = in current EDL."""
    d = _job_dir(job_id)
    try:
        cands, _ = _job_safe_candidates(d)
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, 404)
    edl_f = d / "edl.json"
    edl = json.loads(edl_f.read_text()) if edl_f.exists() else []

    def in_edl(c):
        """selected if the candidate overlaps any EDL clip by >50% of itself."""
        for e in edl:
            inter = min(c["end"], e["end"]) - max(c["start"], e["start"])
            if inter > 0.5 * (c["end"] - c["start"]):
                return True
        return False

    for c in cands:
        c["selected"] = in_edl(c)
        c["start"] = round(float(c["start"]), 3)
        c["end"] = round(float(c["end"]), 3)
        c["score"] = round(float(c["score"]), 1)
        if "motion" in c:
            c["motion"] = round(float(c["motion"]), 2)
    cands.sort(key=lambda c: -c["score"])
    return cands


@app.post("/api/jobs/{job_id}/edl")
def job_set_edl(job_id: str, body: dict):
    """Save a custom EDL and re-render (render + qc stages only)."""
    d = _job_dir(job_id)
    sf = d / "status.json"
    if not sf.exists():
        return JSONResponse({"error": "no such job"}, 404)
    st = json.loads(sf.read_text())
    if st.get("state") in ("running", "queued"):
        return JSONResponse({"error": "job still running"}, 409)
    if st.get("state") != "done":
        return JSONResponse({"error": "only completed jobs can be re-rendered"}, 409)
    try:
        safe_candidates, duration = _job_safe_candidates(d)
    except FileNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, 404)
    edl = []
    categories = []
    for c in body.get("edl", []):
        s, e = float(c["start"]), float(c["end"])
        if not (0 <= s < e <= duration and e-s >= 1.0):
            continue
        matches = [candidate for candidate in safe_candidates
                   if candidate["start"] <= s + 0.01
                   and candidate["end"] >= e - 0.01]
        if not matches:
            return JSONResponse({"error": "clip must stay within a reviewed safe candidate"}, 400)
        category = min(matches, key=lambda candidate: candidate["end"]-candidate["start"])["category"]
        edl.append({"start": round(s, 3), "end": round(e, 3)})
        categories.append(category)
    ordered = sorted(zip(edl, categories), key=lambda item: item[0]["start"])
    edl = [item[0] for item in ordered]
    categories = [item[1] for item in ordered]
    if not edl:
        return JSONResponse({"error": "empty edl"}, 400)
    if any(left["end"] > right["start"] + 0.001
           for left, right in zip(edl, edl[1:])):
        return JSONResponse({"error": "clip ranges cannot overlap"}, 400)
    podium_seconds = sum(c["end"]-c["start"] for c, category in zip(edl, categories)
                        if category == "podium")
    if podium_seconds > 8.001:
        return JSONResponse({"error": "podium footage is limited to 8 seconds"}, 400)
    seen_podium = False
    for category in categories:
        if category == "podium":
            seen_podium = True
        elif seen_podium:
            return JSONResponse({"error": "podium footage must remain at the end"}, 400)
    (d / "edl.json").write_text(json.dumps(edl, indent=2))

    out = config.OUTPUT / job_id / "final.mp4"
    src = Path(json.loads((d / "probe.json").read_text())["video"])
    blocked = _parse_blocked(d)
    started_at = time.time()
    source_filename = st.get("source_filename")
    motion_floor = st.get("min_action_motion")

    def rerender():
        from .pipeline import stage6_render, stage7_qc
        from .pipeline.joblog import jlog
        from .pipeline.cancel import JobCancelled, check_cancel
        try:
            (d / "status.json").write_text(json.dumps({
                "state": "running", "stage": "render", "pct": 85,
                "started_at": started_at, "source_filename": source_filename,
                "min_action_motion": motion_floor}))
            jlog(d, f"re-render: {len(edl)} clips")
            stage6_render.run(src, edl, d, out)
            check_cancel(d)
            (d / "status.json").write_text(json.dumps({
                "state": "running", "stage": "qc", "pct": 95,
                "started_at": started_at, "source_filename": source_filename,
                "min_action_motion": motion_floor}))
            report = stage7_qc.run(out, edl, blocked, d)
            finished_at = time.time()
            (d / "status.json").write_text(json.dumps({
                "state": "done", "stage": "done", "pct": 100,
                "final": str(out), "qc": report, "started_at": started_at,
                "source_filename": source_filename,
                "min_action_motion": motion_floor, "finished_at": finished_at,
                "elapsed_seconds": finished_at-started_at}))
            jlog(d, "re-render done")
        except JobCancelled:
            out.unlink(missing_ok=True)
            finished_at = time.time()
            (d / "status.json").write_text(json.dumps({
                "state": "cancelled", "stage": "cancelled",
                "started_at": started_at, "source_filename": source_filename,
                "min_action_motion": motion_floor, "finished_at": finished_at,
                "elapsed_seconds": finished_at-started_at}))
            jlog(d, "re-render cancelled by user")
        except Exception as e:
            finished_at = time.time()
            (d / "status.json").write_text(json.dumps({
                "state": "failed", "error": str(e), "started_at": started_at,
                "source_filename": source_filename,
                "min_action_motion": motion_floor, "finished_at": finished_at,
                "elapsed_seconds": finished_at-started_at}))
        finally:
            unregister_job(job_id)

    register_job(job_id)
    threading.Thread(target=rerender, daemon=True).start()
    return {"ok": True, "clips": len(edl)}


@app.delete("/api/jobs/{job_id}")
def job_delete(job_id: str):
    d = _job_dir(job_id)
    if not d.exists():
        return JSONResponse({"error": "no such job"}, 404)
    sf = d / "status.json"
    if sf.exists():
        st = json.loads(sf.read_text())
        if st.get("state") in ("running", "queued"):
            return JSONResponse({"error": "job still running"}, 409)
    shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(config.OUTPUT / job_id, ignore_errors=True)
    return {"ok": True}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    d = _job_dir(job_id)
    sf = d / "status.json"
    if not sf.exists():
        return JSONResponse({"error": "no such job"}, 404)
    st = json.loads(sf.read_text())
    st["cancel_requested"] = is_cancelled(d)
    if st.get("state") not in ("queued", "running"):
        now = time.time()
        st.setdefault("elapsed_seconds", max(
            0, (st.get("finished_at") or now) - (st.get("started_at") or now)))
    df = d / "detail.txt"
    if df.exists():
        st["detail"] = df.read_text().strip()
    lf = d / "job.log"
    if lf.exists():
        st["log_tail"] = lf.read_text().splitlines()[-60:]
    llf = d / "llm_stats.json"
    if llf.exists():
        st["llm"] = json.loads(llf.read_text())
    return st


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    d = _job_dir(job_id)
    sf = d / "status.json"
    if not sf.exists():
        return JSONResponse({"error": "no such job"}, 404)
    st = json.loads(sf.read_text())
    if st.get("state") not in ("queued", "running"):
        return JSONResponse({"error": "job is not running"}, 409)
    if not request_cancel(job_id):
        # Recheck terminal state to avoid racing a worker that just finished.
        latest = json.loads(sf.read_text())
        if latest.get("state") not in ("queued", "running"):
            return JSONResponse({"error": "job is not running"}, 409)
        # A job left marked active by a server restart has no worker to signal.
        # Let the user close that stale entry instead of exposing a dead button.
        finished = time.time()
        st.update(state="cancelled", stage="cancelled", finished_at=finished,
                  elapsed_seconds=max(0, finished-(st.get("started_at") or finished)))
        sf.write_text(json.dumps(st, indent=2))
    return {"ok": True}


@app.get("/api/jobs/{job_id}/final.mp4")
def job_final(job_id: str, request: Request):
    f = config.OUTPUT / job_id / "final.mp4"
    if not f.exists():
        return JSONResponse({"error": "not ready"}, 404)
    return range_response(f, request)


@app.get("/api/jobs/{job_id}/download")
def job_download(job_id: str):
    """Download using the source name and the job's elapsed generation time."""
    d = _job_dir(job_id)
    f = config.OUTPUT / job_id / "final.mp4"
    if not f.exists():
        return JSONResponse({"error": "not ready"}, 404)
    status_file = d / "status.json"
    status = json.loads(status_file.read_text()) if status_file.exists() else {}
    source = str(status.get("source_filename") or job_id)
    source = source.replace("\\", "/").rsplit("/", 1)[-1]
    stem = Path(source).stem or job_id
    # Keep only portable filename characters, preventing header/path injection.
    import re
    stem = re.sub(r'[\x00-\x1f<>:"/\\|?*]+', "_", stem).strip(" ._") or job_id
    elapsed = status.get("elapsed_seconds")
    if elapsed is None and status.get("started_at") and status.get("finished_at"):
        elapsed = status["finished_at"] - status["started_at"]
    # Older task records may not contain timer fields; retain a useful suffix.
    if elapsed is None:
        elapsed = (status.get("qc") or {}).get("duration", 0)
    total_seconds = max(0, int(round(float(elapsed))))
    duration_tag = f"{total_seconds // 60}.{total_seconds % 60:02d}"
    return FileResponse(f, media_type="video/mp4",
                        filename=f"{stem}_output_{duration_tag}.mp4")


@app.get("/api/jobs/{job_id}/source.mp4")
def job_source(job_id: str, request: Request):
    meta_f = _job_dir(job_id) / "probe.json"
    if not meta_f.exists():
        return JSONResponse({"error": "no meta"}, 404)
    src = Path(json.loads(meta_f.read_text())["video"])
    if not src.exists():
        return JSONResponse({"error": "source gone"}, 404)
    return range_response(src, request)


@app.get("/api/jobs/{job_id}/segs/{i}.mp4")
def job_seg(job_id: str, i: int, request: Request):
    f = _job_dir(job_id) / "segs" / f"seg_{i:04d}.mp4"
    if not f.exists():
        return JSONResponse({"error": "no such segment"}, 404)
    return range_response(f, request)


@app.get("/api/jobs/{job_id}/edl")
def job_edl(job_id: str):
    f = _job_dir(job_id) / "edl.json"
    return json.loads(f.read_text()) if f.exists() else []


@app.get("/api/jobs/{job_id}/meta")
def job_meta(job_id: str):
    f = _job_dir(job_id) / "probe.json"
    return json.loads(f.read_text()) if f.exists() else {}


@app.get("/api/jobs/{job_id}/frames/{name}")
def job_frame(job_id: str, name: str):
    f = _job_dir(job_id) / "frames" / Path(name).name
    if not f.exists():
        return JSONResponse({"error": "not found"}, 404)
    return FileResponse(f, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/reviews")
def job_reviews(job_id: str):
    f = _job_dir(job_id) / "vlm_reviews.json"
    return json.loads(f.read_text()) if f.exists() else []


@app.get("/api/jobs/{job_id}/events")
def job_events(job_id: str):
    f = _job_dir(job_id) / "audio_events.json"
    return json.loads(f.read_text()) if f.exists() else {"events": []}


@app.get("/api/jobs/{job_id}/blocked")
def job_blocked(job_id: str):
    f = _job_dir(job_id) / "blocked_ranges.md"
    return {"text": f.read_text()} if f.exists() else {"text": ""}
