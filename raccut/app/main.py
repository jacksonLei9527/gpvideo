import json
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               StreamingResponse)

from . import config
from .pipeline.runner import run_pipeline

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
    return {
        "mem_total_gb": round(vm.total / 1e9, 1),
        "mem_used_pct": vm.percent,
        "proc_rss_mb": round(proc.memory_info().rss / 1e6),
        "cpu_pct": psutil.cpu_percent(interval=0.2),
        "jobs_running": sum(
            1 for d in config.WORKDIR.iterdir()
            if (d / "status.json").exists()
            and json.loads((d / "status.json").read_text()).get("state")
            in ("queued", "running")),
    }


def _start(job_id: str, video: Path):
    jobdir = _job_dir(job_id)
    jobdir.mkdir(parents=True, exist_ok=True)
    (jobdir / "status.json").write_text(json.dumps({"state": "queued", "pct": 0}))
    out = config.OUTPUT / job_id
    out.mkdir(parents=True, exist_ok=True)
    final = out / "final.mp4"
    JOBS[job_id] = jobdir
    t = threading.Thread(target=_run_safe, args=(job_id, video, jobdir, final),
                         daemon=True)
    t.start()


def _run_safe(job_id, video, jobdir, final):
    try:
        run_pipeline(video, jobdir, final)
    except Exception:
        pass  # status already recorded


@app.post("/api/jobs/upload")
def job_from_upload(f: UploadFile = File(...)):
    job_id = uuid.uuid4().hex[:10]
    jobdir = _job_dir(job_id)
    jobdir.mkdir(parents=True, exist_ok=True)
    dst = jobdir / "upload.mp4"
    with dst.open("wb") as w:
        shutil.copyfileobj(f.file, w, length=1 << 22)
    _start(job_id, dst)
    return {"job_id": job_id}


@app.get("/api/jobs")
def list_jobs():
    jobs = []
    for d in sorted(config.WORKDIR.iterdir(), key=lambda d: -d.stat().st_mtime):
        sf = d / "status.json"
        if not sf.exists():
            continue
        st = json.loads(sf.read_text())
        jobs.append({"id": d.name, "state": st.get("state"),
                     "pct": st.get("pct"), "stage": st.get("stage"),
                     "mtime": d.stat().st_mtime,
                     "has_final": (config.OUTPUT / d.name / "final.mp4").exists()})
    return jobs[:20]


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


@app.get("/api/jobs/{job_id}/candidates")
def job_candidates(job_id: str):
    """All safe candidate clips with scores; `selected` = in current EDL."""
    import numpy as np
    from .pipeline import stage5_edl
    d = _job_dir(job_id)
    scenes_f, meta_f = d / "scenes.json", d / "probe.json"
    if not scenes_f.exists() or not meta_f.exists():
        return JSONResponse({"error": "job data incomplete"}, 404)
    scenes = json.loads(scenes_f.read_text())
    duration = json.loads(meta_f.read_text())["duration"]
    blocked = _parse_blocked(d)
    flux = np.load(d / "flux.npy") if (d / "flux.npy").exists() else np.array([])
    motion = np.load(d / "motion.npy") if (d / "motion.npy").exists() else np.array([])
    flux_t = np.arange(len(flux)) * 0.5
    flux_s = flux / (np.median(flux) + 1e-6) if len(flux) else np.array([])
    cands = stage5_edl.compute_candidates(scenes, blocked, duration,
                                          flux_t, flux_s, motion)
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
    meta = json.loads((d / "probe.json").read_text())
    duration = meta["duration"]
    edl = []
    for c in body.get("edl", []):
        s, e = float(c["start"]), float(c["end"])
        if 0 <= s < e <= duration and e - s >= 1.0:
            edl.append({"start": round(s, 3), "end": round(e, 3)})
    edl.sort(key=lambda c: c["start"])
    if not edl:
        return JSONResponse({"error": "empty edl"}, 400)
    (d / "edl.json").write_text(json.dumps(edl, indent=2))

    out = config.OUTPUT / job_id / "final.mp4"
    src = Path(meta["video"])
    blocked = _parse_blocked(d)

    def rerender():
        from .pipeline import stage6_render, stage7_qc
        from .pipeline.joblog import jlog
        try:
            (d / "status.json").write_text(json.dumps({
                "state": "running", "stage": "render", "pct": 85}))
            jlog(d, f"re-render: {len(edl)} clips")
            stage6_render.run(src, edl, d, out)
            (d / "status.json").write_text(json.dumps({
                "state": "running", "stage": "qc", "pct": 95}))
            report = stage7_qc.run(out, edl, blocked, d)
            (d / "status.json").write_text(json.dumps({
                "state": "done", "stage": "done", "pct": 100,
                "final": str(out), "qc": report}))
            jlog(d, "re-render done")
        except Exception as e:
            (d / "status.json").write_text(json.dumps({
                "state": "failed", "error": str(e)}))

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


@app.get("/api/jobs/{job_id}/final.mp4")
def job_final(job_id: str, request: Request):
    f = config.OUTPUT / job_id / "final.mp4"
    if not f.exists():
        return JSONResponse({"error": "not ready"}, 404)
    return range_response(f, request)


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
