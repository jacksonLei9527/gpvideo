"""Stage 0: ffprobe inventory + conditional remux to fix timestamps."""
import json
import shutil
import subprocess
from pathlib import Path


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def probe(path: Path) -> dict:
    r = _run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ])
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr[:500]}")
    return json.loads(r.stdout)


def needs_remux(info: dict) -> bool:
    """Heuristic: remux if start_time is badly offset or duration missing."""
    try:
        fmt = info["format"]
        start = float(fmt.get("start_time", 0))
        duration = float(fmt.get("duration", 0))
        if duration <= 0:
            return True
        if abs(start) > 1.0:  # big timestamp offset
            return True
        v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
        if v is None:
            return True
        return False
    except (KeyError, ValueError):
        return True


def remux(src: Path, dst: Path) -> Path:
    r = _run([
        "ffmpeg", "-y", "-err_detect", "ignore_err", "-i", str(src),
        "-c", "copy", "-fflags", "+genpts", str(dst),
    ])
    if r.returncode != 0 or not dst.exists():
        raise RuntimeError(f"remux failed: {r.stderr[-500:]}")
    return dst


def run(video: Path, workdir: Path) -> dict:
    info = probe(video)
    fixed = video
    remuxed = False
    if needs_remux(info):
        fixed = workdir / "fixed.mp4"
        remux(video, fixed)
        info = probe(fixed)
        remuxed = True
    fmt = info["format"]
    vs = next(s for s in info["streams"] if s["codec_type"] == "video")
    out = {
        "video": str(fixed),
        "remuxed": remuxed,
        "duration": float(fmt["duration"]),
        "width": vs["width"],
        "height": vs["height"],
        "fps": eval(vs.get("r_frame_rate", "25/1")),
    }
    (workdir / "probe.json").write_text(json.dumps(out, indent=2))
    return out
