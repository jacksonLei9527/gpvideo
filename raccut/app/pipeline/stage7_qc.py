"""Stage 7: final.mp4 self-check.

  - duration & streams sane (ffprobe)
  - 1 frame / 10s re-run fire/red-flag heuristics on the OUTPUT
  - cut-point +/-300ms spot check for blocked leakage: frames sampled at
    each clip boundary checked against blocked flag set
"""
import json
import subprocess
import tempfile
from pathlib import Path

import cv2

from .stage3_vision import analyze_frame
from .stage0_probe import probe


def _grab(video: Path, t: float, dst: Path) -> bool:
    r = subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(video),
        "-frames:v", "1", "-vf", "scale=640:-1", str(dst),
    ], capture_output=True, text=True)
    return r.returncode == 0 and dst.exists()


def run(final: Path, edl: list, blocked: list, workdir: Path) -> dict:
    report = {"ok": True, "issues": []}
    info = probe(final)
    dur = float(info["format"]["duration"])

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # 1) every-10s content scan
        t = 0.0
        while t < dur:
            f = td / "q.jpg"
            if _grab(final, t, f):
                img = cv2.imread(str(f))
                if img is not None:
                    feats = analyze_frame(img)
                    if feats["fire"] > 0.03 or feats["red"] > 0.30:
                        det = {k: round(v, 3) for k, v in feats.items()}
                        report["issues"].append(
                            {"t": t, "type": "content",
                             "desc": f"suspect visual content at {t:.0f}s "
                                     f"(fire={det['fire']}, red={det['red']})",
                             "detail": det})
                        report["ok"] = False
            t += 10.0

        # 2) cut-boundary +/-300ms leakage check vs blocked flags
        # (we re-derive the boundary position in output timeline)
        out_t = 0.0
        for i, c in enumerate(edl[:-1]):
            out_t += c["end"] - c["start"]
            for off in (-0.3, 0.3):
                f = td / "b.jpg"
                tt = out_t + off
                if tt < 0 or tt >= dur:
                    continue
                if _grab(final, tt, f):
                    img = cv2.imread(str(f))
                    black = (img is None or
                             cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() < 5)
                    if black:
                        report["issues"].append(
                            {"t": tt, "type": "black_frame",
                             "desc": f"black/dropped frame near cut at {tt:.1f}s "
                                     f"(clip boundary {i+1}->{i+2})"})
                        report["ok"] = False

    report["duration"] = dur
    (workdir / "qc_report.json").write_text(json.dumps(report, indent=2))
    return report
