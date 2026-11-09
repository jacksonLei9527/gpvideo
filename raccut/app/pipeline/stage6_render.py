"""Stage 6: deterministic FFmpeg render.

Per-clip re-encode (no -c copy keyframe dependency), 30ms audio fades,
then concat demuxer with stream copy (identical codec params -> safe).
"""
import json
from pathlib import Path
from .cancel import check_cancel, run_command

VBR = "3M"  # 720p-ish bitrate for highlight reel


def _run(cmd, workdir):
    r = run_command(cmd, workdir=workdir, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-600:]}")


def run(video: Path, edl: list, workdir: Path, out_path: Path) -> Path:
    segdir = workdir / "segs"
    segdir.mkdir(exist_ok=True)
    seg_files = []
    for i, c in enumerate(edl):
        check_cancel(workdir)
        dur = c["end"] - c["start"]
        fade_out_start = max(dur - 0.03, 0)
        seg = segdir / f"seg_{i:04d}.mp4"
        _run([
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{c['start']:.3f}", "-t", f"{dur:.3f}", "-i", str(video),
            "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,"
                   "pad=1280:720:(ow-iw)/2:(oh-ih)/2",
            "-af", f"afade=t=in:st=0:d=0.03,"
                   f"afade=t=out:st={fade_out_start:.3f}:d=0.03",
            "-c:v", "libx264", "-preset", "veryfast", "-b:v", VBR,
            "-pix_fmt", "yuv420p", "-r", "25",
            "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
            str(seg),
        ], workdir)
        seg_files.append(seg)

    lst = workdir / "concat.txt"
    lst.write_text("".join(f"file '{s.resolve()}'\n" for s in seg_files))
    check_cancel(workdir)
    _run([
        "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
        "-i", str(lst), "-c", "copy", str(out_path),
    ], workdir)
    return out_path
