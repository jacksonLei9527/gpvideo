"""Per-job logging + detail line for observability.

jlog() appends a timestamped line to job.log and mirrors the last line to
detail.txt (read by the status endpoint as the 'current activity' text).
"""
import time
from pathlib import Path


def jlog(workdir: Path, msg: str):
    line = f"{time.strftime('%H:%M:%S')} {msg}\n"
    with (workdir / "job.log").open("a") as f:
        f.write(line)
    (workdir / "detail.txt").write_text(msg)
