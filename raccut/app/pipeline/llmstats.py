"""LLM/VLM call accounting: llm_stats.json per job."""
import json
import time
from pathlib import Path

F = "llm_stats.json"


def record(workdir: Path, kind: str, n_images: int, latency: float, ok: bool):
    f = workdir / F
    st = json.loads(f.read_text()) if f.exists() else {
        "calls": 0, "errors": 0, "images": 0, "total_s": 0.0, "by_kind": {}}
    st["calls"] += 1
    st["images"] += n_images
    st["total_s"] += latency
    if not ok:
        st["errors"] += 1
    k = st["by_kind"].setdefault(kind, {"calls": 0, "total_s": 0.0})
    k["calls"] += 1
    k["total_s"] += latency
    st["avg_s"] = round(st["total_s"] / max(st["calls"], 1), 2)
    f.write_text(json.dumps(st, indent=2))


class timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        pass

    @property
    def latency(self):
        return time.time() - self.t
