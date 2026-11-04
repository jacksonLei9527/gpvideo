"""Job runner: executes stages sequentially, persists status for resume."""
import json
import time
import traceback
from pathlib import Path

from .. import config
from . import (stage0_probe, stage1_scenes, stage2_audio, stage3_vision,
               stage4_vlm, stage5_edl, stage6_render, stage7_qc)
from .joblog import jlog

STAGES = [
    ("probe", "盘点视频"),
    ("scenes", "场景切分"),
    ("audio", "音频高光分析"),
    ("vision", "视觉嫌疑检测"),
    ("vlm", "VLM 二审"),
    ("edl", "EDL 决策"),
    ("render", "渲染成片"),
    ("qc", "成片自检"),
]


class Job:
    def __init__(self, jobdir: Path):
        self.dir = jobdir
        self.status_file = jobdir / "status.json"

    def status(self) -> dict:
        if self.status_file.exists():
            return json.loads(self.status_file.read_text())
        return {}

    def set_status(self, **kw):
        st = self.status()
        st.update(kw)
        self.status_file.write_text(json.dumps(st, indent=2))


def run_pipeline(video: Path, jobdir: Path, out_path: Path):
    job = Job(jobdir)
    (jobdir / "job.log").write_text("")
    stage_times: dict = {}
    jlog(jobdir, f"job start: {video}")

    def stage(name, pct, fn):
        job.set_status(stage=name, pct=pct)
        t = time.time()
        jlog(jobdir, f"-- stage {name} start")
        try:
            r = fn()
        except Exception:
            jlog(jobdir, f"-- stage {name} FAILED after {time.time()-t:.1f}s")
            raise
        stage_times[name] = round(time.time() - t, 1)
        jlog(jobdir, f"-- stage {name} done in {stage_times[name]}s")
        job.set_status(stage_times=stage_times)
        return r

    try:
        job.set_status(state="running", stage="probe", pct=2)
        info = stage("probe", 2, lambda: stage0_probe.run(video, jobdir))
        src = Path(info["video"])
        duration = info["duration"]
        jlog(jobdir, f"duration {duration:.0f}s {info['width']}x{info['height']}")

        scenes = stage("scenes", 10, lambda: stage1_scenes.run(src, jobdir))
        jlog(jobdir, f"{len(scenes)} scene boundaries")

        audio = stage("audio", 25, lambda: stage2_audio.run(src, jobdir))
        jlog(jobdir, f"{len(audio['events'])} audio highlight events")

        vision = stage("vision", 45,
                       lambda: stage3_vision.run(src, jobdir, duration,
                                                 config.FRAME_FPS))
        jlog(jobdir, f"{len(vision['suspects'])} suspect frames")

        blocked = stage("vlm", 65,
                        lambda: stage4_vlm.run(jobdir, vision, duration))
        (jobdir / "blocked.json").write_text(json.dumps(blocked, indent=2))
        jlog(jobdir, f"{len(blocked)} blocked ranges")

        edl = stage("edl", 78,
                    lambda: stage5_edl.run(jobdir, scenes, audio, blocked,
                                           duration))
        jlog(jobdir, f"EDL: {len(edl)} clips, "
                     f"~{sum(c['end']-c['start'] for c in edl):.0f}s output")

        stage("render", 85, lambda: stage6_render.run(src, edl, jobdir,
                                                    out_path))

        report = stage("qc", 95,
                       lambda: stage7_qc.run(out_path, edl, blocked, jobdir))
        jlog(jobdir, f"QC {'PASS' if report['ok'] else 'ISSUES FOUND'}")

        job.set_status(state="done", stage="done", pct=100,
                       final=str(out_path), qc=report,
                       stage_times=stage_times)
    except Exception as e:
        jlog(jobdir, f"job failed: {e}")
        job.set_status(state="failed", error=str(e),
                       trace=traceback.format_exc()[-2000:],
                       stage_times=stage_times)
        raise
