"""Stage 1: PySceneDetect shot boundaries -> scenes.json (legal cut points)."""
import json
from pathlib import Path


def run(video: Path, workdir: Path) -> list:
    from scenedetect import detect, ContentDetector, open_video

    v = open_video(str(video))
    scenes = detect(str(video), ContentDetector(threshold=27.0), start_in_scene=True)
    # Fallback: scenedetect API variants
    if not scenes:
        from scenedetect import SceneManager
        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=27.0))
        sm.detect_scenes(v)
        scenes = sm.get_scene_list()

    cuts = [0.0]
    for s, e in scenes:
        cuts.append(s.get_seconds())
    cuts.append(scenes[-1][1].get_seconds() if scenes else v.duration.get_seconds())
    cuts = sorted(set(round(c, 3) for c in cuts))

    (workdir / "scenes.json").write_text(json.dumps(cuts))
    return cuts
