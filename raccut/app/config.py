import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKDIR = ROOT / "workdir"
OUTPUT = ROOT / "output"
INPUT_DIR = Path.home() / "videos"

ENV_FILE = INPUT_DIR / ".env"


def load_openrouter_key() -> str:
    """Load OpenRouter key from ~/videos/.env.
    Supports both `OPENROUTER_API_KEY=sk-...` and a bare `sk-...` line.
    """
    if "OPENROUTER_API_KEY" in os.environ:
        return os.environ["OPENROUTER_API_KEY"]
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                if k.strip() == "OPENROUTER_API_KEY":
                    return v.strip()
            elif line.startswith("sk-"):
                return line
    raise RuntimeError("OPENROUTER_API_KEY not found in env or ~/videos/.env")


OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
# Vision-capable, cheap, fast. Override via env if needed.
VLM_MODEL = os.environ.get("RACCUT_VLM_MODEL", "google/gemini-2.5-flash")
LLM_MODEL = os.environ.get("RACCUT_LLM_MODEL", "google/gemini-2.5-flash")

# EDL constraints
TARGET_SECONDS = 420          # 7 min
MIN_TOTAL_SECONDS = 360       # 6 min floor
MAX_TOTAL_SECONDS = 540       # 9 min ceiling
MIN_CLIP = 2.0
MAX_CLIP = 15.0
CLIP_WIN = 9.0          # preferred clip length (seconds)
STATIC_MOTION = 1.2     # mean frame-diff below this = mostly static shot
# Mean 1fps frame difference on stage3's 160x90 proxy; lower action shots are
# usually slow/neutralized coverage (often safety-car laps), not highlights.
DEFAULT_MIN_ACTION_MOTION = 18.0


def get_min_action_motion() -> float:
    """Read the current threshold from the process environment or local .env."""
    value = os.environ.get("RACCUT_MIN_ACTION_MOTION")
    if value is None and ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            key, sep, raw = line.partition("=")
            if sep and key.strip() == "RACCUT_MIN_ACTION_MOTION":
                value = raw.strip().strip("\"'")
                break
    try:
        number = float(value) if value is not None else DEFAULT_MIN_ACTION_MOTION
    except (TypeError, ValueError):
        return DEFAULT_MIN_ACTION_MOTION
    return number if 0 <= number <= 255 else DEFAULT_MIN_ACTION_MOTION


def set_min_action_motion(value: float) -> float:
    """Persist the UI setting without disturbing other local .env entries."""
    value = float(value)
    if not 0 <= value <= 255:
        raise ValueError("motion threshold must be between 0 and 255")
    lines = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []
    result = []
    written = False
    for line in lines:
        key, sep, _ = line.partition("=")
        if sep and key.strip() == "RACCUT_MIN_ACTION_MOTION":
            if not written:
                result.append(f"RACCUT_MIN_ACTION_MOTION={value:g}")
                written = True
        else:
            result.append(line)
    if not written:
        result.append(f"RACCUT_MIN_ACTION_MOTION={value:g}")
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    ENV_FILE.write_text("\n".join(result) + "\n")
    ENV_FILE.chmod(0o600)
    os.environ["RACCUT_MIN_ACTION_MOTION"] = f"{value:g}"
    return value


def clear_min_action_motion() -> float:
    """Remove the override and return to the built-in default."""
    lines = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []
    kept = []
    for line in lines:
        key, sep, _ = line.partition("=")
        if not (sep and key.strip() == "RACCUT_MIN_ACTION_MOTION"):
            kept.append(line)
    if ENV_FILE.exists():
        ENV_FILE.write_text("\n".join(kept) + ("\n" if kept else ""))
        ENV_FILE.chmod(0o600)
    os.environ.pop("RACCUT_MIN_ACTION_MOTION", None)
    return DEFAULT_MIN_ACTION_MOTION


MIN_ACTION_MOTION = get_min_action_motion()
PAD_S = 0.25                  # 250ms safety pad at blocked edges

# Detection
FRAME_FPS = 1                 # compliance pass sampling rate
SUSPECT_HI_FPS = 4            # re-scan rate for flagged segments
