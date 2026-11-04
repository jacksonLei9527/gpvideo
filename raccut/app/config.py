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
TARGET_SECONDS = 480          # 8 min
MIN_TOTAL_SECONDS = 360       # 6 min floor
MAX_TOTAL_SECONDS = 540       # 9 min ceiling
MIN_CLIP = 2.0
MAX_CLIP = 15.0
CLIP_WIN = 9.0          # preferred clip length (seconds)
STATIC_MOTION = 1.2     # mean frame-diff below this = mostly static shot
PAD_S = 0.25                  # 250ms safety pad at blocked edges

# Detection
FRAME_FPS = 1                 # compliance pass sampling rate
SUSPECT_HI_FPS = 4            # re-scan rate for flagged segments
