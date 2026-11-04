"""Stage 2: streaming spectral flux on 16kHz mono audio -> audio events.

OOM-safe: audio is decoded to wav once (~230MB disk), then read in blocks.
Never loads the whole file into memory.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
WIN_S = 1.0          # analysis window
HOP_S = 0.5
N_FFT = 2048


def decode_audio(video: Path, wav: Path):
    r = subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-i", str(video),
        "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(wav),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"audio decode failed: {r.stderr[-500:]}")


def spectral_flux_series(wav: Path) -> np.ndarray:
    """Per-hop spectral flux (positive diff of magnitude spectra)."""
    win = int(WIN_S * SR)
    hop = int(HOP_S * SR)
    window = np.hanning(win).astype(np.float32)
    flux = []
    prev = None
    with sf.SoundFile(str(wav)) as f:
        # process in 60s blocks to bound memory
        block = 60 * SR
        carry = np.zeros(0, dtype=np.float32)
        while True:
            data = f.read(block, dtype="float32")
            if len(data) == 0:
                break
            buf = np.concatenate([carry, data])
            n = (len(buf) - win) // hop + 1
            for i in range(max(n, 0)):
                seg = buf[i * hop:i * hop + win] * window
                mag = np.abs(np.fft.rfft(seg, N_FFT))
                if prev is not None:
                    flux.append(float(np.sum(np.maximum(mag - prev, 0))))
                prev = mag
            carry = buf[-(win - hop):] if len(buf) >= win - hop else buf.copy()
            del data, buf
    return np.array(flux, dtype=np.float32)


def run(video: Path, workdir: Path) -> dict:
    wav = workdir / "audio_16k.wav"
    decode_audio(video, wav)
    flux = spectral_flux_series(wav)

    # robust peak picking: z-score over rolling baseline
    if len(flux) == 0:
        events = []
    else:
        med = np.median(flux)
        mad = np.median(np.abs(flux - med)) + 1e-6
        z = (flux - med) / (1.4826 * mad)
        events = []
        for i, zi in enumerate(z):
            if zi > 6.0:  # strong transient
                t = i * HOP_S
                if events and t - events[-1]["end"] < 3.0:
                    events[-1]["end"] = t + WIN_S
                    events[-1]["score"] = max(events[-1]["score"], float(zi))
                else:
                    events.append({
                        "start": round(t, 2), "end": round(t + WIN_S, 2),
                        "type": "audio_flux", "score": float(zi),
                    })
        np.save(workdir / "flux.npy", flux)

    out = {"events": events, "hop": HOP_S}
    (workdir / "audio_events.json").write_text(json.dumps(out, indent=2))
    wav.unlink(missing_ok=True)  # free disk
    return out
