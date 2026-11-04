#!/usr/bin/env python3
"""生成人造赛车录像用于验证探测器。已知高光区间会打印出来。"""
import subprocess

# (时长, 类型)  0=平静  1=高光
TL = [(60, 0), (15, 1), (75, 0), (18, 1), (132, 0)]

CALM = ["0x2a2a38", "0x32323f"]
HOT = ["red", "blue", "lime", "yellow", "magenta", "cyan", "white", "orange"]

inputs = []          # ffmpeg 输入参数
segs = []            # (label, kind, start, end)
idx = 0
cursor = 0.0

for dur, kind in TL:
    if kind == 0:
        inputs += ["-f", "lavfi", "-i",
                   f"color=c={CALM[idx % 2]}:s=1280x720:d={dur}:r=30"]
        segs.append((f"[{idx}:v]", 0, cursor, cursor + dur))
        idx += 1
    else:
        blocks = int(dur / 0.5)
        labels = []
        for b in range(blocks):
            inputs += ["-f", "lavfi", "-i",
                       f"color=c={HOT[b % len(HOT)]}:s=1280x720:d=0.5:r=30"]
            labels.append(f"[{idx}:v]")
            idx += 1
        segs.append((labels, 1, cursor, cursor + dur))
    cursor += dur

# 视频：每段规整成一个标签，再整体 concat
vfilters = []
vlabels = []
for n, (lab, kind, s, e) in enumerate(segs):
    out = f"[v{n}]"
    if kind == 0:
        vfilters.append(f"{lab}null{out}")
    else:
        vfilters.append(f"{''.join(lab)}concat=n={len(lab)}:v=1:a=0{out}")
    vlabels.append(out)
vfilters.append(f"{''.join(vlabels)}concat=n={len(vlabels)}:v=1:a=0,format=yuv420p[v]")

# 音频
inputs += ["-f", "lavfi", "-i", "sine=frequency=400:sample_rate=48000:duration=300"]
sine_i = idx
idx += 1
inputs += ["-f", "lavfi", "-i", "anoisesrc=d=300:c=pink:r=48000:a=0.02:seed=7"]
noise_i = idx

gates = "+".join(f"between(t\\,{s}\\,{e})" for lab, k, s, e in segs if k == 1) or "0"
afilters = [
    f"[{sine_i}:a]volume=0.05,atrim=0:300,asetpts=N/SR/TB[rum]",
    f"[{noise_i}:a]volume=2.0,atrim=0:300,asetpts=N/SR/TB[nz]",
    f"[nz]volume='{gates}':eval=frame[nzg]",
    "[rum][nzg]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.9[a]",
]

print("预期高光: " + ", ".join(f"{s:.0f}-{e:.0f}s" for _, k, s, e in segs if k == 1))

cmd = (["ffmpeg", "-v", "error", "-stats", "-y"] + inputs +
       ["-filter_complex", ";".join(vfilters + afilters),
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", "race_synth.mp4"])
subprocess.run(cmd, check=True)
print("生成完成: race_synth.mp4")
