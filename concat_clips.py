#!/usr/bin/env python3
"""
把多段片段合成一条完整视频（自动统一分辨率/帧率/编码）

用法：
  # 1) 从 extract_clips.py 的清单合成（推荐）
  python3 concat_clips.py clips/clips_manifest.json -o highlight.mp4

  # 2) 直接给一串文件
  python3 concat_clips.py a.mp4 b.mp4 c.mp4 -o highlight.mp4

说明：
  不同来源的片段参数常不一致（分辨率/帧率/音轨声道数），直接 concat 会花屏或失步。
  本脚本用 filter_complex 统一 scale+pad+fps+音频采样率后再拼，稳但需重编码。
  如果所有片段本来就同源（同一台机器同一设置剪出来的），加 --copy 用流复制，
  速度快几十倍且零损失。
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def probe_duration(path):
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True)
    try:
        return float(p.stdout.strip())
    except ValueError:
        return 0.0


def probe_has_audio(path):
    """该文件是否含音轨。缺失的片段需补静音，否则拼接必然失败。"""
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    return bool(p.stdout.strip())


def collect_files(args):
    files = []
    if args.manifest:
        mp = Path(args.manifest).expanduser().resolve()
        if not mp.exists():
            sys.exit(f"找不到清单: {mp}")
        data = json.loads(mp.read_text())
        for c in data.get("clips", []):
            f = Path(c["file"])
            files.append(f if f.is_absolute() else mp.parent / f.name)
    files.extend(Path(f).expanduser().resolve() for f in args.inputs)
    return [f for f in files if f.exists()]


def main():
    ap = argparse.ArgumentParser(description="合成多段片段为一条视频")
    ap.add_argument("inputs", nargs="*", help="片段文件")
    ap.add_argument("--manifest", help="extract_clips.py 生成的 clips_manifest.json")
    ap.add_argument("-o", "--output", default="highlight.mp4", help="输出文件")
    ap.add_argument("--copy", action="store_true",
                    help="流复制拼接（仅当所有片段参数完全一致时可用，极快）")
    ap.add_argument("--width", type=int, default=1920, help="统一宽度（默认 1920）")
    ap.add_argument("--height", type=int, default=1080, help="统一高度（默认 1080）")
    ap.add_argument("--fps", type=float, default=30, help="统一帧率（默认 30）")
    ap.add_argument("--crf", type=int, default=20, help="质量，越小越好（默认 20）")
    ap.add_argument("--preset", default="medium", help="x264 preset")
    ap.add_argument("--audio", choices=["auto", "drop"], default="auto",
                    help="auto=给缺音轨的片段补静音(默认) drop=丢弃全部音轨")
    args = ap.parse_args()

    files = collect_files(args)
    if not files:
        sys.exit("✗ 没有找到任何输入片段")
    if len(files) == 1:
        print("! 只有一段，直接转码输出")

    out = Path(args.output).expanduser()
    if not out.is_absolute():
        # 相对路径基于当前工作目录，而不是片段所在目录 ——
        # 否则 -o out/highlight.mp4 会被拼成 out/clips/out/highlight.mp4
        out = Path.cwd() / out
    out.parent.mkdir(parents=True, exist_ok=True)

    src_total = sum(probe_duration(f) for f in files)
    print(f"▶ 合成 {len(files)} 段，源合计 {src_total:.1f}s ({src_total / 60:.2f} 分钟)")
    for f in files:
        print(f"  · {f.name}  ({probe_duration(f):.1f}s)")

    if args.copy:
        # 流复制：写 concat 列表再拼
        lst = out.parent / "_concat_list.txt"
        lst.write_text("".join(f"file '{f.as_posix()}'\n" for f in files))
        cmd = ["ffmpeg", "-v", "error", "-stats", "-f", "concat", "-safe", "0",
               "-i", str(lst), "-c", "copy", "-y", str(out)]
        print("\n▶ 流复制拼接...")
        p = subprocess.run(cmd, capture_output=True)
        if p.returncode != 0:
            sys.exit(f"✗ 失败（片段参数可能不一致，去掉 --copy 重试）:\n"
                     f"{p.stderr.decode(errors='replace')[-1200:]}")
    else:
        # 先探测每个片段的音轨情况：缺音轨的片段必须补静音，
        # 否则 filtergraph 里的 [i:a] 会 "matches no streams" 直接失败
        has_audio = [probe_has_audio(f) for f in files]
        n = len(files)
        missing = [i for i, h in enumerate(has_audio) if not h]

        if missing and args.audio == "drop":
            print(f"\n! {len(missing)} 个片段无音轨，且指定 --audio drop → 输出为无声视频")
            use_audio = False
        elif missing:
            print(f"\n! {len(missing)} 个片段无音轨 → 自动补等长静音（保持音画同步）")
            print(f"  （若想改为丢弃全部音轨，用 --audio drop）")
            use_audio = True
        else:
            use_audio = True

        cmd = ["ffmpeg", "-v", "error", "-stats"]
        for f in files:
            cmd += ["-i", str(f)]
        # 为缺音轨的片段追加静音输入源
        silence_idx = {}
        if use_audio:
            for i in missing:
                # 每个静音源覆盖该片段时长；anullsrc 无限长，用 -t 限长
                silence_idx[i] = n + len(silence_idx)
                cmd += ["-f", "lavfi", "-t", f"{max(probe_duration(files[i]), 0.1):.3f}",
                        "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]

        parts = []
        for i in range(n):
            parts.append(
                f"[{i}:v]scale={args.width}:{args.height}:force_original_aspect_ratio=decrease,"
                f"pad={args.width}:{args.height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"setsar=1,fps={args.fps},format=yuv420p[v{i}]")
        if use_audio:
            for i in range(n):
                src = i if has_audio[i] else silence_idx[i]
                parts.append(f"[{src}:a]aresample=48000,aformat=sample_fmts=fltp:"
                             f"channel_layouts=stereo[a{i}]")
            stream_in = "".join(f"[v{i}][a{i}]" for i in range(n))
            parts.append(f"{stream_in}concat=n={n}:v=1:a=1[vout][aout]")
            maps = ["-map", "[vout]", "-map", "[aout]"]
        else:
            stream_in = "".join(f"[v{i}]" for i in range(n))
            parts.append(f"{stream_in}concat=n={n}:v=1:a=0[vout]")
            maps = ["-map", "[vout]"]

        cmd += ["-filter_complex", ";".join(parts)] + maps + [
            "-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf),
        ]
        if use_audio:
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        cmd += ["-movflags", "+faststart", "-y", str(out)]
        print("\n▶ 统一规格后拼接（重编码）...")
        p = subprocess.run(cmd, capture_output=True)
        if p.returncode != 0:
            err = p.stderr.decode(errors="replace")
            sys.exit(f"✗ 拼接失败:\n{err[-1500:]}")

    if not out.exists():
        sys.exit("✗ 输出文件未生成")
    dur = probe_duration(out)
    size_mb = out.stat().st_size / 1048576
    print(f"\n✓ 完成: {out}")
    print(f"  时长 {dur:.1f}s ({dur / 60:.2f} 分钟)  大小 {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
