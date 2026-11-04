#!/usr/bin/env python3
"""
按时间码批量剪出片段

用法：
  # 1) 直接命令行给时间码
  python3 extract_clips.py race.mp4 -c 00:12:30-00:13:05 -c 01:02:10-01:02:48

  # 2) 从 race_detect.py 生成的 JSON 里按序号挑
  python3 extract_clips.py race.mp4 --from-json hl.json --pick 1,3,5-7

模式（--mode，默认 smart）：
  smart    先探测源文件关键帧间隔。切点能落在关键帧附近就流复制（极快、零损失），
           否则自动改用重编码保证精确。剪完还会校验实际时长，超差就自动重剪。
  copy     强制流复制。最快、画质零损失，但切割点被吸附到关键帧，
           实际时长可能比请求的长几秒（实测某片请求 25s 得到 31.1s）。
  precise  强制重编码。帧级精确，卡超车瞬间用这个。慢一些但可控。

为什么默认 smart：
  -c copy 的切点只能落在关键帧上。若源文件关键帧间隔大（如 GPU 录制 10s 一个），
  请求剪 12s 可能得到 20s，会错过真正的高光瞬间。smart 会自动避开这个坑。
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def parse_tc(t):
    """支持 HH:MM:SS.mmm / MM:SS / 纯秒数。"""
    t = t.strip()
    if ":" not in t:
        return float(t)
    parts = t.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    raise ValueError(f"无法解析时间码: {t}")


def fmt_tc(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}{m:02d}{s:06.3f}".replace(".", "_")


def parse_spec(spec):
    """'START-END' -> (float, float)"""
    if "-" not in spec[1:]:
        raise ValueError(f"格式应为 START-END，收到: {spec}")
    # 从右侧找分隔符，避免时间码里的负号歧义
    idx = spec.rindex("-")
    a, b = spec[:idx], spec[idx + 1:]
    return parse_tc(a), parse_tc(b)


def expand_picks(s):
    """'1,3,5-7' -> [1,3,5,6,7]"""
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def keyframe_spacing(path, sample_sec=300):
    """
    估算源文件关键帧平均间隔（秒）。
    用 pkt_pos/pict_type 列表取关键帧时间戳，算相邻差值的中位数。
    只采样前 sample_sec 秒，避免长录像扫描过慢。
    """
    p = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "packet=pts_time,flags",
        "-of", "csv=p=0", "-read_intervals", f"%+{int(sample_sec)}",
        str(path),
    ], capture_output=True, text=True)
    if p.returncode != 0:
        return None
    times = []
    for line in p.stdout.splitlines():
        parts = line.strip().split(",")
        # csv 输出形如 "0.000000,K__" —— flags 里含 K 表示关键帧
        if len(parts) < 2:
            continue
        try:
            t = float(parts[0])
        except ValueError:
            continue
        if "K" in parts[1]:
            times.append(t)
    if len(times) < 3:
        return None
    gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
    if not gaps:
        return None
    return statistics.median(gaps)


def probe_duration(path):
    p = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", str(path),
    ], capture_output=True, text=True)
    try:
        return float(p.stdout.strip())
    except ValueError:
        return 0.0


def check_vfr(path, sample_sec=60):
    """
    检测可变帧率(VFR)。OBS 默认录制常产出 VFR，而采集卡多为 CFR。

    返回 (是否VFR, 变异系数, 说明)。
    原理：取一段 PTS 时间戳，算相邻间隔的「标准差/均值」(变异系数)。
    CFR 时该值≈0；用比值而非绝对差，避免受帧率高低影响。
    """
    p = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "packet=pts_time", "-of", "csv=p=0",
        "-read_intervals", f"%+{int(sample_sec)}", str(path),
    ], capture_output=True, text=True)
    if p.returncode != 0:
        return False, 0.0, "无法检测"
    ts = []
    for line in p.stdout.splitlines():
        try:
            ts.append(float(line.strip().rstrip(",")))
        except ValueError:
            continue
    if len(ts) < 10:
        return False, 0.0, "样本不足"
    gaps = [b - a for a, b in zip(ts, ts[1:]) if b > a]
    if len(gaps) < 5:
        return False, 0.0, "样本不足"
    mean = sum(gaps) / len(gaps)
    if mean <= 0:
        return False, 0.0, "异常"
    var = sum((g - mean) ** 2 for g in gaps) / len(gaps)
    cv = (var ** 0.5) / mean
    # 经验阈值：CFR 的 cv 通常 < 0.02，VFR 常 > 0.05
    return cv > 0.05, cv, f"帧间隔变异系数 {cv:.4f}"


def main():
    ap = argparse.ArgumentParser(description="按时间码批量剪出片段")
    ap.add_argument("video", help="输入视频")
    ap.add_argument("-c", "--clip", action="append", default=[],
                    help="时间码 START-END，可重复")
    ap.add_argument("--from-json", help="race_detect.py 的 JSON 输出")
    ap.add_argument("--pick", help="从 JSON 里挑候选序号，如 1,3,5-7")
    ap.add_argument("--outdir", default="clips", help="输出目录（默认 clips）")
    ap.add_argument("--mode", choices=["smart", "copy", "precise"], default="smart",
                    help="smart=自动选择(默认) copy=强制流复制 precise=强制重编码")
    ap.add_argument("--reencode", action="store_true",
                    help="等价于 --mode precise（保留兼容）")
    ap.add_argument("--crf", type=int, default=20, help="重编码质量，越小越好（默认 20）")
    ap.add_argument("--max-drift", type=float, default=1.0,
                    help="允许的时长偏差秒数，超出则改用重编码（默认 1.0）")
    ap.add_argument("--prefix", default="clip", help="输出文件名前缀")
    args = ap.parse_args()

    mode = "precise" if args.reencode else args.mode

    path = Path(args.video).expanduser().resolve()
    if not path.exists():
        sys.exit(f"找不到文件: {path}")

    ranges = []
    for spec in args.clip:
        try:
            ranges.append(parse_spec(spec))
        except ValueError as exc:
            sys.exit(f"✗ {exc}")

    if args.from_json:
        jp = Path(args.from_json).expanduser().resolve()
        if not jp.exists():
            sys.exit(f"找不到 JSON: {jp}")
        data = json.loads(jp.read_text())
        cands = data.get("candidates", [])
        if args.pick:
            idxs = expand_picks(args.pick)
            for i in idxs:
                match = next((c for c in cands if c["index"] == i), None)
                if not match:
                    print(f"! 跳过不存在的候选 #{i}", file=sys.stderr)
                    continue
                ranges.append((match["start"], match["end"]))
        else:
            ranges.extend((c["start"], c["end"]) for c in cands)

    if not ranges:
        sys.exit("✗ 没有指定任何片段。用 -c START-END 或 --from-json + --pick")

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = path.parent / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # 源文件帧率类型检测：OBS 的 VFR 会让流复制切割产生音画偏移
    is_vfr, cv, vfr_msg = check_vfr(path)
    if is_vfr:
        print(f"⚠ 检测到可变帧率(VFR)：{vfr_msg}")
        print("  OBS 默认录制常产出 VFR，流复制切割可能音画不同步。")
        print("  → 自动改用重编码以重建均匀时间轴。")
        if mode == "copy":
            print("  但你指定了 --mode copy，仍按流复制执行（风险自负）")
        elif mode == "smart":
            mode = "precise"

    # smart 模式：先看关键帧间隔，判断流复制是否可信
    kf = keyframe_spacing(path) if mode == "smart" else None
    if mode == "smart":
        if kf is None:
            print("! 无法探测关键帧间隔，改用重编码以确保精确")
            mode = "precise"
        else:
            shortest = min(b - a for a, b in ranges) if ranges else 0
            print(f"▶ 关键帧间隔约 {kf:.2f}s，最短片段 {shortest:.1f}s")
            if kf > args.max_drift:
                print(f"  间隔大于允许偏差 {args.max_drift}s → 用重编码保证切点精确")
                mode = "precise"
            else:
                print("  间隔足够小 → 流复制（快且零损失）")

    label = {"copy": "流复制(极快/零损失)", "precise": "重编码(帧级精确)"}[mode]
    print(f"\n▶ 剪出 {len(ranges)} 段 → {outdir}")
    print(f"  模式: {label}\n")

    def build_cmd(a, b, out, use_copy):
        c = ["ffmpeg", "-v", "error", "-stats",
             "-ss", f"{a:.3f}", "-i", str(path), "-t", f"{b - a:.3f}",
             "-map", "0:v:0?", "-map", "0:a:0?"]
        if use_copy:
            c += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
        else:
            c += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(args.crf),
                  "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k"]
        return c + ["-y", str(out)]

    manifest = []
    for i, (a, b) in enumerate(ranges, 1):
        if b <= a:
            print(f"! 跳过非法区间 {a} → {b}", file=sys.stderr)
            continue
        out = outdir / f"{args.prefix}_{i:02d}_{fmt_tc(a)}-{fmt_tc(b)}.mp4"
        dur = b - a
        print(f"[{i}/{len(ranges)}] {a:.1f}s → {b:.1f}s  目标 {dur:.1f}s  {out.name}")

        use_copy = (mode == "copy")
        p = subprocess.run(build_cmd(a, b, out, use_copy), capture_output=True)
        if p.returncode != 0 or not out.exists():
            print(f"  ✗ 失败: {p.stderr.decode(errors='replace')[-600:]}", file=sys.stderr)
            continue

        actual = probe_duration(out)
        drift = actual - dur
        # 校验：流复制漂移超限就自动重剪
        if use_copy and abs(drift) > args.max_drift:
            print(f"  ! 流复制偏差 {drift:+.1f}s（关键帧吸附），自动改用重编码重剪...")
            p2 = subprocess.run(build_cmd(a, b, out, False), capture_output=True)
            if p2.returncode == 0 and out.exists():
                actual = probe_duration(out)
                drift = actual - dur
            else:
                print(f"  ✗ 重剪也失败，保留原结果", file=sys.stderr)

        size_mb = out.stat().st_size / 1048576
        flag = "✓" if abs(drift) <= max(args.max_drift, 0.5) else "⚠"
        print(f"  {flag} 实际 {actual:.2f}s (偏差 {drift:+.2f}s)  {size_mb:.1f} MB")
        manifest.append({"file": str(out), "start": a, "end": b,
                         "requested": round(dur, 3), "actual": round(actual, 3),
                         "drift": round(drift, 3)})

    if manifest:
        mf = outdir / "clips_manifest.json"
        mf.write_text(json.dumps({"source": str(path), "clips": manifest},
                                 ensure_ascii=False, indent=2))
        total = sum(m["actual"] for m in manifest)
        worst = max((abs(m["drift"]) for m in manifest), default=0.0)
        print(f"\n✓ 完成 {len(manifest)} 段，合计 {total:.1f}s ({total / 60:.2f} 分钟)")
        print(f"  最大时长偏差 {worst:.2f}s")
        print(f"✓ 清单: {mf}")
        if len(manifest) > 1:
            print("\n下一步：用 concat_clips.py 合成一条完整高光视频。")


if __name__ == "__main__":
    main()
