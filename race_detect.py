#!/usr/bin/env python3
"""
赛车录像高光探测（纯标准库，无需 numpy）

核心信号 —— 为什么不是"音量尖峰"：
  赛车视频音轨大部分时间是持续高位的引擎轰鸣，能量长期处于高位，
  简单峰值检测会被引擎噪音淹没。实测也证实音量几乎没有区分度。

改用「画面活动量」为主信号：
  转播在精彩时刻会切镜头、放慢动作回放、插字幕板/名次板，
  表现为帧间差异像素量骤增。
  实现上用 ffmpeg 的 scdet 滤镜逐帧输出 lavfi.scd.mafd
  （平均绝对帧间差，0-255 尺度），取数值曲线而不是阈值事件——
  连续数值可自适应，不用猜一个固定阈值。

  实测验证（人造测试片，平静段=静态画面，高光段=每0.5s硬切）：
    平静段 0-1 次切换/分钟，高光段 30+ 次/15秒，信噪比极高。

音频作为辅助信号：抓"相对突变"（撞击、锁死刹车、突然的欢呼），
  而不是绝对值，用来给候选段分类标注。

用法：
  python3 race_highlights.py race.mp4
  python3 race_highlights.py race.mp4 --target 420 --thumbs --json hl.json
  python3 race_highlights.py race.mp4 --k 2.0 --min-dur 4
"""
import argparse
import array
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

WIN = 1.0          # 分析窗口（秒）
AUDIO_HZ = 8000    # 音频分析采样率
AUDIO_WIN = 0.5    # 音频分析窗口（秒）


def run(cmd):
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"命令失败: {' '.join(map(str, cmd))}\n"
                           f"{p.stderr.decode(errors='replace')[-1500:]}")
    return p.stdout, p.stderr.decode(errors="replace")


def probe(path):
    out, _ = run(["ffprobe", "-v", "error", "-print_format", "json",
                  "-show_format", "-show_streams", str(path)])
    return json.loads(out)


def media_summary(info):
    fmt = info.get("format", {})
    dur = float(fmt.get("duration") or 0)
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    s = {"duration": dur, "size_mb": round(int(fmt.get("size") or 0) / 1048576, 1),
         "has_video": v is not None, "has_audio": a is not None}
    if v:
        fr = v.get("avg_frame_rate", "0/1")
        try:
            n, d = fr.split("/")
            fps = round(float(n) / float(d), 3) if float(d) else 0
        except (ValueError, ZeroDivisionError):
            fps = 0
        s.update({"resolution": f"{v.get('width')}x{v.get('height')}", "fps": fps,
                  "vcodec": v.get("codec_name")})
    if a:
        s.update({"acodec": a.get("codec_name"), "channels": a.get("channels")})
    return s


def motion_curve(path, scale_w):
    """
    逐帧取 lavfi.scd.mafd，按 WIN 秒分箱取均值。
    返回 (每箱平均MAFD, 每箱内最大MAFD, 实测时长)

    注意：scdet 必须放在 filter_complex 里 —— -vf 与 -filter_complex 不能同时用，
    ffmpeg 会直接报错。
    """
    # 必须显式标输入标签 [0:v]，否则 "Cannot find an unused video input stream"
    # threshold 用 5（较低）只影响是否额外输出 scd.time 行，mafd 数值每帧都有
    vf = f"[0:v]scale={scale_w}:-2,scdet=threshold=5,metadata=print:file=-"
    out, _ = run([
        "ffmpeg", "-v", "error", "-nostats", "-i", str(path),
        "-filter_complex", vf, "-map", "0:v:0",
        "-an", "-f", "null", "-",
    ])

    times, mafds = [], []
    cur_t = None
    for line in out.decode(errors="replace").splitlines():
        line = line.strip()
        # 注意用 search 而非 match：pts_time 在行中间
        # ("frame:0    pts:0       pts_time:0")，match 只锚定行首会全部漏掉
        m = re.search(r"pts_time:([0-9.]+)", line)
        if m:
            cur_t = float(m.group(1))
            continue
        if line.startswith("lavfi.scd.mafd="):
            try:
                v = float(line.split("=", 1)[1])
            except ValueError:
                continue
            if cur_t is not None:
                times.append(cur_t)
                mafds.append(v)

    if not times:
        return [], [], 0.0

    duration = times[-1]
    nbins = max(1, int(duration / WIN) + 1)
    sums = [0.0] * nbins
    counts = [0] * nbins
    peaks = [0.0] * nbins
    for t, v in zip(times, mafds):
        b = min(nbins - 1, int(t / WIN))
        sums[b] += v
        counts[b] += 1
        if v > peaks[b]:
            peaks[b] = v
    means = [sums[i] / counts[i] if counts[i] else 0.0 for i in range(nbins)]
    return means, peaks, duration


def audio_curves(path):
    """返回 (每秒RMS曲线, 每WIN秒相对突变曲线)。"""
    raw, _ = run([
        "ffmpeg", "-v", "error", "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(AUDIO_HZ),
        "-f", "s16le", "-acodec", "pcm_s16le", "-",
    ])
    n = len(raw) // 2
    if n == 0:
        return [], []
    samples = array.array("h")
    samples.frombytes(raw[:n * 2])
    if sys.byteorder == "big":
        samples.byteswap()
    win = int(AUDIO_HZ * AUDIO_WIN)
    rms = []
    for i in range(0, n - win + 1, win):
        acc = 0
        for s in samples[i:i + win]:
            acc += s * s
        rms.append((acc / win) ** 0.5)

    # 相对突变：只关心突然变响（发动机轰鸣被基线吸收）
    delta = [0.0] * len(rms)
    for i in range(1, len(rms)):
        prev = rms[i - 1]
        delta[i] = max(0.0, (rms[i] - prev) / (prev + 200.0))

    nbins = max(1, int(len(rms) * AUDIO_WIN / WIN) + 1)
    delta_bins = [0.0] * nbins
    cnt = [0] * nbins
    for i, d in enumerate(delta):
        b = min(nbins - 1, int(i * AUDIO_WIN / WIN))
        delta_bins[b] += d
        cnt[b] += 1
    return rms, delta_bins


def robust_threshold(values, k):
    """中位数 + k*MAD 的自适应阈值。"""
    if not values:
        return 0.0
    med = statistics.median(values)
    mad = statistics.median([abs(v - med) for v in values]) or 0.0
    return med + k * 1.4826 * mad if mad > 0 else med * 1.5


def pick_segments(motion, peaks, delta_bins, k, min_dur, top, pad, target, duration):
    nbins = len(motion)
    if nbins == 0:
        return [], 0.0, {}

    med = statistics.median(motion)
    thresh = robust_threshold(motion, k)
    # 阈值不低于中位数的 1.3 倍，否则"全程高活动"的素材会全片命中
    thresh = max(thresh, med * 1.3)

    raw_segs = []
    start = None
    for i, v in enumerate(motion):
        if v > thresh:
            if start is None:
                start = i
        elif start is not None:
            raw_segs.append([start, i])
            start = None
    if start is not None:
        raw_segs.append([start, nbins])

    merged = []
    for s, e in raw_segs:
        if merged and (s - merged[-1][1]) < 3:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    filtered = [m for m in merged if (m[1] - m[0]) * WIN >= min_dur]

    segs = []
    for s, e in filtered:
        seg_m = motion[s:e]
        avg = sum(seg_m) / len(seg_m) if seg_m else 0.0
        peak = max(peaks[s:e]) if peaks[s:e] else 0.0
        segs.append({
            "start": max(0.0, s * WIN - pad),
            "end": min(duration, (e + 1) * WIN + pad),
            "motion_avg": avg, "motion_peak": peak, "score": avg,
        })

    segs.sort(key=lambda x: -x["score"])
    segs = segs[:top]
    segs.sort(key=lambda x: x["start"])

    dedup = []
    for s in segs:
        if dedup and s["start"] <= dedup[-1]["end"]:
            dedup[-1]["end"] = max(dedup[-1]["end"], s["end"])
            dedup[-1]["score"] = max(dedup[-1]["score"], s["score"])
            dedup[-1]["motion_peak"] = max(dedup[-1]["motion_peak"], s["motion_peak"])
        else:
            dedup.append(dict(s))
    segs = dedup

    select_note = ""
    if target > 0:
        total = sum(x["end"] - x["start"] for x in segs)
        if total > target:
            # 目标裁剪要的是「尽量贴近目标时长」的子集，不是简单贪心。
            # 两个已实测的贪心陷阱：
            #   1) 先取到一段长的就把后面的全排除 → 目标 300s 只凑到 129s
            #   2) 每段都超过目标时一段都不选 → 交出空白
            # 所以：优先单段贴近；否则试两段组合；再不行就取最接近的一段
            # （宁可稍微超出目标，也不要远短于目标或空白）。
            def dur(x):
                return x["end"] - x["start"]

            ranked = sorted(segs, key=lambda y: -y["score"])
            fits = [x for x in ranked if dur(x) <= target]

            if fits:
                # 单段：取时长最大者（最贴近目标且不超）
                best = [max(fits, key=dur)]
                best_acc = dur(best[0])
            else:
                best, best_acc = [], 0.0

            # 两段组合：短+长两指针，找不超目标的最大和
            if len(ranked) >= 2:
                by_d = sorted(ranked, key=dur)
                lo, hi = 0, len(by_d) - 1
                while lo < hi:
                    s = dur(by_d[lo]) + dur(by_d[hi])
                    if s <= target:
                        if s > best_acc:
                            best, best_acc = [by_d[lo], by_d[hi]], s
                        lo += 1
                    else:
                        hi -= 1

            # 兜底：没有任何组合能装下，就取最接近目标的一段
            # （保留完整高光，不切碎；成品会略超目标）
            if not best:
                best = [min(ranked, key=lambda y: abs(dur(y) - target))]
                best_acc = dur(best[0])

            segs = sorted(best, key=lambda y: y["start"])
            achieved = sum(dur(x) for x in segs)
            if achieved < target * 0.9:
                select_note = (
                    f"目标 {target:.0f}s 只凑到 {achieved:.0f}s（超阈高光总量不足）。"
                    f"想更接近目标就放宽 --k、减小 --pad，或降低 --min-dur。")
            elif achieved > target * 1.05:
                select_note = (
                    f"成品 {achieved:.0f}s 超出目标 {target:.0f}s："
                    f"单段高光本身就比目标长。想截到目标内就用 --pad 减小留白。")

    for s in segs:
        b0 = int(s["start"] / WIN)
        b1 = min(len(delta_bins) - 1, int(s["end"] / WIN))
        s["audio_delta"] = sum(delta_bins[b0:b1 + 1]) if delta_bins and b0 <= b1 else 0.0
        s["kind"] = ("镜头密集/回放" if s["motion_avg"] > med * 4 else
                     "活动量偏高" if s["motion_avg"] > med * 2 else "局部波动")

    diag = {"threshold": thresh, "median": med, "raw_segments": len(raw_segs),
            "after_merge": len(merged), "after_filter": len(filtered)}
    return segs, thresh, diag, select_note


def fmt_tc(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def make_thumbs(path, segs, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    made = []
    for i, s in enumerate(segs, 1):
        t = (s["start"] + s["end"]) / 2
        out = outdir / f"cand_{i:02d}_{int(t // 60):03d}m{int(t % 60):02d}s.jpg"
        try:
            run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path),
                 "-frames:v", "1", "-vf", "scale=640:-2", "-y", str(out)])
            made.append(str(out))
        except RuntimeError:
            pass
    return made


def main():
    ap = argparse.ArgumentParser(description="赛车录像高光探测（画面活动量主信号）")
    ap.add_argument("video")
    ap.add_argument("--top", type=int, default=25, help="最多候选段数（默认 25）")
    ap.add_argument("--target", type=float, default=0,
                    help="目标总时长秒数，如 420 = 7 分钟。0=不限")
    ap.add_argument("--pad", type=float, default=4.0, help="每段前后留白秒数（默认 4）")
    ap.add_argument("--min-dur", type=float, default=3.0, help="候选段最短时长（默认 3s）")
    ap.add_argument("--k", type=float, default=3.0, help="自适应阈值倍数，越小越敏感（默认 3.0）")
    ap.add_argument("--scale", type=int, default=320, help="分析缩放宽度，越小越快（默认 320）")
    ap.add_argument("--thumbs", action="store_true")
    ap.add_argument("--json", metavar="FILE")
    args = ap.parse_args()

    path = Path(args.video).expanduser().resolve()
    if not path.exists():
        sys.exit(f"找不到文件: {path}")

    print(f"▶ 分析: {path.name}")
    m = media_summary(probe(path))
    if not m["has_video"]:
        sys.exit("✗ 没有视频流")
    print(f"  时长 {fmt_tc(m['duration'])} | {m.get('resolution')} @ {m.get('fps')}fps "
          f"| {m['size_mb']} MB | 音轨 {'有' if m['has_audio'] else '无'}")

    print(f"▶ 计算画面活动量曲线（缩放至 {args.scale}px 宽，逐帧 MAFD）...")
    motion, peaks, measured = motion_curve(path, args.scale)
    if not motion:
        sys.exit("✗ 无法计算活动量，请确认文件可正常解码。")
    dur = m["duration"] or measured
    print(f"  {len(motion)} 个 1 秒箱，实测覆盖到 {fmt_tc(measured)}")

    delta_bins = []
    if m["has_audio"]:
        print("▶ 计算音频曲线...")
        _, delta_bins = audio_curves(path)

    segs, thresh, diag, select_note = pick_segments(
        motion, peaks, delta_bins, args.k, args.min_dur, args.top,
        args.pad, args.target, dur)

    med = diag.get("median", 0.0)
    if med:
        print(f"\n▶ 活动量基线(中位数) {med:.2f} → 阈值 {thresh:.2f} "
              f"({thresh / med:.2f}x 基线)")
    print(f"  超阈区间 {diag.get('raw_segments', 0)} 个 → 合并后 "
          f"{diag.get('after_merge', 0)} 个 → 过滤后 {diag.get('after_filter', 0)} 个")
    if select_note:
        print(f"\n! {select_note}")

    if not segs:
        print("\n✗ 没有找到明显高光。对策：")
        print("   · 降低阈值：--k 2.0")
        print("   · 缩短最短时长：--min-dur 2")
        print("   · 若全片活动量均匀（如车载固定机位全程高速），")
        print("     活动量信号失效，需要换信号或人工看缩略图选段。")
        sys.exit(1)

    total = sum(s["end"] - s["start"] for s in segs)
    print(f"\n▶ 候选高光 {len(segs)} 段，合计 {total:.1f}s ({total / 60:.2f} 分钟)")
    if dur:
        print(f"  占全片 {total / dur * 100:.1f}%")
    print(f"\n{'#':>3}  {'起':>12}  {'止':>12}  {'时长':>7}  {'活动量':>7}  "
          f"{'峰值':>6}  {'音频':>6}  类型")
    for i, s in enumerate(segs, 1):
        print(f"{i:>3}  {fmt_tc(s['start']):>12}  {fmt_tc(s['end']):>12}  "
              f"{s['end'] - s['start']:>6.1f}s  {s['motion_avg']:>7.2f}  "
              f"{s['motion_peak']:>6.1f}  {s['audio_delta']:>6.2f}  {s['kind']}")

    payload = {
        "source": str(path), "media": m,
        "params": {"k": args.k, "pad": args.pad, "min_dur": args.min_dur,
                   "scale": args.scale, "target": args.target},
        "threshold": thresh, "motion_median": med,
        "candidates": [
            {"index": i, "start": round(s["start"], 3), "end": round(s["end"], 3),
             "duration": round(s["end"] - s["start"], 3),
             "motion_avg": round(s["motion_avg"], 3),
             "motion_peak": round(s["motion_peak"], 2),
             "audio_delta": round(s["audio_delta"], 3), "kind": s["kind"]}
            for i, s in enumerate(segs, 1)],
    }
    if args.json:
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"\n✓ JSON 已写入 {args.json}")
    if args.thumbs:
        made = make_thumbs(path, segs, path.parent / "thumbs")
        print(f"✓ {len(made)} 张缩略图 → {path.parent / 'thumbs'}")

    print("\n下一步：看缩略图/对时间码挑段 → extract_clips.py 剪出 → concat_clips.py 合成")


if __name__ == "__main__":
    main()
