#!/usr/bin/env python3
"""
一键出片：赛车录像 → 720p 高光视频

把三步串起来，适合你已经确认好参数、想省事的场景：
  ① race_detect.py     找高光候选
  ② extract_clips.py   剪出选中的段
  ③ concat_clips.py    统一转 720p 并合成

用法：
  # 全自动：自动挑候选凑够 8 分钟
  python3 make_highlight.py race.mkv

  # 指定目标时长与规格
  python3 make_highlight.py race.mkv --minutes 8 --height 720

  # 你自己看缩略图挑好了段，只做后两步
  python3 make_highlight.py race.mkv --skip-detect --from-json hl.json --pick 2,5,7,9

  # 只探测不剪（先看看候选质量再决定）
  python3 make_highlight.py race.mkv --detect-only

注意：
  自动挑段只看"活动量"排序，分不清超车和普通过弯。
  想要更准的结果，先用 --detect-only --thumbs 出缩略图人工过一眼，
  再用 --skip-detect --pick 指定要哪几段。
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_step(title, cmd):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    p = subprocess.run(cmd)
    if p.returncode != 0:
        sys.exit(f"\n✗ 失败于: {title}（退出码 {p.returncode}）")
    return True


def main():
    ap = argparse.ArgumentParser(description="一键出 720p 高光视频")
    ap.add_argument("video", help="源录像（MKV/MP4/MOV 均可）")
    ap.add_argument("--minutes", type=float, default=8.0, help="目标时长分钟数（默认 8）")
    ap.add_argument("--height", type=int, default=720, help="输出高度（默认 720）")
    ap.add_argument("--fps", type=float, default=30, help="输出帧率（默认 30）")
    ap.add_argument("--crf", type=int, default=21, help="输出质量，越小越好（默认 21）")
    ap.add_argument("--k", type=float, default=3.0, help="活动量阈值倍数（默认 3.0）")
    ap.add_argument("--pad", type=float, default=4.0, help="每段前后留白秒数")
    ap.add_argument("--thumbs", action="store_true", help="生成候选缩略图供人工确认")
    ap.add_argument("--detect-only", action="store_true", help="只探测，不剪辑")
    ap.add_argument("--skip-detect", action="store_true",
                    help="跳过探测，用已有的 --from-json")
    ap.add_argument("--from-json", default="hl.json", help="候选 JSON 路径（默认 hl.json）")
    ap.add_argument("--pick", help="指定候选序号，如 2,5,7-9。不给则自动凑够目标时长")
    ap.add_argument("-o", "--output", help="输出文件名（默认 highlight_720p.mp4）")
    args = ap.parse_args()

    src = Path(args.video).expanduser().resolve()
    if not src.exists():
        sys.exit(f"找不到文件: {src}")

    workdir = src.parent
    json_path = Path(args.from_json)
    if not json_path.is_absolute():
        json_path = workdir / json_path
    target_sec = args.minutes * 60

    # 默认输出宽度按 16:9 推导
    width = int(round(args.height * 16 / 9 / 2) * 2)
    out_name = args.output or f"highlight_{args.height}p.mp4"
    clips_dir = workdir / "clips"

    print(f"源文件   : {src.name}")
    print(f"目标时长 : {args.minutes:g} 分钟 ({target_sec:.0f} 秒)")
    print(f"输出规格 : {width}x{args.height} @ {args.fps}fps, CRF {args.crf}")

    # ① 探测
    if not args.skip_detect:
        cmd = [sys.executable, str(HERE / "race_detect.py"), str(src),
               "--json", str(json_path), "--k", str(args.k), "--pad", str(args.pad)]
        if args.pick:
            # 指定了序号就不必为了凑时长而筛选
            cmd += ["--top", "50"]
        else:
            cmd += ["--target", str(target_sec)]
        if args.thumbs:
            cmd += ["--thumbs"]
        run_step("① 探测高光候选", cmd)
    else:
        if not json_path.exists():
            sys.exit(f"--skip-detect 需要已存在的候选文件: {json_path}")
        print(f"\n跳过探测，使用 {json_path}")

    if args.detect_only:
        print(f"\n✓ 仅探测完成。候选见 {json_path}")
        if args.thumbs:
            print(f"  缩略图见 {workdir / 'thumbs'}")
        print("  确认后运行：")
        print(f"    python3 {Path(__file__).name} {src.name} --skip-detect "
              f"--pick 2,5,7,9")
        return

    # ② 剪出
    cmd = [sys.executable, str(HERE / "extract_clips.py"), str(src),
           "--from-json", str(json_path), "--outdir", str(clips_dir)]
    if args.pick:
        cmd += ["--pick", args.pick]
    run_step("② 剪出选中的片段", cmd)

    # ③ 合成 720p
    manifest = clips_dir / "clips_manifest.json"
    if not manifest.exists():
        sys.exit(f"✗ 没找到片段清单 {manifest}，请检查上一步输出")
    run_step("③ 转码 720p 并合成",
             [sys.executable, str(HERE / "concat_clips.py"),
              "--manifest", str(manifest),
              "-o", str(workdir / out_name),
              "--width", str(width), "--height", str(args.height),
              "--fps", str(args.fps), "--crf", str(args.crf)])

    final = workdir / out_name
    if final.exists():
        mb = final.stat().st_size / 1048576
        print(f"\n{'=' * 60}")
        print(f"✓ 成品: {final}")
        print(f"  {mb:.0f} MB")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
