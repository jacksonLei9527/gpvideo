# Repository Guidelines

## Project Structure & Module Organization

This repository contains a Python and FFmpeg workflow for finding and assembling racing highlights. The main command-line scripts are at the repository root: `make_highlight.py` coordinates the workflow, `race_detect.py` finds candidate moments, `extract_clips.py` cuts segments, and `concat_clips.py` joins them. `make_test_video.py` generates synthetic input. The `raccut/` directory contains a desktop interface and its dependency list; `old_videos/` contains sample/output media. Put generated artifacts in an output directory.

## Build, Test, and Development Commands

There is no separate build step. Python 3 and an `ffmpeg`/`ffprobe` installation are required for the command-line workflow; the core scripts use the Python standard library.

- `python3 make_highlight.py race.mkv --minutes 8 --height 720` runs detection, extraction, and assembly.
- `python3 race_detect.py race.mkv --json highlights.json --thumbs` finds candidates and writes review artifacts.
- `python3 extract_clips.py race.mkv --from-json highlights.json --pick 1,3 --outdir clips` extracts selected segments.
- `python3 concat_clips.py --manifest clips/clips_manifest.json -o highlight.mp4` assembles the result.
- `python3 make_test_video.py` creates synthetic footage for manual checks.

For the desktop interface, install `raccut/requirements.txt` in a virtual environment. Do not commit generated media or local environment files.

## Coding Style & Naming Conventions

Follow the existing Python style: four spaces for indentation, descriptive `snake_case` names, and small functions with explicit argument parsing. Document CLI options in `--help` and update `README.md` when behavior changes. No formatter or linter configuration is present.

## Testing Guidelines

No automated test suite or coverage target is configured. Use `make_test_video.py` for a reproducible synthetic input and run the relevant workflow command on it. Check candidate times and output duration; for concat changes, confirm playback and audio/video sync. Keep test artifacts outside the source tree.

## Commit & Pull Request Guidelines

This directory has no Git metadata, so existing commit conventions cannot be verified. Use a short imperative commit subject, such as `Fix VFR clip timing`. Pull requests should explain the user-visible change, list the commands or sample media used to validate it, link related issues when applicable, and include screenshots or short clips for desktop/UI changes. Never include source footage, rendered videos, credentials, or `.env` contents in a patch.

## Security & Configuration

Keep local credentials and machine-specific settings in `.env` or other untracked local configuration. Review command arguments and output paths before processing large source videos, since generated files can be substantial.
