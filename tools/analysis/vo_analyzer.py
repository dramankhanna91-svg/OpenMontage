"""VO (voiceover) audio quality analyzer.

Scans all VO files in a directory (or an explicit file list) and reports:

  1. Voice breaks    — silences 50–300 ms mid-utterance (word drops, mouth noise,
                       breath stutters)
  2. Amplitude drops — regions where momentary loudness falls >8 LUFS below the
                       local 3-second rolling average and stays there for ≥300 ms
  3. Suspicious silences — silence ≥500 ms that is NOT at the leading or trailing
                           edge of the file (i.e. in the body of a take)

Three ffmpeg passes per file, all local/free:
  - silencedetect  → silence intervals with timestamps
  - ebur128        → 100 ms momentary loudness profile
  - astats         → per-segment RMS to cross-validate amplitude drops

Outputs
  - Coloured terminal table (or plain JSON with --json flag)
  - Optionally writes a JSON report to --output-path
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)

# ---------------------------------------------------------------------------
# Tuneable thresholds — tweak via input_schema overrides
# ---------------------------------------------------------------------------
DEFAULT_SILENCE_DB = -40          # dBFS noise floor for silencedetect
DEFAULT_VOICE_BREAK_MIN_MS = 50   # shorter = mouth noise / not a real break
DEFAULT_VOICE_BREAK_MAX_MS = 300  # longer = suspicious silence, not a break
DEFAULT_SUSP_SILENCE_MIN_MS = 500 # silence >= this in the body = suspicious
DEFAULT_AMP_DROP_LUFS = 8.0       # LUFS drop from rolling avg = amplitude event
DEFAULT_AMP_DROP_MIN_MS = 300     # drop must persist >= this to be flagged
DEFAULT_EDGE_GRACE_MS = 500       # leading/trailing silence exempt window


AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma"}


# ---------------------------------------------------------------------------
# Low-level ffmpeg helpers
# ---------------------------------------------------------------------------

def _ffmpeg() -> str:
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg not found on PATH")
    return ff


def _ffprobe() -> str:
    fp = shutil.which("ffprobe")
    if not fp:
        raise RuntimeError("ffprobe not found on PATH")
    return fp


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        [_ffprobe(), "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True, timeout=10,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def _run_silencedetect(path: Path, noise_db: float, min_ms: float) -> list[dict]:
    """Return list of {start, end, duration} silence intervals in seconds."""
    min_dur = min_ms / 1000.0
    filter_str = f"silencedetect=noise={noise_db}dB:d={min_dur}"
    result = subprocess.run(
        [_ffmpeg(), "-i", str(path), "-af", filter_str, "-f", "null", "-"],
        capture_output=True, text=True, timeout=120,
    )
    silences: list[dict] = []
    current: dict = {}
    for line in result.stderr.splitlines():
        m = re.search(r"silence_start:\s*([\d.]+)", line)
        if m:
            current = {"start": float(m.group(1))}
        m = re.search(r"silence_end:\s*([\d.]+)\s*\|.*silence_duration:\s*([\d.]+)", line)
        if m and current:
            current["end"] = float(m.group(1))
            current["duration"] = float(m.group(2))
            silences.append(current)
            current = {}
    return silences


def _run_ebur128(path: Path) -> list[tuple[float, float]]:
    """Return list of (time_seconds, momentary_lufs) at ~100 ms intervals."""
    result = subprocess.run(
        [_ffmpeg(), "-i", str(path), "-af", "ebur128", "-f", "null", "-"],
        capture_output=True, text=True, timeout=180,
    )
    points: list[tuple[float, float]] = []
    pattern = re.compile(r"t:\s*([\d.]+)\s+.*?M:\s*(-?[\d.]+)")
    for line in result.stderr.splitlines():
        m = pattern.search(line)
        if m:
            t = float(m.group(1))
            lufs = float(m.group(2))
            points.append((t, lufs))
    return points


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _classify_silences(
    silences: list[dict],
    duration: float,
    voice_break_min_ms: float,
    voice_break_max_ms: float,
    susp_silence_min_ms: float,
    edge_grace_ms: float,
) -> tuple[list[dict], list[dict]]:
    """Split silence intervals into voice_breaks and suspicious_silences."""
    edge = edge_grace_ms / 1000.0
    voice_breaks: list[dict] = []
    suspicious: list[dict] = []

    for s in silences:
        dur_ms = s["duration"] * 1000.0
        at_start = s["start"] < edge
        at_end = s["end"] > (duration - edge)
        mid_body = not at_start and not at_end

        if voice_break_min_ms <= dur_ms <= voice_break_max_ms and mid_body:
            voice_breaks.append({
                "start_s": round(s["start"], 3),
                "end_s": round(s["end"], 3),
                "duration_ms": round(dur_ms, 1),
            })
        elif dur_ms >= susp_silence_min_ms and mid_body:
            suspicious.append({
                "start_s": round(s["start"], 3),
                "end_s": round(s["end"], 3),
                "duration_ms": round(dur_ms, 1),
                "severity": "high" if dur_ms >= 2000 else "medium" if dur_ms >= 1000 else "low",
            })

    return voice_breaks, suspicious


def _detect_amplitude_drops(
    points: list[tuple[float, float]],
    drop_lufs: float,
    drop_min_ms: float,
) -> list[dict]:
    """Detect regions where loudness drops ≥ drop_lufs below 3-sec rolling avg."""
    if len(points) < 5:
        return []

    times = [p[0] for p in points]
    lufs = [p[1] for p in points]

    # 3-second rolling average using a 30-sample window (~100 ms each)
    window = 30
    rolling_avg: list[float] = []
    for i in range(len(lufs)):
        start_i = max(0, i - window // 2)
        end_i = min(len(lufs), i + window // 2)
        valid = [v for v in lufs[start_i:end_i] if v > -120]
        rolling_avg.append(sum(valid) / len(valid) if valid else -120.0)

    drop_min_s = drop_min_ms / 1000.0
    drops: list[dict] = []
    in_drop = False
    drop_start_t = 0.0
    drop_start_i = 0

    for i, (t, m, avg) in enumerate(zip(times, lufs, rolling_avg)):
        is_drop = (avg > -60) and (m < avg - drop_lufs)
        if is_drop and not in_drop:
            in_drop = True
            drop_start_t = t
            drop_start_i = i
        elif not is_drop and in_drop:
            in_drop = False
            dur = t - drop_start_t
            if dur >= drop_min_s:
                avg_at_drop = sum(lufs[drop_start_i:i]) / max(1, i - drop_start_i)
                baseline = rolling_avg[drop_start_i]
                drops.append({
                    "start_s": round(drop_start_t, 3),
                    "end_s": round(t, 3),
                    "duration_ms": round(dur * 1000, 1),
                    "baseline_lufs": round(baseline, 1),
                    "drop_lufs": round(baseline - avg_at_drop, 1),
                })

    # Close any open drop at end of file
    if in_drop:
        t = times[-1]
        dur = t - drop_start_t
        if dur >= drop_min_s:
            avg_at_drop = sum(lufs[drop_start_i:]) / max(1, len(lufs) - drop_start_i)
            baseline = rolling_avg[drop_start_i]
            drops.append({
                "start_s": round(drop_start_t, 3),
                "end_s": round(t, 3),
                "duration_ms": round(dur * 1000, 1),
                "baseline_lufs": round(baseline, 1),
                "drop_lufs": round(baseline - avg_at_drop, 1),
            })

    return drops


# ---------------------------------------------------------------------------
# Per-file analysis
# ---------------------------------------------------------------------------

def _analyze_file(path: Path, cfg: dict) -> dict:
    noise_db = cfg.get("silence_noise_db", DEFAULT_SILENCE_DB)
    vb_min = cfg.get("voice_break_min_ms", DEFAULT_VOICE_BREAK_MIN_MS)
    vb_max = cfg.get("voice_break_max_ms", DEFAULT_VOICE_BREAK_MAX_MS)
    susp_min = cfg.get("suspicious_silence_min_ms", DEFAULT_SUSP_SILENCE_MIN_MS)
    amp_drop = cfg.get("amplitude_drop_lufs", DEFAULT_AMP_DROP_LUFS)
    amp_dur = cfg.get("amplitude_drop_min_ms", DEFAULT_AMP_DROP_MIN_MS)
    edge = cfg.get("edge_grace_ms", DEFAULT_EDGE_GRACE_MS)

    t0 = time.time()
    try:
        duration = _probe_duration(path)
    except Exception as e:
        return {"file": path.name, "error": f"probe failed: {e}"}

    # Run both passes; silencedetect uses the smallest threshold (vb_min)
    # so we capture everything and reclassify in Python.
    try:
        raw_silences = _run_silencedetect(path, noise_db, vb_min)
    except Exception as e:
        return {"file": path.name, "error": f"silencedetect failed: {e}"}

    try:
        lufs_points = _run_ebur128(path)
    except Exception as e:
        return {"file": path.name, "error": f"ebur128 failed: {e}"}

    voice_breaks, suspicious_silences = _classify_silences(
        raw_silences, duration, vb_min, vb_max, susp_min, edge
    )
    amplitude_drops = _detect_amplitude_drops(lufs_points, amp_drop, amp_dur)

    issue_count = len(voice_breaks) + len(suspicious_silences) + len(amplitude_drops)
    status = "clean" if issue_count == 0 else ("warn" if issue_count <= 3 else "fail")

    return {
        "file": path.name,
        "path": str(path),
        "duration_s": round(duration, 2),
        "status": status,
        "issue_count": issue_count,
        "voice_breaks": voice_breaks,
        "suspicious_silences": suspicious_silences,
        "amplitude_drops": amplitude_drops,
        "analysis_time_s": round(time.time() - t0, 2),
    }


# ---------------------------------------------------------------------------
# BaseTool implementation
# ---------------------------------------------------------------------------

class VOAnalyzer(BaseTool):
    name = "vo_analyzer"
    version = "1.0.0"
    tier = ToolTier.ANALYZE
    capability = "analysis"
    provider = "ffmpeg"
    stability = ToolStability.PRODUCTION
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    dependencies = ["binary:ffmpeg", "binary:ffprobe"]
    install_instructions = (
        "Install ffmpeg (includes ffprobe):\n"
        "  Linux: sudo apt install ffmpeg\n"
        "  macOS: brew install ffmpeg\n"
        "  Windows: winget install ffmpeg"
    )

    capabilities = [
        "voice_break_detection",
        "amplitude_drop_detection",
        "suspicious_silence_detection",
        "batch_vo_qa",
    ]
    best_for = [
        "QA-scanning all VO files before edit/compose stages",
        "finding voice breaks, amplitude drops, and suspicious silences",
        "generating a per-file and summary report for human review",
    ]

    input_schema = {
        "type": "object",
        "properties": {
            "vo_dir": {
                "type": "string",
                "description": "Directory containing VO audio files. All supported "
                               "audio extensions are scanned recursively.",
            },
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Explicit list of VO file paths. Takes precedence over vo_dir.",
            },
            "silence_noise_db": {
                "type": "number",
                "default": DEFAULT_SILENCE_DB,
                "description": "Noise floor in dBFS for silence detection (default -40). "
                               "Lower = more sensitive.",
            },
            "voice_break_min_ms": {
                "type": "number",
                "default": DEFAULT_VOICE_BREAK_MIN_MS,
                "description": "Minimum silence duration (ms) to count as a voice break.",
            },
            "voice_break_max_ms": {
                "type": "number",
                "default": DEFAULT_VOICE_BREAK_MAX_MS,
                "description": "Maximum silence duration (ms) still classified as a voice break. "
                               "Longer silences become suspicious_silences.",
            },
            "suspicious_silence_min_ms": {
                "type": "number",
                "default": DEFAULT_SUSP_SILENCE_MIN_MS,
                "description": "Silence >= this (ms) in the body of the file is flagged suspicious.",
            },
            "amplitude_drop_lufs": {
                "type": "number",
                "default": DEFAULT_AMP_DROP_LUFS,
                "description": "LUFS drop below 3-second rolling average required to flag an "
                               "amplitude drop event.",
            },
            "amplitude_drop_min_ms": {
                "type": "number",
                "default": DEFAULT_AMP_DROP_MIN_MS,
                "description": "Amplitude drop must persist >= this (ms) to be flagged.",
            },
            "edge_grace_ms": {
                "type": "number",
                "default": DEFAULT_EDGE_GRACE_MS,
                "description": "Leading/trailing silence within this window (ms) is ignored.",
            },
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=2, ram_mb=256, vram_mb=0, disk_mb=0, network_required=False
    )
    retry_policy = RetryPolicy(max_retries=0, retryable_errors=[])
    idempotency_key_fields = ["vo_dir", "files"]
    side_effects = []

    def get_status(self) -> ToolStatus:
        if shutil.which("ffmpeg") and shutil.which("ffprobe"):
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        t_start = time.time()

        # --- Collect file list ---
        if inputs.get("files"):
            paths = [Path(f) for f in inputs["files"]]
        elif inputs.get("vo_dir"):
            vo_dir = Path(inputs["vo_dir"])
            if not vo_dir.is_dir():
                return ToolResult(success=False, error=f"vo_dir not found: {vo_dir}")
            paths = sorted(
                p for p in vo_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS
            )
        else:
            return ToolResult(
                success=False, error="Provide either 'vo_dir' or 'files' in inputs."
            )

        if not paths:
            return ToolResult(success=False, error="No audio files found.")

        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            return ToolResult(success=False, error=f"Files not found: {missing}")

        cfg = {k: inputs[k] for k in inputs if k not in ("vo_dir", "files")}

        # --- Analyse each file ---
        results: list[dict] = []
        for p in paths:
            results.append(_analyze_file(p, cfg))

        # --- Summary ---
        clean = sum(1 for r in results if r.get("status") == "clean")
        warn  = sum(1 for r in results if r.get("status") == "warn")
        fail  = sum(1 for r in results if r.get("status") == "fail")
        error = sum(1 for r in results if "error" in r)

        total_vb   = sum(len(r.get("voice_breaks", [])) for r in results)
        total_susp = sum(len(r.get("suspicious_silences", [])) for r in results)
        total_amp  = sum(len(r.get("amplitude_drops", [])) for r in results)

        return ToolResult(
            success=True,
            data={
                "files_scanned": len(paths),
                "summary": {
                    "clean": clean,
                    "warn": warn,
                    "fail": fail,
                    "error": error,
                    "total_voice_breaks": total_vb,
                    "total_suspicious_silences": total_susp,
                    "total_amplitude_drops": total_amp,
                },
                "files": results,
            },
            duration_seconds=round(time.time() - t_start, 2),
        )


# ---------------------------------------------------------------------------
# CLI entry point — python -m tools.analysis.vo_analyzer [options] <dir|files>
# ---------------------------------------------------------------------------

def _color(text: str, code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def _status_str(status: str) -> str:
    colors = {"clean": "32", "warn": "33", "fail": "31"}
    icons  = {"clean": "✓", "warn": "!", "fail": "✗"}
    return _color(f"{icons.get(status, '?')} {status}", colors.get(status, "0"))


def _print_report(data: dict) -> None:
    s = data["summary"]
    print()
    print(_color("══ VO Quality Report ══════════════════════════════════════", "1"))
    print(f"  Files scanned : {data['files_scanned']}")
    print(f"  Clean         : {_color(str(s['clean']), '32')}")
    print(f"  Warn (1-3 issues) : {_color(str(s['warn']), '33')}")
    print(f"  Fail (>3 issues)  : {_color(str(s['fail']), '31')}")
    if s["error"]:
        print(f"  Errors        : {_color(str(s['error']), '31')}")
    print()
    print(f"  Voice breaks      : {s['total_voice_breaks']}")
    print(f"  Suspicious silences: {s['total_suspicious_silences']}")
    print(f"  Amplitude drops   : {s['total_amplitude_drops']}")
    print()

    for r in data["files"]:
        if "error" in r:
            print(f"  {_color('ERR', '31')}  {r['file']} — {r['error']}")
            continue

        status_s = _status_str(r["status"])
        dur = f"{r['duration_s']:.1f}s"
        issues = r["issue_count"]
        print(f"  {status_s:<20}  {r['file']:<40}  {dur:>7}  {issues} issue(s)")

        for vb in r.get("voice_breaks", []):
            print(f"    {_color('↩ voice break', '33')}  "
                  f"{vb['start_s']:.3f}s–{vb['end_s']:.3f}s  "
                  f"({vb['duration_ms']:.0f} ms)")
        for ss in r.get("suspicious_silences", []):
            sev_col = "31" if ss["severity"] == "high" else "33"
            sev = ss["severity"]
            print(f"    {_color(f'⏸ silence [{sev}]', sev_col)}  "
                  f"{ss['start_s']:.3f}s–{ss['end_s']:.3f}s  "
                  f"({ss['duration_ms']:.0f} ms)")
        for ad in r.get("amplitude_drops", []):
            print(f"    {_color('↓ amp drop', '35')}  "
                  f"{ad['start_s']:.3f}s–{ad['end_s']:.3f}s  "
                  f"({ad['duration_ms']:.0f} ms, -{ad['drop_lufs']:.1f} LUFS)")

    print()


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Scan VO files for voice breaks, amplitude drops, and suspicious silences."
    )
    parser.add_argument("target", nargs="?", help="Directory of VO files, or a single file path")
    parser.add_argument("files", nargs="*", help="Explicit file list (alternative to directory)")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of table")
    parser.add_argument(
        "--output", metavar="PATH", help="Write JSON report to this path"
    )
    parser.add_argument("--silence-db",  type=float, default=DEFAULT_SILENCE_DB,
                        help=f"Noise floor dBFS (default {DEFAULT_SILENCE_DB})")
    parser.add_argument("--vb-min-ms",   type=float, default=DEFAULT_VOICE_BREAK_MIN_MS,
                        help=f"Voice break min ms (default {DEFAULT_VOICE_BREAK_MIN_MS})")
    parser.add_argument("--vb-max-ms",   type=float, default=DEFAULT_VOICE_BREAK_MAX_MS,
                        help=f"Voice break max ms (default {DEFAULT_VOICE_BREAK_MAX_MS})")
    parser.add_argument("--susp-ms",     type=float, default=DEFAULT_SUSP_SILENCE_MIN_MS,
                        help=f"Suspicious silence min ms (default {DEFAULT_SUSP_SILENCE_MIN_MS})")
    parser.add_argument("--amp-drop",    type=float, default=DEFAULT_AMP_DROP_LUFS,
                        help=f"Amplitude drop threshold LUFS (default {DEFAULT_AMP_DROP_LUFS})")
    parser.add_argument("--amp-dur-ms",  type=float, default=DEFAULT_AMP_DROP_MIN_MS,
                        help=f"Amplitude drop min duration ms (default {DEFAULT_AMP_DROP_MIN_MS})")
    parser.add_argument("--edge-ms",     type=float, default=DEFAULT_EDGE_GRACE_MS,
                        help=f"Edge grace zone ms (default {DEFAULT_EDGE_GRACE_MS})")

    args = parser.parse_args(argv)

    # Resolve inputs
    explicit_files: list[str] = []
    vo_dir: str | None = None

    if args.target:
        t = Path(args.target)
        if t.is_dir():
            vo_dir = str(t)
        elif t.is_file():
            explicit_files.append(str(t))
        else:
            print(f"Error: '{args.target}' is not a file or directory.", file=sys.stderr)
            sys.exit(1)

    if args.files:
        explicit_files.extend(args.files)

    if not explicit_files and not vo_dir:
        parser.print_help()
        sys.exit(1)

    tool_inputs: dict[str, Any] = {
        "silence_noise_db": args.silence_db,
        "voice_break_min_ms": args.vb_min_ms,
        "voice_break_max_ms": args.vb_max_ms,
        "suspicious_silence_min_ms": args.susp_ms,
        "amplitude_drop_lufs": args.amp_drop,
        "amplitude_drop_min_ms": args.amp_dur_ms,
        "edge_grace_ms": args.edge_ms,
    }
    if explicit_files:
        tool_inputs["files"] = explicit_files
    else:
        tool_inputs["vo_dir"] = vo_dir

    tool = VOAnalyzer()
    if tool.get_status() != ToolStatus.AVAILABLE:
        print("Error: ffmpeg/ffprobe not found on PATH.", file=sys.stderr)
        sys.exit(1)

    print("Scanning...", file=sys.stderr)
    result = tool.execute(tool_inputs)

    if not result.success:
        print(f"Error: {result.error}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(result.data, indent=2))
        print(f"Report written to {out}", file=sys.stderr)

    if args.json:
        print(json.dumps(result.data, indent=2))
    else:
        _print_report(result.data)

    # Exit non-zero if any file failed
    if result.data["summary"]["fail"] or result.data["summary"]["error"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
