#!/usr/bin/env python3
"""Smoke test: Gemini Flash TTS via google_tts tool.

Usage:
    GEMINI_API_KEY=<your-key> python tests/test_gemini_tts_smoke.py

Verifies:
  1. Tool reports AVAILABLE when key is set
  2. API call succeeds and returns a non-empty WAV file
  3. WAV file has a valid RIFF header
"""

import os
import sys
import struct
from pathlib import Path

# Resolve project root so imports work regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.audio.google_tts import GoogleTTS

OUTPUT_PATH = Path("projects/smoke-gemini-tts/test_output.wav")
TEST_TEXT = "Hello! This is a Gemini Flash TTS smoke test. The quick brown fox jumps over the lazy dog."


def check_wav(path: Path) -> bool:
    """Return True if file starts with a valid RIFF/WAV header."""
    if not path.exists() or path.stat().st_size < 44:
        return False
    with open(path, "rb") as f:
        header = f.read(12)
    return header[:4] == b"RIFF" and header[8:12] == b"WAVE"


def main() -> None:
    tool = GoogleTTS()

    print("=" * 55)
    print("Gemini Flash TTS — smoke test")
    print("=" * 55)

    # --- Status check ---
    status = tool.get_status()
    print(f"Status:   {status.value}")
    if status.value == "unavailable":
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        print("SKIP: no API key in environment.")
        print("  Set GEMINI_API_KEY or GOOGLE_API_KEY and rerun.")
        sys.exit(2)

    # --- Generate ---
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Text:     {TEST_TEXT!r}")
    print(f"Voice:    Puck (default)")
    print(f"Model:    {tool._DEFAULT_MODEL}")
    print(f"Output:   {OUTPUT_PATH}")
    print("Calling API...")

    result = tool.execute({
        "text": TEST_TEXT,
        "voice": "Puck",
        "output_path": str(OUTPUT_PATH),
    })

    if not result.success:
        print(f"\nFAIL: {result.error}")
        sys.exit(1)

    # --- Verify file ---
    size = OUTPUT_PATH.stat().st_size if OUTPUT_PATH.exists() else 0
    wav_ok = check_wav(OUTPUT_PATH)

    print(f"\nResult:   success={result.success}")
    print(f"Duration: {result.duration_seconds}s")
    print(f"Cost:     ${result.cost_usd}")
    print(f"File:     {size} bytes, valid WAV header: {wav_ok}")
    print(f"Model:    {result.model}")

    if wav_ok and size > 1000:
        print("\nPASS")
        sys.exit(0)
    else:
        print("\nFAIL: output file is missing or too small")
        sys.exit(1)


if __name__ == "__main__":
    main()
