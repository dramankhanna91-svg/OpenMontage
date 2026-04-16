#!/usr/bin/env python3
"""Smoke test: google_tts tool — Gemini AI Studio and Cloud TTS modes.

Usage:
    # Gemini mode (free tier):
    GEMINI_API_KEY=<key>  python tests/test_gemini_tts_smoke.py
    GOOGLE_API_KEY=<key>  python tests/test_gemini_tts_smoke.py

    # Cloud mode:
    GOOGLE_APPLICATION_CREDENTIALS=<path-to-sa.json>  python tests/test_gemini_tts_smoke.py

Verifies:
  1. Correct mode is detected from available credentials
  2. API call succeeds and returns a non-empty audio file
  3. Output file has a valid header (RIFF for WAV, ID3/FF for MP3)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.audio.google_tts import GoogleTTS

OUTPUT_DIR = Path("projects/smoke-gemini-tts")
TEST_TEXT = (
    "Hello! This is a Google TTS smoke test. "
    "The quick brown fox jumps over the lazy dog."
)


def check_audio_file(path: Path) -> bool:
    """Return True if file exists, is non-trivially large, and has a known audio header."""
    if not path.exists() or path.stat().st_size < 1000:
        return False
    with open(path, "rb") as f:
        magic = f.read(4)
    return (
        magic[:4] == b"RIFF"         # WAV
        or magic[:3] == b"ID3"       # MP3 with ID3 tag
        or magic[:2] == b"\xff\xfb"  # MP3 frame sync
        or magic[:2] == b"\xff\xf3"  # MP3 frame sync
        or magic[:4] == b"OggS"      # OGG
    )


def main() -> None:
    tool = GoogleTTS()

    print("=" * 60)
    print("google_tts — smoke test (Gemini AI Studio + Cloud TTS)")
    print("=" * 60)

    # --- Detect mode ---
    mode, api_key = tool._detect_mode()
    status = tool.get_status()
    print(f"Status:   {status.value}")
    print(f"Mode:     {mode or 'none — no credentials found'}")

    if not mode:
        print(
            "\nSKIP: no credentials found in environment.\n"
            "  Gemini mode: set GEMINI_API_KEY or GOOGLE_API_KEY (from aistudio.google.com)\n"
            "  Cloud mode:  set GOOGLE_APPLICATION_CREDENTIALS (service account JSON path)"
        )
        sys.exit(2)

    # --- Describe what we're about to call ---
    if mode == "gemini":
        print(f"Endpoint: generativelanguage.googleapis.com (Gemini AI Studio)")
        print(f"Model:    {tool._GEMINI_DEFAULT_MODEL}")
        print(f"Voice:    {tool._GEMINI_DEFAULT_VOICE} (default)")
        output_path = OUTPUT_DIR / "gemini_output.wav"
        inputs = {"text": TEST_TEXT, "output_path": str(output_path)}
    else:
        print(f"Endpoint: texttospeech.googleapis.com (Google Cloud TTS)")
        print(f"Voice:    en-US-Neural2-D (Cloud default)")
        output_path = OUTPUT_DIR / "cloud_output.mp3"
        inputs = {
            "text": TEST_TEXT,
            "voice": "en-US-Neural2-D",
            "language_code": "en-US",
            "output_path": str(output_path),
        }

    print(f"Text:     {TEST_TEXT!r}")
    print(f"Output:   {output_path}")
    print("Calling API...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = tool.execute(inputs)

    # --- Results ---
    print()
    if not result.success:
        print(f"FAIL: {result.error}")
        sys.exit(1)

    size = output_path.stat().st_size if output_path.exists() else 0
    audio_ok = check_audio_file(output_path)

    print(f"success:      {result.success}")
    print(f"mode:         {result.data.get('mode')}")
    print(f"voice:        {result.data.get('voice')}")
    print(f"format:       {result.data.get('format')}")
    if result.data.get("sample_rate"):
        print(f"sample_rate:  {result.data['sample_rate']} Hz")
    print(f"duration:     {result.duration_seconds}s")
    print(f"cost:         ${result.cost_usd}")
    print(f"file:         {size} bytes, valid audio header: {audio_ok}")
    print(f"model tag:    {result.model}")

    if audio_ok:
        print("\nPASS")
        sys.exit(0)
    else:
        print("\nFAIL: output file is missing, too small, or has an unrecognised header")
        sys.exit(1)


if __name__ == "__main__":
    main()
