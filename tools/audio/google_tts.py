"""Google Gemini Flash Text-to-Speech provider tool.

Uses the Gemini AI Studio API (generativelanguage.googleapis.com) — free tier available.
30 expressive voices, 70+ languages, auto language detection.

IMPORTANT: This tool uses GEMINI_API_KEY (Google AI Studio), NOT a Google Cloud key.
Get a free key at https://aistudio.google.com/apikey
"""

from __future__ import annotations

import base64
import os
import struct
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


def _pcm_to_wav(
    pcm_data: bytes,
    sample_rate: int = 24000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Wrap raw PCM bytes in a RIFF/WAV container header."""
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    data_size = len(pcm_data)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,   # RIFF chunk size
        b"WAVE",
        b"fmt ",
        16,               # PCM fmt subchunk size
        1,                # audio format: PCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        sample_width * 8, # bits per sample
        b"data",
        data_size,
    )
    return header + pcm_data


class GoogleTTS(BaseTool):
    name = "google_tts"
    version = "0.2.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "google_tts"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.API

    dependencies = []
    install_instructions = (
        "Set GEMINI_API_KEY (or GOOGLE_API_KEY) to your Google AI Studio API key.\n"
        "  Get a free key at https://aistudio.google.com/apikey\n"
        "  NOTE: this is NOT a Google Cloud key — use the AI Studio key for the free tier."
    )
    fallback = "openai_tts"
    fallback_tools = ["openai_tts", "elevenlabs_tts", "piper_tts"]
    agent_skills = ["text-to-speech"]

    capabilities = [
        "text_to_speech",
        "voice_selection",
        "multilingual",
        "audio_style_tags",
    ]
    supports = {
        "voice_cloning": False,
        "multilingual": True,
        "offline": False,
        "native_audio": True,
        "ssml": False,
        "audio_tags": True,
    }
    best_for = [
        "free-tier TTS via Google AI Studio (Gemini Flash)",
        "30 expressive voices, 70+ languages, auto language detection",
        "style control via inline audio tags e.g. [whispers] or [cheerfully]",
    ]
    not_good_for = [
        "voice cloning",
        "fully offline production",
        "MP3/OGG output (outputs WAV only)",
    ]

    # Gemini AI Studio base URL — NOT texttospeech.googleapis.com
    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    _DEFAULT_MODEL = "gemini-2.5-flash-preview-tts"
    _DEFAULT_VOICE = "Puck"

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {
                "type": "string",
                "description": (
                    "Text to convert to speech. Supports inline audio style tags "
                    "such as [whispers], [cheerfully], [slowly], [laughs]."
                ),
            },
            "voice": {
                "type": "string",
                "default": _DEFAULT_VOICE,
                "description": (
                    "Gemini voice name. Available voices: Puck (default, neutral/clear), "
                    "Zephyr (airy), Aoede (warm female), Orus (deep male), Kore (bright female), "
                    "Fenrir (gruff), Charon (calm), Leda, Callirrhoe, Autonoe, Enceladus, "
                    "Iapetus, Umbriel, Algieba, Despina, Erinome, Algenib, Rasalgethi, "
                    "Laomedeia, Achernar, Alnilam, Schedar, Gacrux, Pulcherrima, Achird, "
                    "Zubenelgenubi, Vindemiatrix, Sadachbia, Sadaltager, Sulafat."
                ),
            },
            "model": {
                "type": "string",
                "default": _DEFAULT_MODEL,
                "description": "Gemini TTS model ID. Default: gemini-2.5-flash-preview-tts",
            },
            "output_path": {
                "type": "string",
                "description": "Output file path. Always written as a WAV file.",
            },
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=50, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["text", "voice", "model"]
    side_effects = ["writes WAV file to output_path", "calls Gemini AI Studio TTS API"]
    user_visible_verification = ["Listen to generated audio for natural speech quality"]

    def _get_api_key(self) -> str | None:
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def get_status(self) -> ToolStatus:
        if self._get_api_key():
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        # Gemini 2.5 Flash TTS: ~$0.50/1M input tokens; ~100 chars ≈ 25 tokens
        text = inputs.get("text", "")
        tokens = len(text) / 4  # rough chars-to-tokens
        return round(tokens * 0.0000005, 6)  # $0.50 / 1M tokens

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = self._get_api_key()
        if not api_key:
            return ToolResult(
                success=False,
                error="No API key found. " + self.install_instructions,
            )

        start = time.time()
        try:
            result = self._generate(inputs, api_key)
        except Exception as exc:
            return ToolResult(success=False, error=f"Gemini TTS failed: {exc}")

        result.duration_seconds = round(time.time() - start, 2)
        result.cost_usd = self.estimate_cost(inputs)
        return result

    def _generate(self, inputs: dict[str, Any], api_key: str) -> ToolResult:
        import requests

        text = inputs["text"]
        voice_name = inputs.get("voice", self._DEFAULT_VOICE)
        model = inputs.get("model", self._DEFAULT_MODEL)

        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": voice_name,
                        }
                    }
                },
            },
        }

        url = f"{self._BASE_URL}/{model}:generateContent"
        response = requests.post(
            url,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        resp_data = response.json()
        inline = resp_data["candidates"][0]["content"]["parts"][0]["inlineData"]
        mime_type: str = inline["mimeType"]  # e.g. "audio/L16;codec=pcm;rate=24000"
        pcm_bytes = base64.b64decode(inline["data"])

        # Parse sample rate from mime type (default 24000)
        sample_rate = 24000
        for part in mime_type.split(";"):
            part = part.strip()
            if part.startswith("rate="):
                sample_rate = int(part[5:])

        wav_bytes = _pcm_to_wav(pcm_bytes, sample_rate=sample_rate)

        output_path = Path(inputs.get("output_path", "tts_output.wav"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(wav_bytes)

        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "model": model,
                "voice": voice_name,
                "text_length": len(text),
                "output": str(output_path),
                "format": "WAV",
                "sample_rate": sample_rate,
            },
            artifacts=[str(output_path)],
            model=f"gemini-tts/{model}/{voice_name}",
        )
