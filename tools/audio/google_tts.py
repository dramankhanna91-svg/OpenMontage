"""Google Text-to-Speech provider tool.

Supports two backends, auto-detected from available credentials:

  GEMINI mode  — Gemini AI Studio (generativelanguage.googleapis.com)
                 Free tier, 30 expressive voices, 70+ languages.
                 Keys: GEMINI_API_KEY  or  GOOGLE_API_KEY (from aistudio.google.com)

  CLOUD mode   — Google Cloud TTS (texttospeech.googleapis.com)
                 700+ voices, SSML, speaking rate / pitch control.
                 Key:  GOOGLE_APPLICATION_CREDENTIALS (service account JSON)
                       or GOOGLE_API_KEY from Google Cloud Console

Key priority:
  1. GEMINI_API_KEY        → Gemini mode
  2. GOOGLE_APPLICATION_CREDENTIALS → Cloud mode
  3. GOOGLE_API_KEY        → Gemini mode (AI Studio key per .env.example)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,               # PCM fmt subchunk size
        1,                # audio format: PCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        sample_width * 8,
        b"data",
        data_size,
    )
    return header + pcm_data


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class GoogleTTS(BaseTool):
    name = "google_tts"
    version = "0.3.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "google_tts"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.API

    dependencies = []
    install_instructions = (
        "Gemini mode (free tier, recommended):\n"
        "  Set GEMINI_API_KEY or GOOGLE_API_KEY to your Google AI Studio key.\n"
        "  Get one free at https://aistudio.google.com/apikey\n"
        "\n"
        "Cloud mode (700+ voices, SSML, speaking rate/pitch):\n"
        "  Set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON path,\n"
        "  or set GOOGLE_API_KEY to a Cloud Console key with TTS enabled.\n"
        "  Enable at https://console.cloud.google.com/apis/library/texttospeech.googleapis.com"
    )
    fallback = "openai_tts"
    fallback_tools = ["openai_tts", "elevenlabs_tts", "piper_tts"]
    agent_skills = ["text-to-speech"]

    capabilities = [
        "text_to_speech",
        "voice_selection",
        "ssml_support",       # Cloud mode only
        "multilingual",
        "audio_style_tags",   # Gemini mode only
    ]
    supports = {
        "voice_cloning": False,
        "multilingual": True,
        "offline": False,
        "native_audio": True,
        "ssml": True,         # Cloud mode; Gemini uses inline audio tags instead
        "audio_tags": True,   # Gemini mode: [whispers], [cheerfully], etc.
    }
    best_for = [
        "Gemini mode: free-tier TTS, 30 expressive voices, auto language detection",
        "Cloud mode: 700+ voices across 50+ languages, SSML, speaking rate / pitch",
    ]
    not_good_for = [
        "voice cloning",
        "fully offline production",
    ]

    # Gemini AI Studio endpoint
    _GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
    _GEMINI_DEFAULT_MODEL = "gemini-2.5-flash-preview-tts"
    _GEMINI_DEFAULT_VOICE = "Puck"

    # Cloud TTS endpoint
    _CLOUD_BASE = "https://texttospeech.googleapis.com"
    _CLOUD_BETA_VOICE_PREFIXES = ("Chirp", "Journey")

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {
                "type": "string",
                "description": (
                    "Text to synthesize. In Gemini mode supports inline audio tags "
                    "such as [whispers], [cheerfully], [slowly]. "
                    "In Cloud mode supports SSML markup."
                ),
            },
            # --- Gemini mode params ---
            "voice": {
                "type": "string",
                "description": (
                    "Voice name. "
                    "Gemini voices: Puck (default), Zephyr, Aoede, Orus, Kore, Fenrir, Charon, "
                    "Leda, Callirrhoe, Autonoe, Enceladus, Iapetus, Umbriel, Algieba, Despina, "
                    "Erinome, Algenib, Rasalgethi, Laomedeia, Achernar, Alnilam, Schedar, "
                    "Gacrux, Pulcherrima, Achird, Zubenelgenubi, Vindemiatrix, Sadachbia, "
                    "Sadaltager, Sulafat. "
                    "Cloud voices: en-US-Chirp3-HD-Orus, en-US-Neural2-D, en-US-Studio-O, etc."
                ),
            },
            "model": {
                "type": "string",
                "default": _GEMINI_DEFAULT_MODEL,
                "description": "Gemini mode only. TTS model ID (default: gemini-2.5-flash-preview-tts).",
            },
            # --- Cloud mode params ---
            "language_code": {
                "type": "string",
                "default": "en-US",
                "description": "Cloud mode only. BCP-47 language code (e.g. en-US, es-ES, ja-JP).",
            },
            "speaking_rate": {
                "type": "number",
                "default": 1.0,
                "minimum": 0.25,
                "maximum": 4.0,
                "description": "Cloud mode only. Speaking speed; 1.0 = normal.",
            },
            "pitch": {
                "type": "number",
                "default": 0.0,
                "minimum": -20.0,
                "maximum": 20.0,
                "description": "Cloud mode only. Pitch adjustment in semitones.",
            },
            "audio_encoding": {
                "type": "string",
                "default": "MP3",
                "enum": ["MP3", "LINEAR16", "OGG_OPUS", "MULAW", "ALAW"],
                "description": "Cloud mode only. Output encoding. Gemini always outputs WAV.",
            },
            # --- Shared ---
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=50, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["text", "voice", "language_code", "speaking_rate", "pitch"]
    side_effects = ["writes audio file to output_path", "calls Google TTS API"]
    user_visible_verification = ["Listen to generated audio for natural speech quality"]

    # ------------------------------------------------------------------
    # Credential / mode detection
    # ------------------------------------------------------------------

    def _detect_mode(self) -> tuple[str | None, str | None]:
        """Return (mode, api_key_or_None).

        mode is 'gemini', 'cloud', or None (unavailable).
        For Cloud service-account auth, api_key is None (credentials file is used directly).
        """
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if gemini_key:
            return "gemini", gemini_key

        cloud_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if cloud_creds:
            return "cloud", None  # google-auth reads the file automatically

        google_key = os.environ.get("GOOGLE_API_KEY")
        if google_key:
            # Per .env.example, GOOGLE_API_KEY comes from aistudio.google.com → Gemini mode
            return "gemini", google_key

        return None, None

    def get_status(self) -> ToolStatus:
        mode, _ = self._detect_mode()
        return ToolStatus.AVAILABLE if mode else ToolStatus.UNAVAILABLE

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        text = inputs.get("text", "")
        mode, _ = self._detect_mode()
        if mode == "gemini":
            # Gemini 2.5 Flash TTS: ~$0.50/1M input tokens; ~4 chars per token
            tokens = len(text) / 4
            return round(tokens * 0.0000005, 6)
        # Cloud pricing per million characters
        voice = inputs.get("voice", "en-US-Chirp3-HD-Orus")
        char_count = len(text)
        if "Chirp3-HD" in voice:
            rate = 0.000030
        elif "Studio" in voice:
            rate = 0.000160
        elif "Neural2" in voice or "Journey" in voice:
            rate = 0.000016
        elif "WaveNet" in voice:
            rate = 0.000016
        else:
            rate = 0.000004
        return round(char_count * rate, 4)

    # ------------------------------------------------------------------
    # Execution entry point
    # ------------------------------------------------------------------

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        mode, api_key = self._detect_mode()
        if not mode:
            return ToolResult(
                success=False,
                error="No Google credentials found.\n" + self.install_instructions,
            )

        start = time.time()
        try:
            if mode == "gemini":
                result = self._generate_gemini(inputs, api_key)
            else:
                result = self._generate_cloud(inputs, api_key)
        except Exception as exc:
            return ToolResult(success=False, error=f"Google TTS ({mode} mode) failed: {exc}")

        result.duration_seconds = round(time.time() - start, 2)
        result.cost_usd = self.estimate_cost(inputs)
        return result

    # ------------------------------------------------------------------
    # Gemini AI Studio path
    # ------------------------------------------------------------------

    def _generate_gemini(self, inputs: dict[str, Any], api_key: str) -> ToolResult:
        import requests

        text = inputs["text"]
        voice_name = inputs.get("voice", self._GEMINI_DEFAULT_VOICE)
        model = inputs.get("model", self._GEMINI_DEFAULT_MODEL)

        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": voice_name}
                    }
                },
            },
        }

        url = f"{self._GEMINI_BASE}/{model}:generateContent"
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

        inline = response.json()["candidates"][0]["content"]["parts"][0]["inlineData"]
        mime_type: str = inline["mimeType"]  # e.g. "audio/L16;codec=pcm;rate=24000"
        pcm_bytes = base64.b64decode(inline["data"])

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
                "mode": "gemini",
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

    # ------------------------------------------------------------------
    # Google Cloud TTS path
    # ------------------------------------------------------------------

    _EXT_MAP = {
        "MP3": "mp3",
        "LINEAR16": "wav",
        "OGG_OPUS": "ogg",
        "MULAW": "wav",
        "ALAW": "wav",
    }

    def _needs_beta_api(self, voice: str) -> bool:
        return any(prefix in voice for prefix in self._CLOUD_BETA_VOICE_PREFIXES)

    def _generate_cloud(self, inputs: dict[str, Any], api_key: str | None) -> ToolResult:
        import requests

        text = inputs["text"]
        voice_name = inputs.get("voice", "en-US-Chirp3-HD-Orus")
        language_code = inputs.get("language_code", "en-US")
        speaking_rate = inputs.get("speaking_rate", 1.0)
        pitch = inputs.get("pitch", 0.0)
        audio_encoding = inputs.get("audio_encoding", "MP3")

        payload = {
            "input": {"text": text},
            "voice": {"languageCode": language_code, "name": voice_name},
            "audioConfig": {
                "audioEncoding": audio_encoding,
                "speakingRate": speaking_rate,
                "pitch": pitch,
            },
        }

        api_version = "v1beta1" if self._needs_beta_api(voice_name) else "v1"
        url = f"{self._CLOUD_BASE}/{api_version}/text:synthesize"

        headers = {"Content-Type": "application/json"}
        params = {}
        if api_key:
            params["key"] = api_key
        # Service-account auth: let google-auth add the Authorization header
        elif os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            try:
                import google.auth
                import google.auth.transport.requests as ga_requests
                creds, _ = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                creds.refresh(ga_requests.Request())
                headers["Authorization"] = f"Bearer {creds.token}"
            except Exception as exc:
                return ToolResult(
                    success=False,
                    error=f"Service account auth failed: {exc}. Install google-auth: pip install google-auth",
                )

        response = requests.post(url, headers=headers, params=params, json=payload, timeout=120)
        response.raise_for_status()

        audio_content = base64.b64decode(response.json()["audioContent"])
        ext = self._EXT_MAP.get(audio_encoding, "mp3")
        output_path = Path(inputs.get("output_path", f"tts_output.{ext}"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(audio_content)

        return ToolResult(
            success=True,
            data={
                "mode": "cloud",
                "provider": self.provider,
                "voice": voice_name,
                "language_code": language_code,
                "text_length": len(text),
                "output": str(output_path),
                "format": audio_encoding,
                "speaking_rate": speaking_rate,
                "pitch": pitch,
            },
            artifacts=[str(output_path)],
            model=f"google-cloud-tts/{voice_name}",
        )
