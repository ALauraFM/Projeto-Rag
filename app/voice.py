"""
Voice helpers: Speech-to-Text (STT) and Text-to-Speech (TTS) via the OpenAI API.
"""
from __future__ import annotations

import io

from openai import OpenAI

from app.config import settings

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def transcribe(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    """Transcribe audio bytes to text using OpenAI Whisper."""
    client = _get_client()
    buffer = io.BytesIO(audio_bytes)
    buffer.name = filename
    result = client.audio.transcriptions.create(
        model=settings.stt_model,
        file=buffer,
    )
    return result.text


def synthesize(text: str) -> bytes:
    """Convert text to speech (MP3 bytes) using OpenAI TTS."""
    client = _get_client()
    response = client.audio.speech.create(
        model=settings.tts_model,
        voice=settings.tts_voice,
        input=text,
    )
    return response.read()
