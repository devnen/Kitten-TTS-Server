# File: models.py
# Pydantic models for API request and response validation.

from typing import Optional, Literal
from pydantic import BaseModel, Field


class GenerationParams(BaseModel):
    """Common parameters for TTS generation."""

    speed: Optional[float] = Field(
        None,
        ge=0.25,
        le=4.0,
        description="Speed factor for the generated audio. 1.0 is normal speed.",
    )
    language: Optional[str] = Field(
        None,
        description="Language of the text. (Primarily for UI, actual engine may infer)",
    )


class CustomTTSRequest(BaseModel):
    """Request model for the custom /tts endpoint."""

    text: str = Field(..., min_length=1, description="Text to be synthesized.")

    voice: str = Field(
        ...,
        description="Voice identifier (e.g., 'expr-voice-5-m'). Available voices: expr-voice-2-m, expr-voice-2-f, expr-voice-3-m, expr-voice-3-f, expr-voice-4-m, expr-voice-4-f, expr-voice-5-m, expr-voice-5-f",
    )

    output_format: Optional[Literal["wav", "opus", "mp3"]] = Field(
        "wav", description="Desired audio output format."
    )

    split_text: Optional[bool] = Field(
        True,
        description="Whether to automatically split long text into chunks for processing.",
    )
    chunk_size: Optional[int] = Field(
        120,
        ge=50,
        le=500,
        description="Approximate target character length for text chunks when splitting is enabled (50-500).",
    )

    # Embed generation parameters directly
    speed: Optional[float] = Field(
        None, description="Overrides default speed if provided."
    )
    language: Optional[str] = Field(
        None, description="Overrides default language if provided."
    )

    stream: Optional[bool] = Field(
        False,
        description=(
            "When true, stream each synthesized text chunk immediately after encoding. "
            "Compatibility mode (false) buffers and merges all chunks into one downloadable file. "
            "For WAV streaming, each yielded chunk is a standalone WAV file with its own header; "
            "clients expecting one continuous WAV file may only play the first chunk or reject the stream."
        ),
    )


class ErrorResponse(BaseModel):
    """Standard error response model for API errors."""

    detail: str = Field(..., description="A human-readable explanation of the error.")


class UpdateStatusResponse(BaseModel):
    """Response model for status updates, e.g., after saving settings."""

    message: str = Field(
        ..., description="A message describing the result of the operation."
    )
    restart_needed: Optional[bool] = Field(
        False,
        description="Indicates if a server restart is recommended or required for changes to take full effect.",
    )
