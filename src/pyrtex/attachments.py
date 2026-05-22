# src/pyrtex/attachments.py

"""
Attachment handling: source URI parsing, mime type registry, and per-model
size limits.

A pyrtex attachment is always referenced by a source the caller's process can
read — a local ``pathlib.Path`` or a URI string (``s3://...``, ``gs://...``).
pyrtex resolves the source in the caller's environment and stages it into the
job's GCS bucket; Vertex AI only ever sees a ``gs://`` URI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union
from urllib.parse import urlparse

from .exceptions import ValidationError

logger = logging.getLogger(__name__)

AttachmentSource = Union[str, Path]


# Extension → mime type as accepted by Vertex AI's batch prediction endpoint.
# Anything not in this map is rejected at add_request time — no silent
# text/plain fallback for unknown types (Office docs, archives, etc.).
#
# Important: Vertex's batch endpoint accepts a narrow mime allowlist for the
# ``file_data.mime_type`` field. It rejects canonical-but-uncommon types like
# ``application/json``, ``application/xml``, ``text/csv``, ``text/html``,
# ``application/javascript``, etc. with an "mimeType ... is not supported"
# error. All text-like extensions therefore map to ``text/plain`` — the
# generic textual type the endpoint reliably accepts. The model still parses
# the content correctly because it sees the raw bytes.
SUPPORTED_MIME_TYPES: Dict[str, str] = {
    # Plain text family — Vertex only accepts text/plain for text formats.
    ".txt": "text/plain",
    ".md": "text/plain",
    ".rst": "text/plain",
    ".log": "text/plain",
    ".csv": "text/plain",
    ".tsv": "text/plain",
    ".json": "text/plain",
    ".xml": "text/plain",
    ".yaml": "text/plain",
    ".yml": "text/plain",
    ".html": "text/plain",
    ".htm": "text/plain",
    ".py": "text/plain",
    ".js": "text/plain",
    ".css": "text/plain",
    ".sql": "text/plain",
    ".ini": "text/plain",
    ".cfg": "text/plain",
    ".conf": "text/plain",
    ".sh": "text/plain",
    # Documents
    ".pdf": "application/pdf",
    # Images
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".heic": "image/heic",
    ".heif": "image/heif",
    # Audio
    ".mp3": "audio/mp3",
    ".wav": "audio/wav",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".aiff": "audio/aiff",
    # Video
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".mov": "video/mov",
    ".avi": "video/avi",
    ".webm": "video/webm",
    ".wmv": "video/wmv",
    ".flv": "video/flv",
    ".3gp": "video/3gpp",
    ".3gpp": "video/3gpp",
}


@dataclass(frozen=True)
class ModelLimits:
    """Best-effort upper bounds for a model's attachment handling."""

    max_file_bytes: int
    max_files_per_request: int


# Snapshot of Gemini batch-prediction limits. Treat as a soft early guard,
# not a contract — Google can adjust these and the registry will drift.
# Users can override with InfrastructureConfig.skip_size_checks.
_DEFAULT_LIMITS = ModelLimits(
    max_file_bytes=2 * 1024 * 1024 * 1024,  # 2 GiB per file (File API ceiling)
    max_files_per_request=3000,
)

MODEL_LIMITS: Dict[str, ModelLimits] = {
    "gemini-2.0-flash": _DEFAULT_LIMITS,
    "gemini-2.0-flash-lite-001": _DEFAULT_LIMITS,
    "gemini-2.0-flash-001": _DEFAULT_LIMITS,
    "gemini-2.5-flash": _DEFAULT_LIMITS,
    "gemini-2.5-pro": _DEFAULT_LIMITS,
}


def get_model_limits(model: str) -> Optional[ModelLimits]:
    """Returns limits for a model, or None if the model isn't in the registry."""
    if model in MODEL_LIMITS:
        return MODEL_LIMITS[model]
    # Match by prefix so versioned variants (e.g. "gemini-2.5-pro-preview-...")
    # inherit from their family.
    for prefix, limits in MODEL_LIMITS.items():
        if model.startswith(prefix):
            return limits
    return None


def parse_source(source: AttachmentSource) -> tuple[str, str]:
    """
    Classifies an attachment source.

    Returns ``(scheme, identifier)`` where scheme is one of ``file``, ``s3``,
    ``gs``, and identifier is the path/URI suitable for the scheme's handler.
    Raises ``ValidationError`` for unsupported schemes.
    """
    if isinstance(source, Path):
        return "file", str(source)

    if not isinstance(source, str):
        raise ValidationError(
            f"Attachment must be a pathlib.Path or URI string, got {type(source).__name__}."
        )

    parsed = urlparse(source)
    scheme = parsed.scheme.lower()

    if scheme in ("", "file"):
        return "file", parsed.path if scheme == "file" else source
    if scheme == "s3":
        if not parsed.netloc or not parsed.path.lstrip("/"):
            raise ValidationError(
                f"Invalid s3 URI '{source}': expected s3://bucket/key."
            )
        return "s3", source
    if scheme == "gs":
        if not parsed.netloc or not parsed.path.lstrip("/"):
            raise ValidationError(
                f"Invalid gs URI '{source}': expected gs://bucket/object."
            )
        return "gs", source

    raise ValidationError(
        f"Unsupported attachment scheme '{scheme}' in '{source}'. "
        f"Supported: local Path, s3://, gs://."
    )


def mime_type_for(source: AttachmentSource) -> str:
    """
    Resolves the mime type for a source by file extension. Raises
    ``ValidationError`` if the extension is unknown — pyrtex never silently
    falls back to text/plain.
    """
    if isinstance(source, Path):
        suffix = source.suffix.lower()
        name = source.name
    else:
        parsed = urlparse(source)
        path = parsed.path if parsed.scheme else source
        suffix = Path(path).suffix.lower()
        name = path

    if not suffix:
        raise ValidationError(
            f"Attachment '{name}' has no file extension; cannot determine mime type. "
            f"Rename the file with a supported extension."
        )

    mime = SUPPORTED_MIME_TYPES.get(suffix)
    if mime is None:
        raise ValidationError(
            f"Attachment '{name}' has unsupported extension '{suffix}'. "
            f"Gemini does not natively process this file type — convert it "
            f"(e.g. Office docs → PDF) before attaching. "
            f"Supported extensions: {sorted(SUPPORTED_MIME_TYPES.keys())}."
        )
    return mime
