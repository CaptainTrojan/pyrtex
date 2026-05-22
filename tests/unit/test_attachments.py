# tests/unit/test_attachments.py

"""Tests for the attachments module: source parsing, mime registry, limits."""

from pathlib import Path

import pytest

from pyrtex.attachments import (
    SUPPORTED_MIME_TYPES,
    get_model_limits,
    mime_type_for,
    parse_source,
)
from pyrtex.exceptions import ValidationError


class TestParseSource:
    def test_path_object_is_file(self):
        scheme, ident = parse_source(Path("/tmp/foo.pdf"))
        assert scheme == "file"
        assert ident == "/tmp/foo.pdf"

    def test_local_string_path_is_file(self):
        scheme, ident = parse_source("/tmp/foo.pdf")
        assert scheme == "file"
        assert ident == "/tmp/foo.pdf"

    def test_s3_uri(self):
        scheme, ident = parse_source("s3://my-bucket/path/to/file.pdf")
        assert scheme == "s3"
        assert ident == "s3://my-bucket/path/to/file.pdf"

    def test_gs_uri(self):
        scheme, ident = parse_source("gs://my-bucket/path/file.pdf")
        assert scheme == "gs"
        assert ident == "gs://my-bucket/path/file.pdf"

    def test_unsupported_scheme_raises(self):
        with pytest.raises(ValidationError, match="Unsupported attachment scheme"):
            parse_source("ftp://example.com/file.pdf")

    def test_https_unsupported(self):
        with pytest.raises(ValidationError, match="Unsupported attachment scheme"):
            parse_source("https://example.com/file.pdf")

    def test_s3_without_key_raises(self):
        with pytest.raises(ValidationError, match="Invalid s3 URI"):
            parse_source("s3://bucket-only")

    def test_gs_without_object_raises(self):
        with pytest.raises(ValidationError, match="Invalid gs URI"):
            parse_source("gs://bucket-only")

    def test_explicit_file_scheme(self):
        scheme, ident = parse_source("file:///tmp/foo.pdf")
        assert scheme == "file"
        assert ident == "/tmp/foo.pdf"

    def test_non_string_non_path_raises(self):
        with pytest.raises(ValidationError, match="Path or URI string"):
            parse_source(123)  # type: ignore[arg-type]


class TestMimeTypeFor:
    def test_known_extensions(self):
        assert mime_type_for(Path("doc.pdf")) == "application/pdf"
        assert mime_type_for(Path("image.png")) == "image/png"
        # Vertex batch only accepts text/plain for textual formats.
        assert mime_type_for("s3://b/data.csv") == "text/plain"
        assert mime_type_for("gs://b/clip.mp4") == "video/mp4"

    def test_case_insensitive(self):
        assert mime_type_for(Path("DOC.PDF")) == "application/pdf"

    def test_missing_extension_raises(self):
        with pytest.raises(ValidationError, match="no file extension"):
            mime_type_for(Path("/tmp/README"))

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValidationError, match="unsupported extension"):
            mime_type_for(Path("/tmp/report.docx"))

    def test_registry_contains_pdf_image_audio_video(self):
        # Quick sanity on registry coverage.
        assert ".pdf" in SUPPORTED_MIME_TYPES
        assert ".png" in SUPPORTED_MIME_TYPES
        assert ".mp3" in SUPPORTED_MIME_TYPES
        assert ".mp4" in SUPPORTED_MIME_TYPES


class TestModelLimits:
    def test_known_model_returns_limits(self):
        limits = get_model_limits("gemini-2.0-flash-lite-001")
        assert limits is not None
        assert limits.max_file_bytes > 0
        assert limits.max_files_per_request > 0

    def test_prefix_match_for_versioned_variant(self):
        limits = get_model_limits("gemini-2.5-pro-preview-1234")
        assert limits is not None

    def test_unknown_model_returns_none(self):
        assert get_model_limits("some-totally-unknown-model") is None
