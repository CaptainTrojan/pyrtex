# tests/unit/test_attachment_validation.py

"""
Tests covering _attachment_size_bytes, _validate_attachments, and the
_attachment_filename helper for non-Path URI inputs.
"""

import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

from pyrtex.attachments import ModelLimits
from pyrtex.client import Job, _attachment_filename
from pyrtex.exceptions import ValidationError
from tests.conftest import SimpleOutput


def _make_job(model="gemini-2.0-flash-lite-001"):
    return Job(
        model=model,
        output_schema=SimpleOutput,
        prompt_template="Process: {{ word }}",
    )


class TestAttachmentFilenameHelper:
    def test_path_object(self):
        assert _attachment_filename(Path("/tmp/foo.pdf")) == "foo.pdf"

    def test_s3_uri(self):
        assert _attachment_filename("s3://bucket/dir/file.pdf") == "file.pdf"

    def test_gs_uri(self):
        assert _attachment_filename("gs://bucket/x/y.png") == "y.png"

    def test_string_path_without_scheme(self):
        assert _attachment_filename("/tmp/foo.pdf") == "foo.pdf"


class TestAttachmentSizeBytes:
    def test_local_file(self, mock_gcp_clients, tmp_path):
        job = _make_job()
        f = tmp_path / "a.pdf"
        f.write_bytes(b"hello world")
        assert job._attachment_size_bytes(f) == 11

    def test_local_missing_file_raises(self, mock_gcp_clients, tmp_path):
        job = _make_job()
        missing = tmp_path / "nope.pdf"
        with pytest.raises(ValidationError, match="does not exist"):
            job._attachment_size_bytes(missing)

    def test_s3_without_boto3_returns_none(self, mock_gcp_clients, mocker):
        job = _make_job()
        mocker.patch.dict(sys.modules, {"boto3": None})
        assert job._attachment_size_bytes("s3://b/k.pdf") is None

    def test_s3_head_object_returns_size(self, mock_gcp_clients, mocker):
        fake_boto3 = types.ModuleType("boto3")
        fake_client = Mock()
        fake_client.head_object.return_value = {"ContentLength": 4242}
        fake_boto3.client = Mock(return_value=fake_client)
        mocker.patch.dict(sys.modules, {"boto3": fake_boto3})

        job = _make_job()
        assert job._attachment_size_bytes("s3://b/key.pdf") == 4242
        fake_client.head_object.assert_called_once_with(Bucket="b", Key="key.pdf")

    def test_s3_head_object_failure_raises(self, mock_gcp_clients, mocker):
        fake_boto3 = types.ModuleType("boto3")
        fake_client = Mock()
        fake_client.head_object.side_effect = RuntimeError("no perms")
        fake_boto3.client = Mock(return_value=fake_client)
        mocker.patch.dict(sys.modules, {"boto3": fake_boto3})

        job = _make_job()
        with pytest.raises(ValidationError, match="Failed to HEAD S3 object"):
            job._attachment_size_bytes("s3://b/key.pdf")

    def test_gs_returns_blob_size(self, mock_gcp_clients):
        job = _make_job()
        fake_blob = Mock()
        fake_blob.size = 999
        mock_gcp_clients["storage"].bucket.return_value.get_blob.return_value = (
            fake_blob
        )
        assert job._attachment_size_bytes("gs://b/file.pdf") == 999

    def test_gs_blob_missing_raises(self, mock_gcp_clients):
        job = _make_job()
        mock_gcp_clients["storage"].bucket.return_value.get_blob.return_value = None
        with pytest.raises(
            ValidationError, match="does not exist or is not accessible"
        ):
            job._attachment_size_bytes("gs://b/missing.pdf")

    def test_gs_blob_with_none_size_returns_zero(self, mock_gcp_clients):
        """Defensive: GCS occasionally returns blobs with size=None."""
        job = _make_job()
        fake_blob = Mock()
        fake_blob.size = None
        mock_gcp_clients["storage"].bucket.return_value.get_blob.return_value = (
            fake_blob
        )
        assert job._attachment_size_bytes("gs://b/f.pdf") == 0


class TestValidateAttachments:
    def test_too_many_attachments_raises(self, mock_gcp_clients, mocker, tmp_path):
        # Squeeze the model down to a max of 1 to exercise the count check.
        mocker.patch(
            "pyrtex.client.get_model_limits",
            return_value=ModelLimits(max_file_bytes=1_000_000, max_files_per_request=1),
        )
        job = _make_job()
        f1 = tmp_path / "a.pdf"
        f1.write_bytes(b"x")
        f2 = tmp_path / "b.pdf"
        f2.write_bytes(b"x")
        with pytest.raises(ValidationError, match="exceeding the limit"):
            job.add_request("k", attachments=[f1, f2])

    def test_oversized_file_raises(self, mock_gcp_clients, mocker, tmp_path):
        # Tight per-file cap so a small fixture trips it.
        mocker.patch(
            "pyrtex.client.get_model_limits",
            return_value=ModelLimits(max_file_bytes=4, max_files_per_request=10),
        )
        job = _make_job()
        f = tmp_path / "big.pdf"
        f.write_bytes(b"more than four bytes")
        with pytest.raises(ValidationError, match="exceeding the per-file limit"):
            job.add_request("k", attachments=[f])

    def test_unknown_model_skips_validation(self, mock_gcp_clients, caplog, tmp_path):
        """No limits entry → warn, skip size checks, still allow add."""
        import logging

        f = tmp_path / "x.pdf"
        f.write_bytes(b"x" * 10)

        job = Job(
            model="totally-unknown-model-xyz",
            output_schema=SimpleOutput,
            prompt_template="Process: {{ word }}",
        )
        with caplog.at_level(logging.WARNING, logger="pyrtex.client"):
            job.add_request("k", attachments=[f])
        assert any(
            "No size-limit registry entry" in rec.message for rec in caplog.records
        )

    def test_size_lookup_none_is_skipped(self, mock_gcp_clients, mocker):
        """When attachment size cannot be determined cheaply, skip the check."""
        mocker.patch(
            "pyrtex.client.get_model_limits",
            return_value=ModelLimits(max_file_bytes=10, max_files_per_request=10),
        )
        # Force size lookup to return None for the s3 source.
        mocker.patch.object(Job, "_attachment_size_bytes", return_value=None)
        job = _make_job()
        # No exception — the size cap is silently skipped.
        job.add_request("k", attachments=["s3://bucket/file.pdf"])
        assert len(job._requests) == 1
