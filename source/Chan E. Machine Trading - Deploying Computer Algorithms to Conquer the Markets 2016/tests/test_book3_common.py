from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from book3_common import chapter_manifest, safe_project_path, sha256_file  # noqa: E402


def test_parameterized_manifest_and_lock_hash() -> None:
    manifest = chapter_manifest(PROJECT_ROOT / "pyproject.toml", chapter=1)
    assert manifest["title"] == "Chap1 Basics"
    assert len(manifest["sha256"]) == 64
    assert len(sha256_file(PROJECT_ROOT / "uv.lock")) == 64


def test_safe_project_path_rejects_escape() -> None:
    expected = PROJECT_ROOT / "data/raw/book3/chapter_1/example.mat"
    assert safe_project_path(
        PROJECT_ROOT, "data/raw/book3/chapter_1/example.mat"
    ) == expected.resolve()
    with pytest.raises(ValueError, match="escapes project root"):
        safe_project_path(PROJECT_ROOT, "../outside.txt")
