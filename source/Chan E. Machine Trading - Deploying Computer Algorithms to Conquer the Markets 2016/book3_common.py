"""Reusable provenance and notebook helpers for Machine Trading chapters.

The helpers are deliberately chapter-agnostic so later chapters can share the
same checksum, safe-path, idempotent-write, and notebook-execution contracts.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import re
import tempfile
import tomllib
import urllib.request
import zipfile
from importlib.metadata import version
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterable

import nbformat
from nbclient import NotebookClient


DEFAULT_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "nbformat",
    "nbclient",
)

LATEX_COMMAND_WORDS = {
    "Delta", "Phi", "alpha", "approx", "beta", "cdots", "epsilon",
    "frac", "gamma", "geq", "left", "log", "mathbf", "mathrm", "min",
    "mu", "operatorname", "phi", "qquad", "quad", "right", "sigma",
    "sum", "tau", "text", "theta", "times", "to", "top",
}


def assert_clean_markdown_math(text: str, label: str) -> None:
    """Reject control characters and consumed LaTeX backslashes in Markdown."""
    controls = [
        (index, ord(character))
        for index, character in enumerate(text)
        if ord(character) < 32 and character != "\n"
    ]
    if controls:
        raise AssertionError(f"{label} contains control characters: {controls[:5]}")
    bare_commands: list[str] = []
    for math_match in re.finditer("[$]{1,2}(.*?)[$]{1,2}", text, flags=re.DOTALL):
        span = math_match.group(1)
        for word_match in re.finditer("[A-Za-z]+", span):
            word = word_match.group(0)
            if word not in LATEX_COMMAND_WORDS:
                continue
            if word_match.start() == 0 or span[word_match.start() - 1] != chr(92):
                bare_commands.append(word)
    if bare_commands:
        raise AssertionError(
            f"{label} contains bare LaTeX command words: {sorted(set(bare_commands))}"
        )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_versions(
    packages: Iterable[str] = DEFAULT_PACKAGES,
) -> dict[str, str]:
    return {
        "python": platform.python_version(),
        **{package: version(package) for package in packages},
    }


def write_text_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
        handle.write(payload)
    os.replace(temporary_path, path)


def chapter_manifest(pyproject_path: Path, chapter: int | str) -> dict[str, Any]:
    """Load one parameterized Book 3 chapter manifest."""
    key = str(chapter)
    if not key.startswith("chapter_"):
        key = f"chapter_{key}"
    with pyproject_path.open("rb") as handle:
        manifests = tomllib.load(handle)["tool"]["book3"]
    if key not in manifests:
        raise KeyError(f"Book 3 manifest not found: {key}")
    return manifests[key]


def safe_project_path(project_root: Path, relative_path: str) -> Path:
    root = project_root.resolve()
    destination = (root / relative_path).resolve()
    if not destination.is_relative_to(root):
        raise ValueError(f"Destination escapes project root: {relative_path}")
    return destination


def download_verified_archive(
    manifest: dict[str, Any],
    *,
    user_agent: str = "chan-book-experiments/0.1",
    timeout: int = 60,
) -> bytes:
    request = urllib.request.Request(
        manifest["url"], headers={"User-Agent": user_agent}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    if len(payload) != manifest["size_bytes"]:
        raise ValueError(
            f"Archive size mismatch: expected {manifest['size_bytes']}, "
            f"got {len(payload)}"
        )
    if sha256_bytes(payload) != manifest["sha256"]:
        raise ValueError("Official archive SHA-256 mismatch")
    with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
        bad_member = bundle.testzip()
        if bad_member is not None:
            raise zipfile.BadZipFile(f"CRC failure in {bad_member}")
    return payload


def materialize_verified_members(
    project_root: Path,
    manifest: dict[str, Any],
    archive_payload: bytes,
    *,
    force: bool = False,
) -> list[str]:
    statuses: list[str] = []
    with zipfile.ZipFile(io.BytesIO(archive_payload)) as bundle:
        for member in manifest["members"]:
            member_payload = bundle.read(member["archive_path"])
            if sha256_bytes(member_payload) != member["sha256"]:
                raise ValueError(
                    f"Member SHA-256 mismatch: {member['archive_path']}"
                )
            destination = safe_project_path(project_root, member["destination"])
            if destination.exists():
                if sha256_file(destination) == member["sha256"]:
                    statuses.append(
                        f"current  {destination.relative_to(project_root)}"
                    )
                    continue
                if not force:
                    raise FileExistsError(
                        "Refusing to replace modified file without "
                        f"--force-download: {destination}"
                    )
            atomic_write(destination, member_payload)
            statuses.append(f"written  {destination.relative_to(project_root)}")
    return statuses


def validate_manifest_members(
    project_root: Path, manifest: dict[str, Any]
) -> None:
    for member in manifest["members"]:
        destination = safe_project_path(project_root, member["destination"])
        if not destination.exists():
            raise FileNotFoundError(
                f"Missing official asset in offline mode: {destination}"
            )
        if sha256_file(destination) != member["sha256"]:
            raise ValueError(f"Checksum mismatch in offline mode: {destination}")


def materialize_chapter_archive(
    project_root: Path,
    manifest: dict[str, Any],
    archive_payload: bytes,
    *,
    force: bool = False,
) -> list[str]:
    """Extract a whole chapter ZIP into tracked sources and ignored raw data.

    The archive hash authenticates the complete bundle.  A generated, tracked
    extraction manifest then pins every individual source and raw-data member
    for later offline validation.
    """
    prefix = PurePosixPath(manifest["archive_prefix"])
    source_root = safe_project_path(project_root, manifest["source_root"])
    raw_root = safe_project_path(project_root, manifest["raw_root"])
    source_suffixes = {
        suffix.lower() for suffix in manifest.get("source_suffixes", [".m", ".fig"])
    }
    statuses: list[str] = []
    entries: list[dict[str, Any]] = []
    with zipfile.ZipFile(io.BytesIO(archive_payload)) as bundle:
        for info in bundle.infolist():
            member_path = PurePosixPath(info.filename)
            if info.is_dir():
                continue
            try:
                relative = member_path.relative_to(prefix)
            except ValueError as error:
                raise ValueError(
                    f"Archive member is outside declared prefix: {info.filename}"
                ) from error
            if not relative.parts or ".." in relative.parts:
                raise ValueError(f"Unsafe archive member: {info.filename}")
            is_source = relative.suffix.lower() in source_suffixes
            destination_root = source_root if is_source else raw_root
            destination = (destination_root / Path(*relative.parts)).resolve()
            if not destination.is_relative_to(destination_root.resolve()):
                raise ValueError(f"Archive destination escapes root: {info.filename}")
            payload = bundle.read(info)
            member_sha256 = sha256_bytes(payload)
            if destination.exists():
                if sha256_file(destination) == member_sha256:
                    status = "current"
                elif not force:
                    raise FileExistsError(
                        "Refusing to replace modified extracted file without "
                        f"--force-download: {destination}"
                    )
                else:
                    atomic_write(destination, payload)
                    status = "written"
            else:
                atomic_write(destination, payload)
                status = "written"
            relative_destination = destination.relative_to(project_root)
            statuses.append(f"{status:<8} {relative_destination}")
            entries.append(
                {
                    "archive_path": info.filename,
                    "destination": str(relative_destination),
                    "sha256": member_sha256,
                    "size_bytes": len(payload),
                    "kind": "original_source" if is_source else "research_data",
                }
            )

    extraction_manifest = {
        "source_page": manifest["source_page"],
        "archive_url": manifest["url"],
        "archive_sha256": manifest["sha256"],
        "archive_size_bytes": manifest["size_bytes"],
        "members": entries,
    }
    manifest_path = source_root / "SOURCE_MANIFEST.json"
    write_text_if_changed(
        manifest_path,
        json.dumps(extraction_manifest, indent=2, ensure_ascii=False) + "\n",
    )
    statuses.append(f"manifest {manifest_path.relative_to(project_root)}")
    return statuses


def validate_chapter_extraction(project_root: Path, manifest: dict[str, Any]) -> None:
    """Validate every extracted member without network access."""
    manifest_path = safe_project_path(
        project_root, f"{manifest['source_root']}/SOURCE_MANIFEST.json"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing extraction manifest: {manifest_path}")
    extraction = json.loads(manifest_path.read_text(encoding="utf-8"))
    if extraction["archive_sha256"] != manifest["sha256"]:
        raise ValueError("Extraction manifest archive SHA-256 mismatch")
    for member in extraction["members"]:
        destination = safe_project_path(project_root, member["destination"])
        if not destination.exists():
            raise FileNotFoundError(f"Missing extracted member: {destination}")
        if sha256_file(destination) != member["sha256"]:
            raise ValueError(f"Extracted member checksum mismatch: {destination}")


def execute_notebook(
    notebook: nbformat.NotebookNode,
    notebook_path: Path,
    *,
    workdir: Path,
    timeout: int = 300,
) -> None:
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        record_timing=False,
        resources={"metadata": {"path": str(workdir)}},
    )
    executed = client.execute()
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(executed, notebook_path)
