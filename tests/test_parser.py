"""Tests for parser.py."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from parser import CodeChunk, DocSection, parse_repo


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "docs").mkdir()

    (tmp_path / "src" / "service.py").write_text(
        textwrap.dedent(
            '''
            """Service module."""

            def greet(name: str) -> str:
                """Return a greeting."""
                return f"Hello, {name}"


            class Greeter:
                """Greets people."""

                def hello(self, name: str) -> str:
                    """Say hello."""
                    return greet(name)
            '''
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    (tmp_path / "docs" / "guide.md").write_text(
        textwrap.dedent(
            """
            # Guide

            Welcome to the project.

            ## Usage

            Call `greet()` to say hi.

            ### Examples

            ```python
            greet("world")
            ```
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    return tmp_path


def test_parse_python_functions_and_methods(sample_repo: Path) -> None:
    code_chunks, _ = parse_repo(sample_repo)

    keys = {chunk.key for chunk in code_chunks}
    assert "src/service.py::greet" in keys
    assert "src/service.py::Greeter.hello" in keys

    greet = next(c for c in code_chunks if c.name == "greet")
    assert greet.docstring == "Return a greeting."
    assert "name: str" in greet.signature
    assert greet.parent is None

    hello = next(c for c in code_chunks if c.name == "hello")
    assert hello.parent == "Greeter"


def test_parse_markdown_sections(sample_repo: Path) -> None:
    _, doc_sections = parse_repo(sample_repo)

    keys = {section.key for section in doc_sections}
    assert "docs/guide.md::Guide" in keys
    assert "docs/guide.md::Guide::Usage" in keys
    assert "docs/guide.md::Guide::Usage::Examples" in keys

    usage = next(s for s in doc_sections if s.heading_path == ["Guide", "Usage"])
    assert "Call `greet()`" in usage.body
    assert usage.heading_level == 2


def test_models_are_pydantic_v2() -> None:
    chunk = CodeChunk(
        key="a.py::f",
        filepath="a.py",
        name="f",
        signature="def f() -> None",
        lineno=1,
    )
    section = DocSection(
        key="README.md::Intro",
        filepath="README.md",
        heading_path=["Intro"],
        heading_level=1,
        body="Hello",
        lineno=1,
    )
    assert chunk.model_dump()["name"] == "f"
    assert section.model_dump()["heading_path"] == ["Intro"]
